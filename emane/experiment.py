"""
emane/experiment.py — EMANE experiment runner and result analyser.

Experiment plan:
  1. Baseline Delivery:      Apply optimized G² schedule, measure delivery ratio.
  2. SINR Counterexample:   3-node line with u,w co-transmitting → packet loss at v.
  3. Predicted vs Measured:  Compare SINR auditor predictions to EMANE measurements.

Data collection uses EMANE's native statistics tables:
  - RxSlotStatusTable     : RX-during-TX, missed slots, RX counts
  - PacketDropTable       : per-drop reason counts
  - NeighborMetricTable   : SINR, NF, received power per-neighbour
  
NOTE: This module documents the approach. Full execution requires a running
EMANE environment (see Dockerfile). All table column names should be verified
against your EMANE version before relying on them — Sonnet's warning applies.
"""

import os
import json
import subprocess
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# EMANE shell interface (emanesh)
# ---------------------------------------------------------------------------

def run_emanesh(command: str, nem_id: Optional[int] = None) -> str:
    """
    Run an emanesh command and return its stdout.
    
    Example commands:
      get table 1 phy RxSlotStatusTable
      get table 1 phy NeighborMetricTable
      get table 1 mac PacketDropTable
    """
    if nem_id is not None:
        full_cmd = f"emanesh get table {nem_id} {command}"
    else:
        full_cmd = f"emanesh {command}"
    
    try:
        result = subprocess.run(
            full_cmd.split(),
            capture_output=True, text=True, timeout=10
        )
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return f"[emanesh error] {e}"


def publish_schedule(xml_path: str, nem_ids: List[int]) -> bool:
    """
    Publish a TDMA schedule XML to all specified NEMs.
    
    Uses EMANE's emaneevent-tdmaschedule publisher.
    All NEMs must receive their schedule before starting traffic.
    """
    cmd = ["emaneevent-tdmaschedule", xml_path]
    cmd += [str(nem_id) for nem_id in nem_ids]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"[EMANE] Schedule published to NEMs {nem_ids}")
            return True
        else:
            print(f"[EMANE] Schedule publish failed: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[EMANE] Error publishing schedule: {e}")
        return False


# ---------------------------------------------------------------------------
# Experiment 1 — Baseline Delivery
# ---------------------------------------------------------------------------

def experiment_baseline_delivery(
    nem_ids: List[int],
    packet_count: int = 1000,
) -> Dict:
    """
    Experiment 1: Measure delivery ratio under the optimised G² schedule.
    
    Expected outcome:
    - Links that the SINR auditor predicted DELIVER → high delivery ratio
    - Links that the SINR auditor predicted SINR_FAIL → reduced delivery ratio
    
    Data to collect:
      For each NEM: RxSlotStatusTable, NeighborMetricTable
    
    Returns dict of per-NEM statistics.
    """
    print("\n[Experiment 1] Baseline Delivery Measurement")
    print("=" * 50)
    
    results = {}
    for nem in nem_ids:
        # Read RX slot status
        rx_status = run_emanesh(f"phy RxSlotStatusTable", nem_id=nem)
        # Read neighbour metrics (SINR, received power)
        nbr_metrics = run_emanesh(f"phy NeighborMetricTable", nem_id=nem)
        
        results[f"NEM_{nem}"] = {
            "rx_slot_status": rx_status,
            "neighbour_metrics": nbr_metrics,
        }
        print(f"  NEM {nem}: collected RxSlotStatus + NeighbourMetrics")
    
    return results


# ---------------------------------------------------------------------------
# Experiment 2 — SINR Counterexample (3-node line)
# ---------------------------------------------------------------------------

def experiment_sinr_counterexample(
    u_nem: int = 1,  # NEM for u at 0m
    v_nem: int = 2,  # NEM for v at 450m (receiver)
    w_nem: int = 3,  # NEM for w at 970m
    slot_count: int = 2,
) -> Dict:
    """
    Experiment 2: Demonstrate the G²-safe-but-SINR-fail counterexample.
    
    Topology: u=0m, v=450m, w=970m (line)
    G² says u and w can share a slot.
    
    Sub-experiment 2a — G² schedule (u,w in same slot):
      - u in slot 0, v in slot 1, w in slot 0
      - Expected: u→v delivery degrades (SINR ≈ 1.26 dB < 10 dB threshold)
    
    Sub-experiment 2b — Fixed schedule (u,w in different slots):
      - u in slot 0, v in slot 1, w in slot 2
      - Expected: u→v delivery restored (no co-slot interference from w)
    
    Compare RxSlotStatusTable and NeighbourMetricTable between 2a and 2b.
    """
    print("\n[Experiment 2] SINR Counterexample — G² vs Physics")
    print("=" * 50)
    print(f"  Topology: u(NEM={u_nem}) at 0m, v(NEM={v_nem}) at 450m, "
          f"w(NEM={w_nem}) at 970m")
    print()
    
    results = {
        "topology": {"u_nem": u_nem, "v_nem": v_nem, "w_nem": w_nem},
        "prediction": {
            "g2_verdict": "SAFE — u and w can share a slot",
            "physics_verdict": "FAIL — SINR at v ≈ 1.26 dB < 10 dB",
            "interference_ratio": 3.162,
        },
        "sub_experiment_2a": {
            "description": "G² schedule: u and w in same slot",
            "schedule": {f"u(NEM {u_nem})": "slot 0",
                         f"v(NEM {v_nem})": "slot 1",
                         f"w(NEM {w_nem})": "slot 0"},
            "expected": "u→v delivery degraded",
            "measurement": "[run EMANE to collect]",
        },
        "sub_experiment_2b": {
            "description": "Fixed schedule: u and w in different slots",
            "schedule": {f"u(NEM {u_nem})": "slot 0",
                         f"v(NEM {v_nem})": "slot 1",
                         f"w(NEM {w_nem})": "slot 2"},
            "expected": "u→v delivery restored",
            "measurement": "[run EMANE to collect]",
        },
    }
    
    print("  Sub-experiment 2a (G² schedule):")
    print(f"    {results['sub_experiment_2a']['schedule']}")
    print(f"    Expected: {results['sub_experiment_2a']['expected']}")
    print()
    print("  Sub-experiment 2b (Fixed schedule):")
    print(f"    {results['sub_experiment_2b']['schedule']}")
    print(f"    Expected: {results['sub_experiment_2b']['expected']}")
    
    return results


# ---------------------------------------------------------------------------
# Experiment 3 — Predicted vs Measured Delivery Matrix
# ---------------------------------------------------------------------------

def experiment_predicted_vs_measured(
    audit_predictions: Dict,  # From sinr_auditor.audit_schedule()
    nem_ids: List[int],
    node_order: List[str],
) -> Dict:
    """
    Experiment 3: Compare SINR auditor predictions to EMANE measurements.
    
    For each link in the connectivity graph:
      1. Auditor prediction: DELIVER | SINR_FAIL | BELOW_SENSITIVITY
      2. EMANE measurement: actual delivery ratio from NeighbourMetricTable
    
    Output: side-by-side comparison matrix showing where the auditor
    is correct (the goal: it faithfully models EMANE's PHY).
    """
    print("\n[Experiment 3] Predicted vs Measured Delivery Matrix")
    print("=" * 60)
    
    node_to_nem = {name: idx + 1 for idx, name in enumerate(node_order)}
    
    comparison = []
    for pred in audit_predictions:
        tx_nem = node_to_nem.get(pred["tx_node"])
        rx_nem = node_to_nem.get(pred["rx_node"])
        
        if tx_nem is None or rx_nem is None:
            continue
        
        # Collect EMANE measurement for this link
        nbr_table = run_emanesh(f"phy NeighborMetricTable", nem_id=rx_nem)
        
        comparison.append({
            "tx": pred["tx_node"],
            "rx": pred["rx_node"],
            "slot": pred["slot"],
            "auditor_sinr_db": pred["sinr_db"],
            "auditor_prediction": pred["delivery_prediction"],
            "emane_measurement": nbr_table,  # Parse for this NEM pair
            "g2_said_safe": pred["g2_says_safe"],
        })
    
    # Print header
    print(f"{'TX':>10}  {'RX':>10}  {'SINR(pred)':>10}  "
          f"{'Auditor':>12}  {'G² Safe':>8}  {'EMANE':>10}")
    print("-" * 65)
    for row in comparison:
        print(
            f"{row['tx']:>10}  {row['rx']:>10}  "
            f"{row['auditor_sinr_db']:>10.1f}  "
            f"{row['auditor_prediction']:>12}  "
            f"{'Yes' if row['g2_said_safe'] else 'No':>8}  "
            f"[measure]"
        )
    
    return {"comparison": comparison}


# ---------------------------------------------------------------------------
# EMANE statistics parsing helpers
# ---------------------------------------------------------------------------

def parse_rx_slot_status(raw_output: str) -> Dict:
    """
    Parse emanesh RxSlotStatusTable output.
    
    NOTE: Column layout varies by EMANE version. Verify column names
    against your installed version before relying on this parser.
    Expected columns (from EMANE docs): RxSlot, Rx, RxIdleSlot, 
    RxBusy, RxDuringTx, MissedSlot
    """
    result = {}
    lines = [l.strip() for l in raw_output.strip().split("\n") if l.strip()]
    if not lines:
        return result
    
    # Simple key: value extraction (adapt to actual emanesh output format)
    for line in lines:
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def parse_neighbour_metrics(raw_output: str) -> List[Dict]:
    """
    Parse emanesh NeighborMetricTable output.
    
    NOTE: Verify actual column names in your EMANE version.
    Expected columns: NEM, SINR, NF, ReceivedPower, Timestamp
    """
    rows = []
    lines = [l.strip() for l in raw_output.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return rows
    
    headers = lines[0].split()
    for line in lines[1:]:
        values = line.split()
        if len(values) == len(headers):
            rows.append(dict(zip(headers, values)))
    return rows


# ---------------------------------------------------------------------------
# Main experiment orchestrator
# ---------------------------------------------------------------------------

def run_full_experiment_suite(
    schedule: Dict[str, int],
    node_order: List[str],
    audit_predictions: Optional[Dict] = None,
    output_dir: str = "emane_results",
) -> None:
    """
    Orchestrate all three experiments and save results.
    
    Pre-conditions (must be satisfied before calling):
      1. Docker container running with EMANE + 16 NEMs
      2. Schedule XML published to all NEMs
      3. Pathloss events published
      4. Application traffic running
    """
    os.makedirs(output_dir, exist_ok=True)
    nem_ids = list(range(1, len(node_order) + 1))
    
    print("\n" + "=" * 60)
    print(" EMANE EXPERIMENT SUITE")
    print("=" * 60)
    
    # Experiment 1
    e1 = experiment_baseline_delivery(nem_ids)
    with open(os.path.join(output_dir, "exp1_baseline.json"), "w") as f:
        json.dump(e1, f, indent=2)
    
    # Experiment 2 (always run — uses fixed 3-node setup)
    e2 = experiment_sinr_counterexample()
    with open(os.path.join(output_dir, "exp2_counterexample.json"), "w") as f:
        json.dump(e2, f, indent=2)
    
    # Experiment 3
    if audit_predictions:
        e3 = experiment_predicted_vs_measured(
            audit_predictions, nem_ids, node_order
        )
        with open(os.path.join(output_dir, "exp3_predicted_vs_measured.json"), "w") as f:
            json.dump(e3, f, indent=2)
    
    print(f"\n[EMANE] Results saved to {output_dir}/")


if __name__ == "__main__":
    print("EMANE experiment module loaded.")
    print("Run experiment_sinr_counterexample() to see the approach.")
    result = experiment_sinr_counterexample()
    print(json.dumps(result, indent=2))
