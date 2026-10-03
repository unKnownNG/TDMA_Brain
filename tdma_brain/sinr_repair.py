"""
sinr_repair.py — SINR-aware schedule repair loop.

This is the architectural bridge between Part 1 (graph theory) and Part 2
(physical-layer auditing).  Rather than just auditing and reporting, this
module closes the loop:

    G → G² → Exact Coloring → Candidate Schedule
                                     ↓
                              SINR Audit
                                     ↓
                         Find Physical Conflicts
                                     ↓
                        Augment Conflict Graph
                                     ↓
                              Re-color
                                     ↓
                         EMANE Validation

The contribution: a two-layer TDMA planner that first obtains a provably
optimal graph-theoretic schedule, then detects and resolves physical-layer
conflicts that the graph model cannot represent.

The G² schedule provides a proven lower bound.  The SINR-repaired schedule
provides a physics-safe upper bound.  The gap between them is the "price
of physics" — the extra slots needed because interference propagates
further than decoding range.
"""

import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .topology import Coordinates, build_connectivity_graph, euclidean_distance
from .conflicts import build_conflict_graph
from .solver import solve, ScheduleResult
from .sinr_auditor import (
    RadioParams,
    audit_schedule,
    SINRAuditReport,
    LinkPrediction,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RepairIteration:
    """Record of one iteration of the repair loop."""
    iteration: int
    new_conflict_edges: List[Tuple[str, str]]   # Edges added this round
    slot_count: int                               # K after re-coloring
    sinr_failures: int                            # Remaining failures
    optimality_status: str


@dataclass
class SINRRepairResult:
    """Full result of the SINR-aware repair process."""
    g2_schedule: Dict[str, int]          # Original G² schedule
    g2_slot_count: int                   # K under pure graph model
    repaired_schedule: Dict[str, int]    # Physics-safe schedule
    repaired_slot_count: int             # K after repair
    price_of_physics: int                # Extra slots needed
    iterations: List[RepairIteration]    # Repair history
    total_edges_added: int               # Conflict edges added by physics
    g2_lower_bound: int                  # Proven minimum under G²
    repaired_lower_bound: int            # Lower bound of augmented graph
    repaired_optimality: str             # Optimality of repaired schedule
    remaining_sinr_failures: int         # 0 if fully repaired
    params: RadioParams                  # Radio parameters used


# ---------------------------------------------------------------------------
# Core repair loop
# ---------------------------------------------------------------------------

def repair_schedule_sinr(
    coords: Coordinates,
    range_m: float = 500.0,
    params: Optional[RadioParams] = None,
    max_iterations: int = 10,
    timeout_sec: float = 5.0,
    seed: Optional[int] = None,
) -> SINRRepairResult:
    """
    Iteratively repair a G² schedule until it is SINR-safe.

    Algorithm:
      1. Build G, G², solve for optimal G² coloring.
      2. Run SINR audit on the schedule.
      3. For each link where G² said SAFE but SINR says FAIL:
         - Identify the interfering transmitter pair (tx, interferer).
         - Add (tx, interferer) as a conflict edge in an augmented graph.
      4. Re-solve the augmented conflict graph.
      5. Repeat from step 2 until no new SINR failures, or max_iterations.

    The augmented conflict graph H_aug ⊇ G² includes edges that the graph
    model cannot represent: pairs of nodes whose simultaneous transmission
    degrades SINR at a receiver below the required threshold.

    Parameters
    ----------
    coords        : node coordinates
    range_m       : communication range
    params        : radio parameters (default: RadioParams())
    max_iterations: maximum repair iterations
    timeout_sec   : solver timeout per iteration
    seed          : random seed for solver

    Returns
    -------
    SINRRepairResult with full repair history
    """
    if params is None:
        params = RadioParams()

    # Step 1: Build baseline G² schedule
    G = build_connectivity_graph(coords, range_m)
    H = build_conflict_graph(G)
    g2_result = solve(H, seed=seed, timeout_sec=timeout_sec)

    g2_schedule = dict(g2_result.node_to_slot)
    g2_k = g2_result.slot_count
    g2_lower_bound = g2_result.lower_bound

    # Augmented conflict graph starts as a copy of H
    H_aug = H.copy()
    current_schedule = dict(g2_schedule)
    current_k = g2_k
    current_result = g2_result

    iterations: List[RepairIteration] = []
    total_edges_added = 0

    for iteration in range(max_iterations):
        # Step 2: SINR audit
        audit = audit_schedule(G, current_schedule, params, H)

        # Step 3: Find G²-safe but SINR-fail links
        # For each such link, the interferer(s) need new conflict edges
        new_edges: List[Tuple[str, str]] = []

        for pred in audit.g2_safe_but_sinr_fail:
            tx = pred.tx_node
            for interferer in pred.active_interferers:
                # Add conflict edge between tx and interferer
                # (they should not co-transmit)
                if not H_aug.has_edge(tx, interferer):
                    H_aug.add_edge(
                        tx, interferer,
                        conflict_type="sinr-physics",
                        via_nodes=[pred.rx_node],
                        via_node=pred.rx_node,
                        distance=euclidean_distance(
                            G.nodes[tx].get("pos", (0, 0)),
                            G.nodes[interferer].get("pos", (0, 0)),
                        ),
                        sinr_at_victim_db=pred.sinr_db,
                        victim_node=pred.rx_node,
                    )
                    new_edges.append((tx, interferer))

        iterations.append(RepairIteration(
            iteration=iteration + 1,
            new_conflict_edges=new_edges,
            slot_count=current_k,
            sinr_failures=len(audit.g2_safe_but_sinr_fail),
            optimality_status=current_result.optimality_status,
        ))

        if not new_edges:
            # No new conflicts found — schedule is SINR-safe (or no
            # further augmentation possible)
            break

        total_edges_added += len(new_edges)

        # Step 4: Re-solve the augmented conflict graph
        current_result = solve(H_aug, seed=seed, timeout_sec=timeout_sec)
        current_schedule = dict(current_result.node_to_slot)
        current_k = current_result.slot_count

    # Final audit to count remaining failures
    final_audit = audit_schedule(G, current_schedule, params, H_aug)
    remaining = len(final_audit.g2_safe_but_sinr_fail)

    return SINRRepairResult(
        g2_schedule=g2_schedule,
        g2_slot_count=g2_k,
        repaired_schedule=current_schedule,
        repaired_slot_count=current_k,
        price_of_physics=current_k - g2_k,
        iterations=iterations,
        total_edges_added=total_edges_added,
        g2_lower_bound=g2_lower_bound,
        repaired_lower_bound=current_result.lower_bound,
        repaired_optimality=current_result.optimality_status,
        remaining_sinr_failures=remaining,
        params=params,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_repair_report(result: SINRRepairResult) -> str:
    """Format the SINR repair result as a human-readable report."""
    p = result.params
    lines = [
        "",
        "SINR-AWARE SCHEDULE REPAIR",
        "=" * 65,
        f"Two-Layer TDMA Planning: G² optimality + physical-layer safety",
        "",
        f"Radio Parameters:",
        f"  TX power:          {p.tx_power_dbm} dBm",
        f"  Pathloss exponent: α = {p.pathloss_exponent}",
        f"  SINR threshold:    {p.sinr_threshold_db} dB",
        f"  Sensitivity:       {p.sensitivity_dbm} dBm",
        "",
        f"Layer 1 — Graph Model (G²):",
        f"  Slot count:    {result.g2_slot_count}",
        f"  Lower bound:   {result.g2_lower_bound}",
        "",
        f"Layer 2 — Physics-Aware Repair:",
        f"  Slot count:    {result.repaired_slot_count}",
        f"  Lower bound:   {result.repaired_lower_bound}",
        f"  Optimality:    {result.repaired_optimality}",
        f"  Edges added:   {result.total_edges_added} (physics-discovered conflicts)",
        f"  Iterations:    {len(result.iterations)}",
        "",
        f"Price of Physics: +{result.price_of_physics} slots "
        f"({result.g2_slot_count} → {result.repaired_slot_count})",
    ]

    if result.remaining_sinr_failures > 0:
        lines.append(
            f"\n  WARNING: {result.remaining_sinr_failures} SINR failures remain "
            f"(may need more iterations or denser conflict modelling)."
        )
    else:
        lines.append(
            f"\n  All links pass SINR check under the hard-threshold model."
        )

    if result.iterations:
        lines.append("")
        lines.append("Repair Iterations:")
        lines.append(f"  {'#':>3}  {'New Edges':>10}  {'Slots':>6}  "
                      f"{'Failures':>9}  {'Status':>15}")
        lines.append(f"  {'-'*3}  {'-'*10}  {'-'*6}  {'-'*9}  {'-'*15}")
        for it in result.iterations:
            lines.append(
                f"  {it.iteration:>3}  {len(it.new_conflict_edges):>10}  "
                f"{it.slot_count:>6}  {it.sinr_failures:>9}  "
                f"{it.optimality_status:>15}"
            )

    lines.append("=" * 65)
    return "\n".join(lines)
