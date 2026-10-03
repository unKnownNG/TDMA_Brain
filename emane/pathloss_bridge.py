"""
emane/pathloss_bridge.py — Generate realistic EMANE pathloss events from coordinates.

IMPORTANT DESIGN DECISION (corrected from naive approach):
  Do NOT use a hard-cutoff pathloss model (0 dB < 500m, 200 dB > 500m).
  That makes validation circular and hides the most interesting EMANE physics:
  the two-radius effect where interference reaches beyond the decode boundary.

  Instead: compute actual free-space pathloss for EVERY pair, regardless of
  whether they are in G's edge set. This means:
    - Nodes at 450m:  realistic pathloss (good signal, strong decode)
    - Nodes at 520m:  slightly higher pathloss (outside decode range,
                      BUT still produces interference if above sensitivity)
    - Nodes at 970m:  much higher pathloss (weak, but possibly above
                      sensitivity → counts as noise in EMANE)

  This is what makes the SINR counterexample manifest correctly in EMANE.
"""

import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Tuple

from tdma_brain.topology import Coordinates, euclidean_distance


# ---------------------------------------------------------------------------
# Pathloss calculation (same model as sinr_auditor.py)
# ---------------------------------------------------------------------------

def free_space_pathloss_db(
    distance_m: float,
    pathloss_exponent: float = 2.0,
    frequency_mhz: float = 2400.0,
    reference_distance_m: float = 1.0,
) -> float:
    """
    Log-distance pathloss model (same formula as sinr_auditor.py).
    Must match exactly for auditor predictions to match EMANE measurements.
    """
    if distance_m <= 0:
        return 0.0
    if distance_m < reference_distance_m:
        distance_m = reference_distance_m

    freq_hz = frequency_mhz * 1e6
    wavelength = 3e8 / freq_hz
    l_d0 = 20 * math.log10(4 * math.pi * reference_distance_m / wavelength)
    pathloss = l_d0 + 10 * pathloss_exponent * math.log10(
        distance_m / reference_distance_m
    )
    return max(pathloss, 0.0)


# ---------------------------------------------------------------------------
# Pathloss event generation
# ---------------------------------------------------------------------------

def generate_pathloss_table(
    coords: Coordinates,
    node_order: List[str],
    pathloss_exponent: float = 2.0,
    frequency_mhz: float = 2400.0,
) -> List[Dict]:
    """
    Compute realistic pathloss for all node pairs.

    Returns list of dicts:
        {nem_tx: int, nem_rx: int, pathloss_db: float, distance_m: float}

    NEM IDs are 1-based indices into node_order.
    All pairs are included (not just connected ones) so EMANE can model
    out-of-range interference correctly.
    """
    node_to_nem = {name: idx + 1 for idx, name in enumerate(node_order)}
    table = []

    for i, name_i in enumerate(node_order):
        for j, name_j in enumerate(node_order):
            if i == j:
                continue
            pos_i = coords[name_i]
            pos_j = coords[name_j]
            dist = euclidean_distance(pos_i, pos_j)
            pl = free_space_pathloss_db(
                dist, pathloss_exponent, frequency_mhz
            )
            table.append({
                "nem_tx": node_to_nem[name_i],
                "nem_rx": node_to_nem[name_j],
                "pathloss_db": round(pl, 2),
                "distance_m": round(dist, 2),
            })

    return table


def generate_pathloss_xml(
    coords: Coordinates,
    node_order: List[str],
    output_path: str,
    pathloss_exponent: float = 2.0,
    frequency_mhz: float = 2400.0,
) -> str:
    """
    Generate an EMANE PathlossEvent XML for precomputed pathloss.

    This is consumed by EMANE's physical layer model.
    All pairs are included with realistic free-space values.

    Returns the XML string and writes to output_path.
    """
    table = generate_pathloss_table(
        coords, node_order, pathloss_exponent, frequency_mhz
    )

    root = ET.Element("pathloss")
    root.set("nem", "all")

    for row in table:
        entry = ET.SubElement(root, "entry")
        entry.set("nem", str(row["nem_rx"]))
        entry.set("forward", str(row["pathloss_db"]))
        entry.set("reverse", str(row["pathloss_db"]))

    raw = ET.tostring(root, encoding="unicode")
    parsed = minidom.parseString(raw)
    xml_str = parsed.toprettyxml(indent="  ")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"[EMANE] Pathloss XML written to {output_path}")
    return xml_str


def print_pathloss_summary(
    table: List[Dict],
    node_order: List[str],
    range_m: float = 500.0,
) -> None:
    """
    Print a human-readable summary showing:
    - In-range pairs (would be G edges)
    - Out-of-range but above likely sensitivity (potential interference sources)
    """
    print("\nPATHLOSS SUMMARY")
    print("=" * 60)
    print(f"{'NEM TX':>6} → {'NEM RX':>6}  {'Dist (m)':>10}  "
          f"{'PL (dB)':>8}  {'Status':>20}")
    print("-" * 60)
    node_to_nem = {name: idx + 1 for idx, name in enumerate(node_order)}
    nem_to_node = {v: k for k, v in node_to_nem.items()}

    # Show only unique pairs (i < j)
    seen = set()
    for row in table:
        pair = (min(row["nem_tx"], row["nem_rx"]), max(row["nem_tx"], row["nem_rx"]))
        if pair in seen:
            continue
        seen.add(pair)
        status = "IN RANGE (G edge)" if row["distance_m"] <= range_m else "out of range"
        print(
            f"{nem_to_node[row['nem_tx']]:>10} → {nem_to_node[row['nem_rx']]:>10}  "
            f"{row['distance_m']:>10.1f}  {row['pathloss_db']:>8.1f}  {status:>20}"
        )
    print("=" * 60)
