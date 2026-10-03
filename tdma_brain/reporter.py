"""
reporter.py — All output generation: ASCII report, JSON, HTML/SVG topology map.
"""

import json
import hashlib
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

import networkx as nx

from .solver import ScheduleResult
from .validator import ValidationResult
from .topology import Coordinates


# ---------------------------------------------------------------------------
# Binary matrix builder
# ---------------------------------------------------------------------------

def build_matrix(
    schedule: Dict[str, int], node_order: List[str]
) -> List[List[int]]:
    """Build K×N binary matrix. matrix[slot][node_col] = 1 if node transmits in slot."""
    if not schedule:
        return []
    k = max(schedule.values()) + 1
    n = len(node_order)
    matrix = [[0] * n for _ in range(k)]
    node_to_col = {name: idx for idx, name in enumerate(node_order)}
    for node, slot in schedule.items():
        if node in node_to_col:
            matrix[slot][node_to_col[node]] = 1
    return matrix


# ---------------------------------------------------------------------------
# ASCII report (exact format demanded by the problem statement)
# ---------------------------------------------------------------------------

def format_ascii_report(
    result: ScheduleResult,
    G: nx.Graph,
    node_order: List[str],
    range_m: float = 500.0,
    validation: Optional[ValidationResult] = None,
) -> str:
    """
    Generate the exact ASCII report format from the problem statement, plus
    the optimality certificate block.
    """
    schedule = result.node_to_slot
    k = result.slot_count
    n = len(node_order)
    matrix = build_matrix(schedule, node_order)

    # Slot stats
    slot_to_nodes: Dict[int, List[str]] = {}
    for node, slot in schedule.items():
        slot_to_nodes.setdefault(slot, []).append(node)

    lines = []
    lines.append("=" * 65)
    lines.append(" TDMA TOPOLOGY OPTIMIZATION REPORT")
    lines.append("=" * 65)
    lines.append(f"Total Nodes Processed : {n}")
    lines.append(f"Configured Radio Range : {range_m} meters")
    lines.append(
        f"Optimized Frame Length : {k} unique timeslots (Lower is better)"
    )
    lines.append(
        f"Connectivity Edges     : {G.number_of_edges()}"
    )
    lines.append(
        f"Average Spatial Reuse  : {n}/{k} = {n/k:.2f} simultaneous TX/slot"
    )
    lines.append(
        f"Frame Reduction vs. Serial TDMA: {(1 - k/n)*100:.1f}%"
    )
    lines.append("-" * 65)

    lines.append("NODE -> SLOT ASSIGNMENTS:")
    for node in node_order:
        lines.append(f"  {node}: Slot {schedule.get(node, '?')}")

    lines.append("")
    lines.append("STRUCTURAL TDMA SCHEDULE MATRIX (Slot x Node Boolean Matrix):")

    # Header row
    header_nums = " | ".join(n[-2:] for n in node_order)
    lines.append(f"Slot \\ Node | {header_nums}")
    lines.append("-" * (12 + 5 * len(node_order)))

    for s in range(k):
        row_vals = " |  ".join(str(matrix[s][col]) for col in range(n))
        nodes_in_slot = sorted(slot_to_nodes.get(s, []))
        spatial_note = f"  <- {len(nodes_in_slot)} TX" if len(nodes_in_slot) > 1 else ""
        lines.append(f"Slot {s:02d}     |  {row_vals}{spatial_note}")

    lines.append("-" * (12 + 5 * len(node_order)))

    # Optimality certificate
    lines.append("")
    lines.append("OPTIMALITY CERTIFICATE:")
    lines.append(f"  Slots used   : {k}")
    lines.append(f"  Lower bound  : {result.lower_bound}")
    lines.append(f"  Status       : {result.optimality_status}")
    lines.append(f"  Algorithm    : {result.algorithm}")
    lines.append(f"  Solve time   : {result.runtime_ms:.1f} ms")
    if result.clique_witness:
        lines.append(
            f"  Clique witness: {', '.join(result.clique_witness)}"
        )
        lines.append(
            f"  (These {len(result.clique_witness)} nodes mutually conflict — "
            f"proves ≥{result.lower_bound} slots required)"
        )

    # Validation result
    if validation is not None:
        lines.append("")
        status_str = "✓ CONFLICT-FREE (under G² model)" if validation.is_valid else "✗ VIOLATIONS DETECTED"
        lines.append(f"VALIDATION: {status_str}")
        if not validation.is_valid:
            for v in validation.violations:
                lines.append(f"  ✗ {v}")
        else:
            for c in validation.checks_passed:
                lines.append(f"  ✓ {c}")

    lines.append("=" * 65)
    lines.append(
        "Execution finalized cleanly. "
        + ("Schedule verified conflict-free." if (validation and validation.is_valid) else "")
    )
    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark table
# ---------------------------------------------------------------------------

def format_benchmark_table(benchmark: List[Dict]) -> str:
    """Format heuristic benchmark comparison as a text table."""
    if not benchmark:
        return ""
    lines = [
        "",
        "HEURISTIC BENCHMARK COMPARISON",
        "=" * 50,
        f"{'Algorithm':>22}  {'Seed':>8}  {'Slots':>6}  {'Time (ms)':>10}",
        "-" * 50,
    ]
    for row in sorted(benchmark, key=lambda r: (r["slot_count"], r["runtime_ms"])):
        seed_str = str(row.get("seed", "—")) if row.get("seed") is not None else "—"
        lines.append(
            f"{row['algorithm']:>22}  {seed_str:>8}  "
            f"{row['slot_count']:>6}  {row['runtime_ms']:>10.3f}"
        )
    lines.append("=" * 50)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def compute_topology_hash(coords: Coordinates, range_m: float) -> str:
    """SHA-256 of the sorted coordinate JSON + range. Stable across runs."""
    payload = json.dumps(
        {"coords": coords, "range_m": range_m}, sort_keys=True
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()[:16]


def build_json_output(
    result: ScheduleResult,
    G: nx.Graph,
    node_order: List[str],
    coords: Coordinates,
    range_m: float = 500.0,
    validation: Optional[ValidationResult] = None,
) -> Dict:
    """Build the canonical JSON output dictionary."""
    matrix = build_matrix(result.node_to_slot, node_order)
    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "range_m": range_m,
            "node_count": len(node_order),
            "topology_hash": compute_topology_hash(coords, range_m),
            "seed": result.seed,
        },
        "topology": {
            "connectivity_edges": G.number_of_edges(),
            "max_degree_G": max(dict(G.degree()).values()) if G.nodes() else 0,
        },
        "schedule": {
            "node_order": node_order,
            "node_to_slot": result.node_to_slot,
            "slot_count": result.slot_count,
            "matrix": matrix,
        },
        "optimality": {
            "lower_bound": result.lower_bound,
            "status": result.optimality_status,
            "clique_witness": result.clique_witness,
            "algorithm": result.algorithm,
            "runtime_ms": result.runtime_ms,
        },
        "validation": {
            "is_valid": validation.is_valid if validation else None,
            "violations": validation.violations if validation else [],
            "checks_passed": validation.checks_passed if validation else [],
        },
        "benchmark": result.benchmark,
    }


# ---------------------------------------------------------------------------
# HTML/SVG topology visualiser
# ---------------------------------------------------------------------------

# Slot colour palette (distinct, accessible)
SLOT_COLORS = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
    "#264653", "#a8dadc", "#f77f00", "#023e8a", "#d62828",
    "#7b2d8b", "#4cc9f0", "#06d6a0", "#fb8500", "#8338ec",
    "#3a86ff",
]


def _scale_coords(
    coords: Coordinates,
    canvas_w: int = 700,
    canvas_h: int = 700,
    padding: int = 60,
) -> Dict[str, tuple]:
    """Scale real-world coordinates to SVG canvas coordinates."""
    xs = [xy[0] for xy in coords.values()]
    ys = [xy[1] for xy in coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x or 1
    span_y = max_y - min_y or 1

    usable_w = canvas_w - 2 * padding
    usable_h = canvas_h - 2 * padding

    scaled = {}
    for name, (x, y) in coords.items():
        sx = padding + (x - min_x) / span_x * usable_w
        # SVG y-axis is inverted
        sy = canvas_h - padding - (y - min_y) / span_y * usable_h
        scaled[name] = (round(sx, 1), round(sy, 1))
    return scaled


def generate_html_topology(
    G: nx.Graph,
    schedule: Dict[str, int],
    coords: Coordinates,
    clique_witness: Optional[List[str]] = None,
    title: str = "TDMA Schedule Topology",
) -> str:
    """
    Generate a self-contained HTML file with an inline SVG showing:
    - Nodes coloured by slot assignment
    - Connectivity edges
    - Clique-witness nodes highlighted with a bold ring
    - Hover tooltips with node info
    - Legend for slot colours
    """
    canvas_w, canvas_h = 750, 650
    scaled = _scale_coords(coords, canvas_w, canvas_h)
    clique_set = set(clique_witness or [])
    k = max(schedule.values()) + 1 if schedule else 0

    svg_lines = []
    svg_lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{canvas_w}" height="{canvas_h}" '
        f'style="background:#1a1a2e;border-radius:12px;">'
    )

    # Draw edges first (under nodes)
    for u, v in G.edges():
        x1, y1 = scaled[u]
        x2, y2 = scaled[v]
        svg_lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#4a4a6a" stroke-width="1.5" opacity="0.6"/>'
        )

    # Draw nodes
    r = 18
    for name in G.nodes():
        x, y = scaled[name]
        slot = schedule.get(name, 0)
        color = SLOT_COLORS[slot % len(SLOT_COLORS)]
        is_clique = name in clique_set
        ring = (
            f'<circle cx="{x}" cy="{y}" r="{r+5}" '
            f'fill="none" stroke="white" stroke-width="2.5" opacity="0.9"/>'
            if is_clique else ""
        )
        svg_lines.append(ring)
        svg_lines.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" '
            f'stroke="#ffffff" stroke-width="1">'
            f'<title>{name}\nSlot {slot}'
            f'{" (clique witness)" if is_clique else ""}</title>'
            f'</circle>'
        )
        # Node label (short number)
        short = name.split("_")[-1].lstrip("0") or "0"
        svg_lines.append(
            f'<text x="{x}" y="{y+1}" text-anchor="middle" '
            f'dominant-baseline="middle" fill="white" '
            f'font-family="monospace" font-size="11" font-weight="bold">'
            f'{short}</text>'
        )

    svg_lines.append("</svg>")

    # Legend
    legend_items = ""
    for s in range(k):
        color = SLOT_COLORS[s % len(SLOT_COLORS)]
        legend_items += (
            f'<span style="display:inline-block;margin:4px 8px;">'
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{color};border-radius:50%;vertical-align:middle;'
            f'margin-right:4px;"></span>Slot {s}</span>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ background: #0f0f1a; color: #e0e0f0; font-family: 'Segoe UI', sans-serif;
          display: flex; flex-direction: column; align-items: center; padding: 32px; }}
  h1 {{ color: #a0c4ff; font-size: 1.4rem; margin-bottom: 4px; }}
  .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 24px; }}
  .legend {{ margin-top: 20px; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>🛰️ {title}</h1>
<p class="subtitle">Node colours = time slots · Bold ring = clique witness (lower bound proof)</p>
{''.join(svg_lines)}
<div class="legend">{legend_items}</div>
</body>
</html>"""
    return html
