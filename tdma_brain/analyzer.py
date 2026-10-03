"""
analyzer.py — Operational intelligence features:

  1. δ-Bounded Movement Robustness Guarantee
     Build connectivity graph at range R+2δ. Any valid coloring at this
     expanded range guarantees collision-freedom under δ-bounded movement.

  2. Algorithm-Independent Conflict Explanations
     For each node, report the concrete 1-hop or 2-hop conflict that blocks
     every lower-numbered slot. Works regardless of which algorithm solved it.

  3. Low-Churn Dynamic Replanning (Heuristic)
     Given old and new topologies, produce a valid schedule that attempts
     to minimize node reassignments using a greedy most-constrained-first
     heuristic. This does NOT guarantee globally minimum churn; the exact
     minimisation of  min Σ_{v∈V} [c_new(v) ≠ c_old(v)]  subject to valid
     coloring is itself NP-hard.
"""

import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .topology import build_connectivity_graph, Coordinates
from .conflicts import build_conflict_graph
from .solver import solve, ScheduleResult


# ---------------------------------------------------------------------------
# 1. δ-Bounded Movement Robustness
# ---------------------------------------------------------------------------

@dataclass
class RobustnessPoint:
    delta_m: float            # Movement tolerance in metres
    slot_count: int           # Slots needed at this tolerance
    lower_bound: int          # Proven minimum at this range
    optimality: str           # PROVEN_OPTIMAL | BOUNDED | TIMEOUT
    new_edges: int            # Additional edges vs. baseline (δ=0)
    expanded_range_m: float   # R + 2δ


def robustness_sweep(
    coords: Coordinates,
    range_m: float = 500.0,
    deltas: Optional[List[float]] = None,
    timeout_sec: float = 2.0,
) -> List[RobustnessPoint]:
    """
    For each δ, build G at range R+2δ, compute G², find optimal coloring.

    The guarantee: if every node moves ≤ δ metres, any link formed after
    movement was within R+2δ before the move.  Coloring at R+2δ is safe
    under all δ-bounded movements.  This is conservative (real ≥ stated).

    Parameters
    ----------
    coords    : node coordinates
    range_m   : nominal communication range (default 500 m)
    deltas    : list of δ values to sweep (default: 0..100 m)
    timeout_sec: per-point solver timeout

    Returns
    -------
    List[RobustnessPoint] ordered by ascending δ
    """
    if deltas is None:
        deltas = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

    # Baseline edge count at δ=0
    G0 = build_connectivity_graph(coords, range_m)
    baseline_edges = G0.number_of_edges()

    results: List[RobustnessPoint] = []
    for delta in deltas:
        expanded_range = range_m + 2 * delta
        G_exp = build_connectivity_graph(coords, expanded_range)
        H_exp = build_conflict_graph(G_exp)
        result = solve(H_exp, timeout_sec=timeout_sec)
        results.append(RobustnessPoint(
            delta_m=delta,
            slot_count=result.slot_count,
            lower_bound=result.lower_bound,
            optimality=result.optimality_status,
            new_edges=G_exp.number_of_edges() - baseline_edges,
            expanded_range_m=expanded_range,
        ))

    return results


def format_robustness_report(points: List[RobustnessPoint]) -> str:
    """Format the robustness sweep as a text table."""
    lines = [
        "",
        "ROBUSTNESS ANALYSIS (δ-bounded movement guarantee)",
        "=" * 60,
        f"{'δ (m)':>6}  {'Range (m)':>10}  {'Slots':>6}  "
        f"{'Lower':>6}  {'Status':>15}  {'+Edges':>7}",
        "-" * 60,
    ]
    baseline_slots = points[0].slot_count if points else 0
    for p in points:
        diff = p.slot_count - baseline_slots
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        lines.append(
            f"{p.delta_m:>6.0f}  {p.expanded_range_m:>10.0f}  "
            f"{p.slot_count:>6}  {p.lower_bound:>6}  "
            f"{p.optimality:>15}  {p.new_edges:>6}  "
            f"({diff_str} slots)"
        )
    lines.append("=" * 60)
    lines.append(
        "Guarantee: schedule built at row's range is safe if all nodes "
        "move ≤ δ metres."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Algorithm-Independent Conflict Explanations
# ---------------------------------------------------------------------------

def explain_slot_assignment(
    G: nx.Graph,
    H: nx.Graph,
    schedule: Dict[str, int],
    node: str,
) -> List[str]:
    """
    For a node assigned to slot S, explain why slots 0..S-1 were unavailable.

    For each blocked slot s < S:
      Find a conflict neighbour u (in H) such that slot(u) == s.
      Report whether it's a 1-hop or 2-hop conflict, with distances.

    Example output lines:
      "Slot 0 blocked by Node_01 (1-hop, 400m direct)"
      "Slot 1 blocked by Node_04 (2-hop via Node_05, 400m + 400m)"
    """
    if node not in schedule:
        return [f"{node} not in schedule."]

    assigned_slot = schedule[node]
    lines: List[str] = []

    # Map: slot → [nodes using that slot]
    slot_to_nodes: Dict[int, List[str]] = {}
    for n, s in schedule.items():
        slot_to_nodes.setdefault(s, []).append(n)

    for s in range(assigned_slot):
        nodes_in_slot = slot_to_nodes.get(s, [])
        # Find conflict neighbour using this slot
        blocker: Optional[str] = None
        for candidate in nodes_in_slot:
            if H.has_edge(node, candidate):
                blocker = candidate
                break

        if blocker is None:
            lines.append(f"Slot {s}: (no direct conflict found — slot taken by non-adjacent node)")
            continue

        edge_data = H[node][blocker]
        ctype = edge_data.get("conflict_type", "unknown")

        if ctype == "1-hop":
            dist = edge_data.get("distance", 0.0)
            lines.append(
                f"Slot {s} blocked by {blocker} "
                f"(1-hop neighbour, {dist:.0f}m direct)"
            )
        else:
            via_nodes = edge_data.get("via_nodes", [])
            # Fall back to legacy via_node if via_nodes not present
            if not via_nodes:
                via = edge_data.get("via_node")
                via_nodes = [via] if via else []
            # Report all shared relay nodes
            via_parts = []
            for via in via_nodes:
                if via and G.has_edge(node, via) and G.has_edge(via, blocker):
                    d1 = G[node][via].get("distance", 0.0)
                    d2 = G[via][blocker].get("distance", 0.0)
                    via_parts.append(f"{via} ({d1:.0f}m + {d2:.0f}m)")
            if via_parts:
                lines.append(
                    f"Slot {s} blocked by {blocker} "
                    f"(2-hop via {', '.join(via_parts)})"
                )
            else:
                dist = edge_data.get("distance", 0.0)
                lines.append(
                    f"Slot {s} blocked by {blocker} "
                    f"(2-hop, {dist:.0f}m)"
                )

    # Note spatial reuse if this node's slot is also used by others
    same_slot_nodes = [n for n in slot_to_nodes.get(assigned_slot, []) if n != node]
    if same_slot_nodes:
        # Find nearest same-slot node
        pos_self = G.nodes[node].get("pos", (0, 0))
        min_dist = float("inf")
        nearest = same_slot_nodes[0]
        for other in same_slot_nodes:
            pos_other = G.nodes[other].get("pos", (0, 0))
            d = ((pos_self[0] - pos_other[0])**2 + (pos_self[1] - pos_other[1])**2)**0.5
            if d < min_dist:
                min_dist = d
                nearest = other
        lines.append(
            f"↳ SPATIAL REUSE: shares Slot {assigned_slot} with "
            f"{same_slot_nodes} "
            f"(nearest: {nearest} at {min_dist:.0f}m, no conflict)"
        )

    return lines


def format_all_explanations(
    G: nx.Graph,
    H: nx.Graph,
    schedule: Dict[str, int],
    node_order: List[str],
) -> str:
    """Format per-node conflict explanations for all nodes as a report block."""
    lines = [
        "",
        "SLOT ASSIGNMENT EXPLANATIONS",
        "=" * 60,
    ]
    for node in node_order:
        slot = schedule.get(node, "?")
        lines.append(f"\n{node} → Slot {slot}")
        explanation = explain_slot_assignment(G, H, schedule, node)
        for e in explanation:
            lines.append(f"  {e}")
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Minimum-Churn Dynamic Replanning
# ---------------------------------------------------------------------------

@dataclass
class ReplanResult:
    old_schedule: Dict[str, int]
    new_schedule: Dict[str, int]
    reassigned_nodes: List[Tuple[str, int, int]]  # (node, old_slot, new_slot)
    churn_count: int
    old_slot_count: int
    new_slot_count: int
    reason: str


def replan_minimum_churn(
    coords_old: Coordinates,
    coords_new: Coordinates,
    old_schedule: Dict[str, int],
    range_m: float = 500.0,
    timeout_sec: float = 5.0,
) -> ReplanResult:
    """
    Given a topology change (coords_old → coords_new), produce a new valid
    schedule that attempts to minimize node reassignments.

    NOTE: This uses a greedy heuristic (most-constrained-first) rather than
    an exact minimum-churn solver.  The result is a valid schedule with low
    churn, but not necessarily the globally minimum number of reassignments.

    Strategy:
      1. Build G_new and H_new = G_new²
      2. Check which old assignments are still valid under H_new
      3. For violated nodes (sorted by conflict degree, most constrained
         first), try to reassign within existing slots first
      4. Only open a new slot if no existing slot can accommodate the node
      5. Return the full diff

    Parameters
    ----------
    coords_old  : original node coordinates
    coords_new  : updated node coordinates
    old_schedule: the existing slot assignments
    range_m     : communication range
    timeout_sec : solver timeout if full re-solve is needed

    Returns
    -------
    ReplanResult
    """
    G_new = build_connectivity_graph(coords_new, range_m)
    H_new = build_conflict_graph(G_new)

    old_k = max(old_schedule.values()) + 1 if old_schedule else 0

    # Step 1: Identify which old assignments are still valid
    # A node's assignment is "violated" if it now conflicts with a neighbour
    # that uses the same slot in the old schedule
    violated: List[str] = []
    for node in H_new.nodes():
        if node not in old_schedule:
            violated.append(node)
            continue
        my_slot = old_schedule[node]
        for nbr in H_new.neighbors(node):
            if nbr in old_schedule and old_schedule[nbr] == my_slot:
                if node not in violated:
                    violated.append(node)
                break

    if not violated:
        # Nothing changed — old schedule is still valid
        return ReplanResult(
            old_schedule=old_schedule,
            new_schedule=dict(old_schedule),
            reassigned_nodes=[],
            churn_count=0,
            old_slot_count=old_k,
            new_slot_count=old_k,
            reason="No new conflicts detected. Old schedule remains valid.",
        )

    # Step 2: Fix violated nodes greedily using existing slots first
    new_schedule = dict(old_schedule)
    reassigned: List[Tuple[str, int, int]] = []
    current_k = old_k

    # Sort violated nodes by number of conflicts (most constrained first)
    violated_sorted = sorted(
        violated,
        key=lambda n: H_new.degree(n),
        reverse=True,
    )

    for node in violated_sorted:
        old_slot = old_schedule.get(node, -1)
        # Try to find a valid slot, preferring the old slot, then lower slots
        assigned = False
        slots_to_try = list(range(current_k)) + [current_k]

        for slot in slots_to_try:
            # Check if this slot is valid for node in H_new given current assignments
            conflict = False
            for nbr in H_new.neighbors(node):
                if new_schedule.get(nbr) == slot:
                    conflict = True
                    break
            if not conflict:
                new_schedule[node] = slot
                if slot == current_k:
                    current_k += 1
                if slot != old_slot:
                    reassigned.append((node, old_slot, slot))
                assigned = True
                break

        if not assigned:
            # Fallback: open a new slot
            new_schedule[node] = current_k
            reassigned.append((node, old_slot, current_k))
            current_k += 1

    reason_parts = []
    for n, old_s, new_s in reassigned:
        reason_parts.append(f"{n}: Slot {old_s} → Slot {new_s}")

    return ReplanResult(
        old_schedule=old_schedule,
        new_schedule=new_schedule,
        reassigned_nodes=reassigned,
        churn_count=len(reassigned),
        old_slot_count=old_k,
        new_slot_count=current_k,
        reason="; ".join(reason_parts) if reason_parts else "No changes required.",
    )


def format_replan_report(result: ReplanResult) -> str:
    """Format the replan result as a text report."""
    lines = [
        "",
        "DYNAMIC REPLANNING RESULT",
        "=" * 50,
        f"Previous schedule : {result.old_slot_count} slots",
        f"New schedule      : {result.new_slot_count} slots",
        f"Nodes reassigned  : {result.churn_count} / "
        f"{len(result.new_schedule)}",
        "",
    ]
    if result.reassigned_nodes:
        lines.append("Changes:")
        for node, old_s, new_s in result.reassigned_nodes:
            marker = " (new slot)" if new_s >= result.old_slot_count else ""
            lines.append(f"  {node}: Slot {old_s} → Slot {new_s}{marker}")
    else:
        lines.append("No changes required. Old schedule remains valid.")
    lines.append("=" * 50)
    return "\n".join(lines)
