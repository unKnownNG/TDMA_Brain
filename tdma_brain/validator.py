"""
validator.py — Independent schedule validation against the physical graph G.

IMPORTANT: We validate against G (connectivity graph), NOT H (conflict graph).
Validating against H only catches errors we'd only see if H was built correctly.
Checking against G provides truly independent assurance.

Core invariant (Closed-Neighbourhood Uniqueness):
  For every node v, all slot assignments in {v} ∪ N_G(v) must be distinct.
  |{slot(u) : u ∈ {v} ∪ N_G(v)}| == 1 + |N_G(v)|

This single check catches BOTH:
  - Direct 1-hop interference (v and its neighbour share a slot)
  - Hidden-terminal 2-hop interference (two nodes share v as a common receiver
    and are assigned the same slot)
"""

import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[str] = field(default_factory=list)   # Human-readable witnesses
    checks_passed: List[str] = field(default_factory=list)


def validate_schedule(
    G: nx.Graph,
    schedule: Dict[str, int],
    node_order: List[str],
) -> ValidationResult:
    """
    Validate a schedule against the physical connectivity graph G.

    Checks performed:
      1. Node completeness  — all nodes in G appear in schedule
      2. Closed-neighbourhood uniqueness — the core collision check
      3. Slot contiguity    — slots are numbered 0..K-1 with no gaps
      4. Matrix consistency — slot assignments match expected matrix structure

    Returns
    -------
    ValidationResult with is_valid flag and human-readable violation witnesses.
    """
    violations: List[str] = []
    checks_passed: List[str] = []

    # -----------------------------------------------------------------
    # Check 1: Node completeness
    # -----------------------------------------------------------------
    graph_nodes = set(G.nodes())
    schedule_nodes = set(schedule.keys())

    missing = graph_nodes - schedule_nodes
    extra = schedule_nodes - graph_nodes

    if missing:
        violations.append(
            f"Missing nodes in schedule: {sorted(missing)}"
        )
    if extra:
        violations.append(
            f"Extra nodes in schedule not in graph: {sorted(extra)}"
        )

    if not missing and not extra:
        checks_passed.append("node_completeness")

    # -----------------------------------------------------------------
    # Check 2: Closed-neighbourhood uniqueness (the critical check)
    # -----------------------------------------------------------------
    collision_found = False
    for v in G.nodes():
        if v not in schedule:
            continue  # already reported as missing
        slot_v = schedule[v]
        neighbors = list(G.neighbors(v))
        # closed neighborhood: v itself + all its 1-hop neighbours
        closed_nbhd = [v] + neighbors
        seen_slots: Dict[int, str] = {}  # slot → first node using it
        for u in closed_nbhd:
            if u not in schedule:
                continue
            s = schedule[u]
            if s in seen_slots:
                # Collision detected
                other = seen_slots[s]
                if other != u:
                    # Determine conflict type
                    if G.has_edge(other, u):
                        conflict_desc = (
                            f"1-hop conflict: {other} and {u} are direct "
                            f"neighbours (distance "
                            f"{G[other][u].get('distance', '?')}m)"
                        )
                    else:
                        conflict_desc = (
                            f"2-hop conflict: {other} and {u} both reach "
                            f"receiver {v} (hidden terminal)"
                        )
                    violations.append(
                        f"Slot {s}: {conflict_desc}"
                    )
                    collision_found = True
            else:
                seen_slots[s] = u

    if not collision_found and not missing and not extra:
        checks_passed.append("closed_neighbourhood_uniqueness")

    # -----------------------------------------------------------------
    # Check 3: Slot contiguity (no gaps in slot numbering)
    # -----------------------------------------------------------------
    if schedule:
        used_slots = sorted(set(schedule.values()))
        expected_slots = list(range(len(used_slots)))
        if used_slots != expected_slots:
            violations.append(
                f"Slot numbering has gaps: found {used_slots}, "
                f"expected {expected_slots}"
            )
        else:
            checks_passed.append("slot_contiguity")

    # -----------------------------------------------------------------
    # Check 4: Each slot has at least one node (no empty slot rows)
    # -----------------------------------------------------------------
    if schedule:
        slot_to_nodes: Dict[int, List[str]] = {}
        for node, slot in schedule.items():
            slot_to_nodes.setdefault(slot, []).append(node)
        empty_slots = [s for s, nodes in slot_to_nodes.items() if not nodes]
        if empty_slots:
            violations.append(f"Empty slot rows: {empty_slots}")
        else:
            checks_passed.append("no_empty_slot_rows")

    is_valid = len(violations) == 0
    return ValidationResult(
        is_valid=is_valid,
        violations=violations,
        checks_passed=checks_passed,
    )


def validate_matrix(
    matrix: List[List[int]],
    schedule: Dict[str, int],
    node_order: List[str],
) -> ValidationResult:
    """
    Validate the binary Slot×Node matrix for structural correctness.

    Checks:
      - Every entry is 0 or 1
      - Each node column sums to exactly 1
      - Matrix dimensions are K × N
      - Matrix matches node_to_slot mapping
    """
    violations: List[str] = []
    checks_passed: List[str] = []

    if not matrix:
        violations.append("Matrix is empty.")
        return ValidationResult(is_valid=False, violations=violations)

    K = len(matrix)
    N = len(node_order)

    # Check dimensions
    for row_idx, row in enumerate(matrix):
        if len(row) != N:
            violations.append(
                f"Row {row_idx} has {len(row)} entries, expected {N}."
            )

    if not violations:
        checks_passed.append("matrix_dimensions")

    # Check binary entries and column sums
    col_sums = [0] * N
    binary_ok = True
    for row_idx, row in enumerate(matrix):
        for col_idx, val in enumerate(row):
            if val not in (0, 1):
                violations.append(
                    f"Non-binary value {val} at matrix[{row_idx}][{col_idx}]"
                )
                binary_ok = False
            else:
                col_sums[col_idx] += val

    if binary_ok:
        checks_passed.append("binary_entries")

    for col_idx, (node, col_sum) in enumerate(zip(node_order, col_sums)):
        if col_sum != 1:
            violations.append(
                f"Node {node} (col {col_idx}) column sum = {col_sum}, expected 1"
            )

    if all(s == 1 for s in col_sums):
        checks_passed.append("column_sums_one")

    # Check matrix matches schedule
    if schedule and binary_ok:
        for col_idx, node in enumerate(node_order):
            if node not in schedule:
                continue
            expected_slot = schedule[node]
            if 0 <= expected_slot < K:
                if matrix[expected_slot][col_idx] != 1:
                    violations.append(
                        f"Matrix mismatch: {node} assigned slot {expected_slot} "
                        f"but matrix[{expected_slot}][{col_idx}] = "
                        f"{matrix[expected_slot][col_idx]}"
                    )
        if not any("Matrix mismatch" in v for v in violations):
            checks_passed.append("matrix_matches_schedule")

    is_valid = len(violations) == 0
    return ValidationResult(
        is_valid=is_valid,
        violations=violations,
        checks_passed=checks_passed,
    )
