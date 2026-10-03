"""
test_validator.py — Tests for validator.py

Validates that the independent G-based checker catches all collision types.
"""

import pytest
from tdma_brain.topology import build_connectivity_graph, get_node_order
from tdma_brain.conflicts import build_conflict_graph
from tdma_brain.solver import solve
from tdma_brain.validator import validate_schedule, validate_matrix
from tdma_brain.reporter import build_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coords_path_4():
    return {
        "A": (0.0, 0.0), "B": (400.0, 0.0),
        "C": (800.0, 0.0), "D": (1200.0, 0.0),
    }

def _full_pipeline(coords, range_m=500.0):
    G = build_connectivity_graph(coords, range_m)
    H = build_conflict_graph(G)
    result = solve(H)
    node_order = get_node_order(coords)
    return G, H, result, node_order


# ---------------------------------------------------------------------------
# Valid schedules pass
# ---------------------------------------------------------------------------

def test_valid_schedule_passes():
    coords = _coords_path_4()
    G, H, result, node_order = _full_pipeline(coords)
    val = validate_schedule(G, result.node_to_slot, node_order)
    assert val.is_valid, f"Valid schedule failed: {val.violations}"

def test_valid_grid_passes():
    coords = {f"Node_{r*4+c+1:02d}": (c*400.0, r*400.0)
              for r in range(4) for c in range(4)}
    G, H, result, node_order = _full_pipeline(coords)
    val = validate_schedule(G, result.node_to_slot, node_order)
    assert val.is_valid, f"Valid grid schedule failed: {val.violations}"


# ---------------------------------------------------------------------------
# 1-hop collision detection
# ---------------------------------------------------------------------------

def test_direct_neighbour_conflict_caught():
    """Two directly-connected nodes in the same slot must be flagged."""
    coords = {"A": (0.0, 0.0), "B": (400.0, 0.0), "C": (1200.0, 0.0)}
    G = build_connectivity_graph(coords)
    node_order = get_node_order(coords)
    # A and B are direct neighbours — putting them in same slot is invalid
    bad_schedule = {"A": 0, "B": 0, "C": 1}
    val = validate_schedule(G, bad_schedule, node_order)
    assert not val.is_valid
    assert any("1-hop" in v or "Slot 0" in v for v in val.violations)


# ---------------------------------------------------------------------------
# 2-hop hidden-terminal collision detection
# ---------------------------------------------------------------------------

def test_hidden_terminal_conflict_caught():
    """
    A - B - C  (A and C are 2-hop via B)
    Putting A and C in the same slot must be flagged:
    both transmit → B cannot decode either.
    """
    coords = {"A": (0.0, 0.0), "B": (400.0, 0.0), "C": (800.0, 0.0)}
    G = build_connectivity_graph(coords)
    node_order = get_node_order(coords)
    bad_schedule = {"A": 0, "B": 1, "C": 0}
    val = validate_schedule(G, bad_schedule, node_order)
    assert not val.is_valid
    # Should mention 2-hop conflict
    combined = " ".join(val.violations)
    assert "2-hop" in combined or "hidden terminal" in combined


# ---------------------------------------------------------------------------
# Missing node detection
# ---------------------------------------------------------------------------

def test_missing_node_caught():
    coords = {"A": (0.0, 0.0), "B": (400.0, 0.0), "C": (800.0, 0.0)}
    G = build_connectivity_graph(coords)
    node_order = get_node_order(coords)
    incomplete_schedule = {"A": 0, "B": 1}  # C is missing
    val = validate_schedule(G, incomplete_schedule, node_order)
    assert not val.is_valid
    assert any("C" in v for v in val.violations)


# ---------------------------------------------------------------------------
# Slot contiguity
# ---------------------------------------------------------------------------

def test_slot_gap_caught():
    """Slots 0 and 2 but not 1 → should flag as non-contiguous."""
    coords = {"A": (0.0, 0.0), "B": (2000.0, 0.0), "C": (4000.0, 0.0)}
    G = build_connectivity_graph(coords, range_m=500.0)
    node_order = get_node_order(coords)
    gapped_schedule = {"A": 0, "B": 2, "C": 0}  # slot 1 missing
    val = validate_schedule(G, gapped_schedule, node_order)
    assert not val.is_valid
    assert any("gap" in v.lower() or "contiguity" in v.lower() or
               "found" in v.lower() for v in val.violations)


# ---------------------------------------------------------------------------
# Matrix validation
# ---------------------------------------------------------------------------

def test_matrix_column_sum_one():
    """Every column (node) must have exactly one 1 in the matrix."""
    node_order = ["A", "B", "C"]
    schedule = {"A": 0, "B": 1, "C": 0}
    matrix = build_matrix(schedule, node_order)
    val = validate_matrix(matrix, schedule, node_order)
    assert val.is_valid

def test_matrix_column_sum_zero_caught():
    node_order = ["A", "B", "C"]
    schedule = {"A": 0, "B": 1, "C": 0}
    matrix = build_matrix(schedule, node_order)
    # Corrupt: set A's entry to 0 (column sum becomes 0)
    matrix[0][0] = 0
    val = validate_matrix(matrix, schedule, node_order)
    assert not val.is_valid

def test_matrix_non_binary_caught():
    node_order = ["A", "B"]
    schedule = {"A": 0, "B": 1}
    matrix = [[2, 0], [0, 1]]   # '2' is non-binary
    val = validate_matrix(matrix, schedule, node_order)
    assert not val.is_valid

def test_matrix_mismatch_caught():
    """Matrix that doesn't match the schedule must be flagged."""
    node_order = ["A", "B", "C"]
    schedule = {"A": 0, "B": 1, "C": 0}
    matrix = build_matrix(schedule, node_order)
    # Move A from slot 0 to slot 1 in matrix (but schedule still says slot 0)
    matrix[0][0] = 0
    matrix[1][0] = 1
    val = validate_matrix(matrix, schedule, node_order)
    assert not val.is_valid


# ---------------------------------------------------------------------------
# Checks passed list
# ---------------------------------------------------------------------------

def test_valid_schedule_reports_checks_passed():
    coords = _coords_path_4()
    G, H, result, node_order = _full_pipeline(coords)
    val = validate_schedule(G, result.node_to_slot, node_order)
    assert "closed_neighbourhood_uniqueness" in val.checks_passed
    assert "node_completeness" in val.checks_passed
