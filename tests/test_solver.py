"""
test_solver.py — Tests for solver.py

Covers optimality on known topologies, isolated nodes, determinism.
"""

import pytest
from tdma_brain.topology import build_connectivity_graph
from tdma_brain.conflicts import build_conflict_graph
from tdma_brain.solver import solve, find_max_clique


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solve(coords, range_m=500.0, **kwargs):
    G = build_connectivity_graph(coords, range_m)
    H = build_conflict_graph(G)
    return solve(H, **kwargs)


def _make_grid(spacing, size=4):
    coords = {}
    for r in range(size):
        for c in range(size):
            coords[f"Node_{r*size+c+1:02d}"] = (c * spacing, r * spacing)
    return coords


# ---------------------------------------------------------------------------
# Trivial topologies
# ---------------------------------------------------------------------------

def test_single_node():
    coords = {"A": (0.0, 0.0)}
    r = _solve(coords)
    assert r.slot_count == 1
    assert r.optimality_status == "PROVEN_OPTIMAL"

def test_all_isolated_nodes():
    """16 isolated nodes — all share slot 0."""
    coords = {f"Node_{i:02d}": (i * 2000.0, 0.0) for i in range(16)}
    r = _solve(coords)
    assert r.slot_count == 1
    assert r.optimality_status == "PROVEN_OPTIMAL"

def test_complete_graph_k4():
    """4 mutually interfering nodes → 4 slots."""
    coords = {f"Node_{i:02d}": (0.0, 0.0) for i in range(4)}
    r = _solve(coords)
    assert r.slot_count == 4
    assert r.optimality_status == "PROVEN_OPTIMAL"


# ---------------------------------------------------------------------------
# Path graphs
# ---------------------------------------------------------------------------

def test_3_node_path():
    """
    A - B - C (400m apart)
    A and C both 2-hop (via B) → 3 different slots required.
    """
    coords = {"A": (0.0, 0.0), "B": (400.0, 0.0), "C": (800.0, 0.0)}
    r = _solve(coords)
    assert r.slot_count == 3
    assert r.optimality_status == "PROVEN_OPTIMAL"

def test_4_node_path():
    """
    A - B - C - D (400m apart)
    A and D are 3 hops apart → they CAN share a slot.
    Minimum = 3 slots.
    """
    coords = {
        "A": (0.0, 0.0), "B": (400.0, 0.0),
        "C": (800.0, 0.0), "D": (1200.0, 0.0),
    }
    r = _solve(coords)
    assert r.slot_count == 3
    assert r.optimality_status == "PROVEN_OPTIMAL"
    # A and D must share a slot
    assert r.node_to_slot["A"] == r.node_to_slot["D"]


# ---------------------------------------------------------------------------
# Grid topologies — the key test cases
# ---------------------------------------------------------------------------

def test_grid_400m_5_slots():
    """4×4 grid at 400m → 5 slots, PROVEN_OPTIMAL."""
    coords = _make_grid(400)
    r = _solve(coords)
    assert r.slot_count == 5
    assert r.optimality_status == "PROVEN_OPTIMAL"
    assert r.lower_bound == 5

def test_grid_300m_9_slots():
    """
    4×4 grid at 300m → 9 slots (NOT 5!).
    This is the 'sample schedule trap' from the brief.
    300√2 ≈ 424m < 500m, so diagonal neighbours are connected,
    creating much denser conflict graphs.
    """
    coords = _make_grid(300)
    r = _solve(coords)
    assert r.slot_count == 9, (
        f"Expected 9 slots for 300m grid, got {r.slot_count}. "
        "This is the sample schedule trap — the brief's sample is wrong!"
    )
    assert r.optimality_status == "PROVEN_OPTIMAL"


# ---------------------------------------------------------------------------
# Disconnected components — reuse across components
# ---------------------------------------------------------------------------

def test_disconnected_two_clusters():
    """Two separate clusters far apart can share slot labels."""
    coords = {
        # Cluster 1 (tight, 3 nodes)
        "A": (0.0, 0.0), "B": (400.0, 0.0), "C": (800.0, 0.0),
        # Cluster 2 (tight, 3 nodes — 10km away)
        "D": (10000.0, 0.0), "E": (10400.0, 0.0), "F": (10800.0, 0.0),
    }
    r = _solve(coords)
    # Both clusters are paths → each needs 3 slots, but can reuse same slot numbers
    assert r.slot_count == 3


# ---------------------------------------------------------------------------
# Optimality certificate
# ---------------------------------------------------------------------------

def test_clique_witness_returned():
    """Solver must return a non-empty clique witness."""
    coords = _make_grid(400)
    r = _solve(coords)
    assert len(r.clique_witness) >= 1
    # Witness must match lower bound
    assert len(r.clique_witness) == r.lower_bound

def test_lower_bound_leq_slot_count():
    """Lower bound must never exceed slot count."""
    coords = _make_grid(400)
    r = _solve(coords)
    assert r.lower_bound <= r.slot_count


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_same_seed():
    """Same input + seed → same output."""
    coords = _make_grid(400)
    r1 = _solve(coords, seed=42)
    r2 = _solve(coords, seed=42)
    assert r1.node_to_slot == r2.node_to_slot
    assert r1.slot_count == r2.slot_count


# ---------------------------------------------------------------------------
# find_max_clique
# ---------------------------------------------------------------------------

def test_find_max_clique_path():
    """In a 4-node path, max clique in H = 3 (any 3 consecutive nodes)."""
    coords = {
        "A": (0.0, 0.0), "B": (400.0, 0.0),
        "C": (800.0, 0.0), "D": (1200.0, 0.0),
    }
    G = build_connectivity_graph(coords)
    H = build_conflict_graph(G)
    size, clique = find_max_clique(H)
    assert size == 3

def test_find_max_clique_complete():
    """K4 in conflict graph → max clique = 4."""
    coords = {f"Node_{i}": (0.0, 0.0) for i in range(4)}
    G = build_connectivity_graph(coords)
    H = build_conflict_graph(G)
    size, _ = find_max_clique(H)
    assert size == 4
