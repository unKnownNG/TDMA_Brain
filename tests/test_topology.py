"""
test_topology.py — Tests for topology.py

Covers boundary conditions, isolated nodes, exact-distance edges.
"""

import math
import pytest
import networkx as nx

from tdma_brain.topology import (
    squared_distance,
    euclidean_distance,
    build_connectivity_graph,
    get_node_order,
    topology_stats,
)


# ---------------------------------------------------------------------------
# squared_distance
# ---------------------------------------------------------------------------

def test_squared_distance_zero():
    assert squared_distance((0, 0), (0, 0)) == 0.0

def test_squared_distance_basic():
    assert squared_distance((0, 0), (3, 4)) == 25.0  # 3-4-5 triangle

def test_squared_distance_300_400_500():
    """300² + 400² = 90000 + 160000 = 250000 = 500²"""
    assert squared_distance((0, 0), (300, 400)) == pytest.approx(250000.0)


# ---------------------------------------------------------------------------
# build_connectivity_graph — edge inclusion
# ---------------------------------------------------------------------------

def test_exactly_500m_included():
    """A node exactly 500m away must be connected (inclusive boundary)."""
    # 300² + 400² = 500²  — exact boundary
    coords = {"A": (0.0, 0.0), "B": (300.0, 400.0)}
    G = build_connectivity_graph(coords, range_m=500.0)
    assert G.has_edge("A", "B"), "Exact 500m distance must produce an edge"

def test_just_over_500m_excluded():
    """A node 500.001m away must NOT be connected."""
    coords = {"A": (0.0, 0.0), "B": (500.001, 0.0)}
    G = build_connectivity_graph(coords, range_m=500.0)
    assert not G.has_edge("A", "B"), "500.001m must NOT produce an edge"

def test_just_under_500m_included():
    coords = {"A": (0.0, 0.0), "B": (499.999, 0.0)}
    G = build_connectivity_graph(coords, range_m=500.0)
    assert G.has_edge("A", "B")

def test_zero_distance_self_loop_not_added():
    """Nodes at same position — no self-loop should appear."""
    coords = {"A": (0.0, 0.0), "B": (0.0, 0.0)}
    G = build_connectivity_graph(coords, range_m=500.0)
    assert G.has_edge("A", "B")  # They are neighbours
    assert not G.has_edge("A", "A")  # No self-loop


# ---------------------------------------------------------------------------
# Isolated nodes preserved
# ---------------------------------------------------------------------------

def test_isolated_nodes_preserved():
    """Nodes with no neighbours within range must still appear in the graph."""
    coords = {
        "A": (0.0, 0.0),
        "B": (10000.0, 10000.0),   # Very far away
    }
    G = build_connectivity_graph(coords, range_m=500.0)
    assert "A" in G.nodes()
    assert "B" in G.nodes()
    assert G.number_of_edges() == 0


def test_all_isolated():
    """16 nodes all far apart — no edges."""
    coords = {f"Node_{i:02d}": (i * 2000.0, 0.0) for i in range(16)}
    G = build_connectivity_graph(coords, range_m=500.0)
    assert G.number_of_nodes() == 16
    assert G.number_of_edges() == 0


# ---------------------------------------------------------------------------
# Complete graph (all nodes within range)
# ---------------------------------------------------------------------------

def test_complete_graph():
    """All nodes at the same location — complete graph K_N."""
    coords = {f"Node_{i:02d}": (0.0, 0.0) for i in range(5)}
    G = build_connectivity_graph(coords, range_m=500.0)
    # K_5 has 10 edges
    assert G.number_of_edges() == 10


# ---------------------------------------------------------------------------
# Node order
# ---------------------------------------------------------------------------

def test_node_order_lexicographic():
    coords = {"Node_10": (0, 0), "Node_02": (1, 0), "Node_01": (2, 0)}
    order = get_node_order(coords)
    assert order == ["Node_01", "Node_02", "Node_10"]


# ---------------------------------------------------------------------------
# Edge attributes
# ---------------------------------------------------------------------------

def test_edge_has_distance_attribute():
    coords = {"A": (0.0, 0.0), "B": (300.0, 0.0)}
    G = build_connectivity_graph(coords, range_m=500.0)
    assert "distance" in G["A"]["B"]
    assert G["A"]["B"]["distance"] == pytest.approx(300.0, abs=0.01)


# ---------------------------------------------------------------------------
# Node attributes
# ---------------------------------------------------------------------------

def test_node_has_pos_attribute():
    coords = {"A": (100.0, 200.0)}
    G = build_connectivity_graph(coords, range_m=500.0)
    assert G.nodes["A"]["pos"] == (100.0, 200.0)


# ---------------------------------------------------------------------------
# 4×4 grid at 400m — 24 edges (horizontal + vertical, no diagonals)
# ---------------------------------------------------------------------------

def _make_grid_400():
    coords = {}
    for r in range(4):
        for c in range(4):
            name = f"Node_{r*4+c+1:02d}"
            coords[name] = (c * 400.0, r * 400.0)
    return coords

def test_grid_400m_edge_count():
    coords = _make_grid_400()
    G = build_connectivity_graph(coords, range_m=500.0)
    # 4×4 grid: 2×(3×4) = 24 horizontal+vertical edges
    # Diagonal distance = 400√2 ≈ 566m > 500m → not connected
    assert G.number_of_edges() == 24


# ---------------------------------------------------------------------------
# 4×4 grid at 300m — 24 horizontal/vertical + 18 diagonal edges = 42
# ---------------------------------------------------------------------------

def _make_grid_300():
    coords = {}
    for r in range(4):
        for c in range(4):
            name = f"Node_{r*4+c+1:02d}"
            coords[name] = (c * 300.0, r * 300.0)
    return coords

def test_grid_300m_diagonal_edges():
    """300m grid: diagonal = 300√2 ≈ 424m < 500m → diagonals are edges."""
    coords = _make_grid_300()
    G = build_connectivity_graph(coords, range_m=500.0)
    # Diagonals: 3×3 = 9 per direction × 2 = 18, plus 24 orthogonal = 42
    assert G.number_of_edges() == 42


# ---------------------------------------------------------------------------
# topology_stats
# ---------------------------------------------------------------------------

def test_topology_stats_isolated():
    coords = {"A": (0, 0), "B": (10000, 0)}
    G = build_connectivity_graph(coords, range_m=500.0)
    stats = topology_stats(G)
    assert stats["edge_count"] == 0
    assert stats["isolated_nodes"] == ["A", "B"]
    assert stats["is_connected"] is False
