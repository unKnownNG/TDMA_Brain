"""
topology.py — Build the physical connectivity graph G from node coordinates.

(u, v) ∈ E  iff  (xu - xv)² + (yu - yv)²  ≤  range_m²

Uses squared distances throughout to avoid float errors at exact boundaries
(e.g., 300² + 400² = 250000 = 500² exactly — must be included).
"""

import math
import networkx as nx
from typing import Dict, List, Tuple

# Type alias
Coordinates = Dict[str, Tuple[float, float]]


def squared_distance(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> float:
    """
    Return the squared Euclidean distance between two 2D points.

    Using squared distance avoids the sqrt and is exact for integer
    or rational coordinates such as the 300-400-500 right triangle.
    """
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def euclidean_distance(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> float:
    """Return Euclidean distance between two 2D points."""
    return math.sqrt(squared_distance(p1, p2))


def get_node_order(coords: Coordinates) -> List[str]:
    """
    Return a stable, deterministic ordering of node names.

    Sorts lexicographically: Node_01, Node_02, ..., Node_09, Node_10, ...
    This ensures consistent matrix column ordering across all runs.
    """
    return sorted(coords.keys())


def build_connectivity_graph(
    coords: Coordinates, range_m: float = 500.0
) -> nx.Graph:
    """
    Build and return the undirected connectivity graph G = (V, E).

    An edge (u, v) is added iff the Euclidean distance between u and v
    is **at most** range_m (inclusive boundary).

    Node attributes stored:
        pos : (x, y)  — 2D coordinates

    Edge attributes stored:
        distance : float  — actual Euclidean distance in metres

    Parameters
    ----------
    coords : dict  — {"Node_01": [x, y], ...}
    range_m : float — communication radius in metres (default 500.0)

    Returns
    -------
    nx.Graph
    """
    G = nx.Graph()
    range_sq = range_m * range_m
    nodes = list(coords.keys())

    # Add all nodes first so isolated nodes are preserved
    for name, (x, y) in coords.items():
        G.add_node(name, pos=(float(x), float(y)))

    # Add edges for nodes within range
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            p1 = coords[u]
            p2 = coords[v]
            dist_sq = squared_distance(p1, p2)
            if dist_sq <= range_sq:
                dist = math.sqrt(dist_sq)
                G.add_edge(u, v, distance=round(dist, 4))

    return G


def topology_stats(G: nx.Graph) -> Dict:
    """
    Return a summary dictionary of the connectivity graph.

    Keys: node_count, edge_count, max_degree, min_degree,
          avg_degree, isolated_nodes, is_connected.
    """
    degrees = dict(G.degree())
    isolated = [n for n, d in degrees.items() if d == 0]
    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "max_degree": max(degrees.values()) if degrees else 0,
        "min_degree": min(degrees.values()) if degrees else 0,
        "avg_degree": round(
            sum(degrees.values()) / len(degrees), 3
        ) if degrees else 0.0,
        "isolated_nodes": isolated,
        "is_connected": nx.is_connected(G),
    }
