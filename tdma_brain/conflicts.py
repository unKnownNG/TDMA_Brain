"""
conflicts.py — Build the Distance-2 conflict graph H = G².

Two nodes u, v conflict iff d_G(u, v) ∈ {1, 2}.
That is: they are either direct neighbours OR share a common neighbour.

This models two distinct interference scenarios:
  Distance-1 (Direct):      A ←→ B  →  A and B cannot transmit simultaneously.
  Distance-2 (Hidden-term): A → C ← B  →  A and B both reach receiver C; they
                             cannot transmit simultaneously or C loses both packets.

We use nx.power(G, 2) which correctly implements this via graph-theoretic
path length — NOT a 1000 m Euclidean radius.  Two nodes 800 m apart do NOT
conflict unless an intermediate relay actually exists between them.
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple

from .topology import Coordinates


def build_conflict_graph(G: nx.Graph) -> nx.Graph:
    """
    Build the Distance-2 conflict graph H = G².

    For every edge (u, v) in H we annotate:
        conflict_type : "1-hop" | "2-hop"
        via_nodes     : List[str]  — all shared neighbours (2-hop only)
        via_node      : str | None — canonical first witness (for convenience)
        distance      : float  — Euclidean distance between u and v (from G)

    Parameters
    ----------
    G : nx.Graph — connectivity graph (from topology.build_connectivity_graph)

    Returns
    -------
    nx.Graph — conflict graph H
    """
    # nx.power(G, 2) adds an edge (u,v) if shortest path ≤ 2 in G
    H = nx.power(G, 2)

    # Annotate edges with conflict type and shared neighbour
    for u, v in H.edges():
        if G.has_edge(u, v):
            # 1-hop: direct neighbours
            dist = G[u][v].get("distance", 0.0)
            H[u][v]["conflict_type"] = "1-hop"
            H[u][v]["via_nodes"] = []
            H[u][v]["via_node"] = None
            H[u][v]["distance"] = dist
        else:
            # 2-hop: find ALL shared neighbours (not just the first)
            shared = sorted(nx.common_neighbors(G, u, v))
            # Use Euclidean distance between u and v for reference
            pos_u = G.nodes[u].get("pos", (0, 0))
            pos_v = G.nodes[v].get("pos", (0, 0))
            dist = (
                ((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2)
                ** 0.5
            )
            H[u][v]["conflict_type"] = "2-hop"
            H[u][v]["via_nodes"] = shared
            H[u][v]["via_node"] = shared[0] if shared else None
            H[u][v]["distance"] = round(dist, 4)

    # Copy node coordinates from G to H for downstream use
    for n, data in G.nodes(data=True):
        H.nodes[n].update(data)

    return H


def explain_conflict(G: nx.Graph, H: nx.Graph, u: str, v: str) -> str:
    """
    Return a human-readable string explaining why u and v conflict in H.

    Examples
    --------
    "Node_03 and Node_07 conflict: 1-hop neighbours (distance 400m)"
    "Node_03 and Node_07 conflict: 2-hop via Node_05 (400m + 400m)"
    """
    if not H.has_edge(u, v):
        return f"{u} and {v} do NOT conflict."

    edge_data = H[u][v]
    ctype = edge_data.get("conflict_type", "unknown")

    if ctype == "1-hop":
        dist = edge_data.get("distance", 0.0)
        return (
            f"{u} and {v} conflict: 1-hop neighbours "
            f"(distance {dist:.0f}m)"
        )
    else:
        via_nodes = edge_data.get("via_nodes", [])
        # Fall back to legacy via_node if via_nodes not present
        if not via_nodes:
            via = edge_data.get("via_node")
            via_nodes = [via] if via else []
        if via_nodes:
            # Report all shared relay nodes
            via_parts = []
            for via in via_nodes:
                if G.has_edge(u, via) and G.has_edge(via, v):
                    d1 = G[u][via].get("distance", 0.0)
                    d2 = G[via][v].get("distance", 0.0)
                    via_parts.append(f"{via} ({d1:.0f}m + {d2:.0f}m)")
            if via_parts:
                return (
                    f"{u} and {v} conflict: 2-hop via "
                    f"{', '.join(via_parts)}"
                )
        dist = edge_data.get("distance", 0.0)
        return (
            f"{u} and {v} conflict: 2-hop "
            f"(direct distance {dist:.0f}m)"
        )


def get_all_shared_neighbours(
    G: nx.Graph, u: str, v: str
) -> List[str]:
    """Return all nodes that are neighbours of both u and v in G."""
    return list(nx.common_neighbors(G, u, v))


def conflict_stats(H: nx.Graph) -> Dict:
    """
    Return a summary of the conflict graph.

    Keys: node_count, conflict_edges, one_hop_edges, two_hop_edges,
          max_conflict_degree, chromatic_lower_bound_delta.
    """
    one_hop = sum(
        1 for u, v, d in H.edges(data=True) if d.get("conflict_type") == "1-hop"
    )
    two_hop = sum(
        1 for u, v, d in H.edges(data=True) if d.get("conflict_type") == "2-hop"
    )
    degrees = dict(H.degree())
    max_deg = max(degrees.values()) if degrees else 0
    return {
        "node_count": H.number_of_nodes(),
        "conflict_edges": H.number_of_edges(),
        "one_hop_edges": one_hop,
        "two_hop_edges": two_hop,
        "max_conflict_degree": max_deg,
        "chromatic_lower_bound_delta": max_deg + 1,
    }
