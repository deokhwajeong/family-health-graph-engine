# backend/app/graph_viz.py
"""
Graph visualization utilities.
Renders NetworkX graphs as SVG images and provides JSON data for interactive JS visualizations.
"""
import io
import json
import re
from typing import Dict, Optional, List, Any

import matplotlib
matplotlib.use("Agg")  # headless environments

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")

# ─── Shared palette ───
TYPE_COLORS = {
    "household": "#1f77b4",
    "person":    "#ff7f0e",
    "day":       "#2ca02c",
    "activity":  "#17becf",
    "meal":      "#d62728",
    "sleep":     "#9467bd",
    "food":      "#e377c2",
    "metric":    "#bcbd22",
    "other":     "#7f7f7f",
}

TYPE_SIZES = {
    "household": 300,
    "person":    250,
    "day":       100,
    "activity":  80,
    "meal":      60,
    "sleep":     80,
    "food":      50,
    "metric":    40,
    "other":     40,
}


def _infer_node_type(node_id: str) -> str:
    if isinstance(node_id, str) and ":" in node_id:
        return node_id.split(":", 1)[0]
    return "other"


def _extract_date(node_id: str) -> Optional[str]:
    if not isinstance(node_id, str):
        return None
    m = DATE_PATTERN.search(node_id)
    return m.group(0) if m else None


def _short_label(node_id: str) -> str:
    if not isinstance(node_id, str):
        return str(node_id)
    parts = node_id.split(":")
    if len(parts) == 1:
        return node_id
    node_type = parts[0]
    if node_type == "day":
        date_str = _extract_date(node_id)
        return date_str if date_str else parts[-1]
    if node_type == "meal":
        return parts[-1]
    if node_type == "metric":
        return parts[-1] if len(parts) > 1 else node_id
    return parts[-1]


# ═══════════════════════════════════════════
#  Layout computation
# ═══════════════════════════════════════════

def _compute_vertical_timeline_layout(G: nx.Graph) -> Dict[str, List[float]]:
    """Vertical timeline layout: Y = date, X = node type column."""
    node_types = {n: _infer_node_type(n) for n in G.nodes}

    dates = []
    for n in G.nodes:
        d = _extract_date(str(n))
        if d is not None:
            dates.append(d)

    unique_dates = sorted(set(dates))
    date_to_index = {d: i for i, d in enumerate(unique_dates)}

    if not unique_dates:
        return nx.spring_layout(G, seed=42)

    y_step = -1.5
    base_y = 0.0

    def y_for_date(date_str: str) -> float:
        return base_y + y_step * date_to_index[date_str]

    type_to_x = {
        "household": -3.0,
        "person":    -2.0,
        "day":        0.0,
        "sleep":      1.5,
        "activity":   3.0,
        "meal":       4.5,
        "food":       6.5,
        "metric":    -1.0,
        "other":      8.0,
    }

    pos: Dict[str, List[float]] = {}

    # Place day nodes
    for n in G.nodes:
        if node_types[n] == "day":
            d = _extract_date(str(n))
            if d:
                pos[n] = [type_to_x["day"], y_for_date(d)]

    # Place dated event nodes
    counters: Dict[str, int] = {}
    for n in G.nodes:
        t = node_types[n]
        if t in {"activity", "meal", "sleep", "metric"}:
            d = _extract_date(str(n))
            if d is None:
                continue
            x = type_to_x.get(t, type_to_x["other"])
            y = y_for_date(d)
            key = f"{t}:{d}"
            counters[key] = counters.get(key, 0) + 1
            jitter = counters[key] * 0.15
            pos[n] = [x + jitter, y]

    # Place food nodes at average neighbor date
    for n in G.nodes:
        if node_types[n] != "food":
            continue
        neighbor_dates = []
        for nb in list(G.predecessors(n)) + list(G.neighbors(n)):
            d = _extract_date(str(nb))
            if d and d in date_to_index:
                neighbor_dates.append(d)
        if neighbor_dates:
            avg_idx = sum(date_to_index[d] for d in neighbor_dates) / len(neighbor_dates)
            y = base_y + y_step * avg_idx
        else:
            y = base_y + y_step * (-1)
        jitter = (hash(n) % 7) * 0.12
        pos[n] = [type_to_x["food"] + jitter, y]

    # Household and person
    min_y = min(v[1] for v in pos.values()) if pos else 0
    for n in G.nodes:
        t = node_types[n]
        if t in {"household", "person"}:
            x = type_to_x.get(t, 0)
            y = min_y + (2.5 if t == "household" else 1.5)
            jitter = (hash(n) % 5) * 0.3
            pos[n] = [x + jitter, y]

    # Remaining
    for n in G.nodes:
        if n not in pos:
            t = node_types[n]
            x = type_to_x.get(t, type_to_x["other"])
            pos[n] = [x, base_y]

    return pos


# ═══════════════════════════════════════════
#  SVG Rendering (matplotlib)
# ═══════════════════════════════════════════

def graph_to_svg(G: nx.Graph) -> bytes:
    """Render a NetworkX graph as SVG bytes using matplotlib."""

    pos = _compute_vertical_timeline_layout(G)
    node_types = {n: _infer_node_type(n) for n in G.nodes}

    # Group nodes by type for layered drawing
    nodes_by_type: Dict[str, list] = {}
    for n in G.nodes:
        t = node_types[n]
        nodes_by_type.setdefault(t, []).append(n)

    # Figure size based on graph size
    num_dates = len(set(
        _extract_date(str(n)) for n in G.nodes if _extract_date(str(n))
    ))
    fig_h = max(8, min(60, num_dates * 1.2 + 4))
    fig_w = max(14, 18)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#2e3345',
        alpha=0.3,
        width=0.5,
        arrows=True,
        arrowsize=5,
        connectionstyle="arc3,rad=0.05",
    )

    # Draw nodes by type
    for t, nodes in nodes_by_type.items():
        color = TYPE_COLORS.get(t, TYPE_COLORS["other"])
        size = TYPE_SIZES.get(t, 40)
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, ax=ax,
            node_color=color,
            node_size=size,
            alpha=0.85,
            edgecolors='#ffffff22',
            linewidths=0.5,
        )

    # Labels for important nodes only
    label_types = {"household", "person", "day"}
    labels = {n: _short_label(n) for n in G.nodes if node_types[n] in label_types}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=6,
        font_color='#c0c0c0',
        font_weight='normal',
    )

    # Legend
    legend_patches = []
    for t in ["household", "person", "day", "meal", "food",
              "activity", "sleep", "metric"]:
        if t in nodes_by_type:
            count = len(nodes_by_type[t])
            legend_patches.append(
                mpatches.Patch(color=TYPE_COLORS[t], label=f"{t} ({count})")
            )
    ax.legend(
        handles=legend_patches, loc='upper right',
        fontsize=8, framealpha=0.5,
        facecolor='#1a1d27', edgecolor='#2e3345',
        labelcolor='#e4e6f0',
    )

    ax.set_title(
        f"Health Graph — {len(G.nodes())} nodes, {len(G.edges())} edges",
        color='#e4e6f0', fontsize=12, pad=15,
    )
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", facecolor='#0f1117')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════
#  Interactive Graph JSON (for D3.js frontend)
# ═══════════════════════════════════════════

def graph_to_d3_json(G: nx.Graph, max_nodes: int = 500) -> Dict[str, Any]:
    """
    Convert a NetworkX graph to a D3.js force-directed graph JSON.
    Samples nodes if graph is too large.
    Returns { nodes: [...], links: [...], stats: {...} }
    """
    node_types = {n: _infer_node_type(n) for n in G.nodes}

    # If graph is too large, sample a coherent subgraph
    all_nodes = list(G.nodes())
    if len(all_nodes) > max_nodes:
        priority = {"household": 0, "person": 1, "day": 2, "metric": 3,
                     "sleep": 4, "activity": 5, "meal": 6, "food": 7, "other": 8}
        all_nodes.sort(key=lambda n: (priority.get(node_types[n], 9), n))

        sampled: set = set()
        for n in all_nodes:
            if node_types[n] in {"household", "person"}:
                sampled.add(n)

        # Add recent day nodes and their neighbors
        day_nodes = [n for n in all_nodes if node_types[n] == "day"]
        day_nodes.sort(key=lambda n: _extract_date(str(n)) or "", reverse=True)
        for dn in day_nodes[:max_nodes // 10]:
            sampled.add(dn)
            for nb in list(G.successors(dn)) + list(G.predecessors(dn)):
                sampled.add(nb)
                if len(sampled) >= max_nodes:
                    break
            if len(sampled) >= max_nodes:
                break

        G = G.subgraph(sampled).copy()

    nodes = []
    for n in G.nodes():
        t = _infer_node_type(n)
        d = G.nodes[n]
        nodes.append({
            "id": n,
            "label": _short_label(n),
            "type": t,
            "color": TYPE_COLORS.get(t, TYPE_COLORS["other"]),
            "size": max(3, TYPE_SIZES.get(t, 40) // 20),
            "name": d.get("name", ""),
            "date": _extract_date(str(n)),
        })

    links = []
    for u, v, d in G.edges(data=True):
        links.append({
            "source": u,
            "target": v,
            "type": d.get("type", ""),
        })

    type_counts = {}
    for n in nodes:
        t = n["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "nodes": nodes,
        "links": links,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "node_types": type_counts,
        }
    }