# backend/app/graph_viz.py
import io
import re
from typing import Dict, Optional, List

import matplotlib
matplotlib.use("Agg")  # headless environments

import matplotlib.pyplot as plt
import networkx as nx


DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


def _infer_node_type(node_id: str) -> str:
    """
    Infer a coarse node type from the node id string.

    Examples:
        "household:jeong-family"                -> "household"
        "person:yooni"                          -> "person"
        "day:yooni:2025-01-01"                  -> "day"
        "meal:yooni:2025-01-01:breakfast"       -> "meal"
        "food:banana"                           -> "food"
        "activity:yooni:2025-01-01:walk"        -> "activity"
        "sleep:yooni:2025-01-01"                -> "sleep"
    """
    if isinstance(node_id, str) and ":" in node_id:
        return node_id.split(":", 1)[0]
    return "other"


def _extract_date(node_id: str) -> Optional[str]:
    """
    Extract a YYYY-MM-DD date string from the node id if present.
    Returns None if no date is found.
    """
    if not isinstance(node_id, str):
        return None
    m = DATE_PATTERN.search(node_id)
    if m:
        return m.group(0)
    return None


def _short_label(node_id: str) -> str:
    """
    Produce a shorter, more readable label for visualization.

    Strategy:
        - For day nodes: show just the date (2025-01-01)
        - For meal nodes: show meal type (breakfast / lunch / dinner / snack)
        - For others: show the last segment after ':'
    """
    if not isinstance(node_id, str):
        return str(node_id)

    parts = node_id.split(":")

    if len(parts) == 1:
        return node_id

    node_type = parts[0]

    if node_type == "day":
        # day:yooni:2025-01-01 -> 2025-01-01
        date_str = _extract_date(node_id)
        return date_str if date_str is not None else parts[-1]

    if node_type == "meal":
        # meal:yooni:2025-01-01:breakfast -> breakfast
        return parts[-1]

    # Default: last segment
    return parts[-1]


def _compute_vertical_timeline_layout(G: nx.Graph) -> Dict[str, List[float]]:
    """
    Compute a vertical timeline layout.

    - Y axis: time (date) flowing downward
    - X axis: node type columns
      household | person | day | activity | meal | sleep | food

    Nodes without a date (household, person, some foods) are placed
    at aggregated positions (top row for household/person, average
    of neighbor dates for foods).
    """
    node_types: Dict[str, str] = {n: _infer_node_type(n) for n in G.nodes}

    # 1. Collect all dates from nodes that have one
    dates: List[str] = []
    for n in G.nodes:
        d = _extract_date(str(n))
        if d is not None:
            dates.append(d)

    unique_dates = sorted(set(dates))
    date_to_index: Dict[str, int] = {d: i for i, d in enumerate(unique_dates)}

    # If we somehow have no dates at all, fall back to spring layout
    if not unique_dates:
        return nx.spring_layout(G, seed=42)

    # Vertical spacing between days
    y_step = -1.5  # negative so time flows downward
    base_y = 0.0

    def y_for_date(date_str: str) -> float:
        idx = date_to_index[date_str]
        return base_y + y_step * idx

    # 2. Define x-position per node type (columns)
    type_to_x = {
        "household": -2.0,
        "person": -1.0,
        "day": 0.0,
        "activity": 1.0,
        "meal": 2.0,
        "sleep": 3.0,
        "food": 4.0,
        "other": 5.0,
    }

    pos: Dict[str, List[float]] = {}

    # 3. First place day nodes (they define the main timeline)
    for n in G.nodes:
        t = node_types[n]
        if t == "day":
            date_str = _extract_date(str(n))
            if date_str is None:
                continue
            x = type_to_x.get("day", 0.0)
            y = y_for_date(date_str)
            pos[n] = [x, y]

    # 4. Place dated event nodes (activity, meal, sleep)
    for n in G.nodes:
        t = node_types[n]
        if t in {"activity", "meal", "sleep"}:
            date_str = _extract_date(str(n))
            if date_str is None:
                continue
            x = type_to_x.get(t, type_to_x["other"])
            y = y_for_date(date_str)

            # small horizontal jitter to avoid perfect overlap
            jitter = (hash(n) % 7) * 0.04
            pos[n] = [x + jitter, y]

    # 5. Place food nodes: align with the average of their neighbors' dates
    for n in G.nodes:
        t = node_types[n]
        if t != "food":
            continue

        neighbor_dates = [
            _extract_date(str(nb))
            for nb in G.neighbors(n)
            if _extract_date(str(nb)) is not None
        ]
        neighbor_dates = [d for d in neighbor_dates if d is not None]

        if neighbor_dates:
            # average date index
            idx_values = [date_to_index[d] for d in neighbor_dates]
            avg_idx = sum(idx_values) / len(idx_values)
            y = base_y + y_step * avg_idx
        else:
            # if no dated neighbors, put them near the top
            y = base_y + y_step * (-1)

        x = type_to_x.get("food", type_to_x["other"])
        jitter = (hash(n) % 5) * 0.05
        pos[n] = [x + jitter, y]

    # 6. Place household and person nodes near the top above the first day
    min_y = min(v[1] for v in pos.values())

    for n in G.nodes:
        t = node_types[n]
        if t in {"household", "person"}:
            x = type_to_x.get(t, type_to_x["other"])
            # place them above the earliest date row
            y = min_y + 1.5 if t == "person" else min_y + 2.5
            pos[n] = [x, y]

    # 7. Any remaining nodes (other) get a default position
    for n in G.nodes:
        if n not in pos:
            t = node_types[n]
            x = type_to_x.get(t, type_to_x["other"])
            y = base_y
            pos[n] = [x, y]

    return pos


def graph_to_svg(G: nx.Graph) -> bytes:
    """
    Render a NetworkX graph as an SVG image and return the raw SVG bytes.

    Vertical timeline layout:
        - Y axis encodes date (time flows downward)
        - X axis encodes node type (household, person, day, activity, meal, sleep, food)
    """

    # Layout computation
    pos = _compute_vertical_timeline_layout(G)

    # Prepare styling info
    node_types: Dict[str, str] = {n: _infer_node_type(n) for n in G.nodes}

    type_to_color = {
        "household": "#1f77b4",
        "person": "#ff7f0e",
        "day": "#2ca02c",
        "activity": "#17becf",
        "meal": "#d62728",
    }