"""
Graph builder for the Household Health Graph Engine.

This module follows the schema defined in docs/graph-schema.md.
It uses a property graph on top of NetworkX.
"""

from typing import Optional, Dict, Any
import networkx as nx


# Node and edge type constants (to avoid typo hell)
NODE_HOUSEHOLD = "household"
NODE_PERSON = "person"
NODE_DAY = "day"
NODE_MEAL = "meal"
NODE_FOOD = "food"
NODE_SLEEP = "sleep"
NODE_ACTIVITY = "activity"
NODE_METRIC = "metric"

EDGE_HAS_MEMBER = "HAS_MEMBER"
EDGE_HAS_DAY = "HAS_DAY"
EDGE_NEXT_DAY = "NEXT_DAY"
EDGE_HAS_MEAL = "HAS_MEAL"
EDGE_CONTAINS_FOOD = "CONTAINS_FOOD"
EDGE_HAS_SLEEP = "HAS_SLEEP"
EDGE_HAS_ACTIVITY = "HAS_ACTIVITY"
EDGE_HAS_METRIC = "HAS_METRIC"


class GraphBuilder:
    """
    Thin wrapper around a NetworkX DiGraph that enforces
    the schema in docs/graph-schema.md as much as possible.
    """

    def __init__(self) -> None:
        # Directed graph is more natural for these relations
        self.G: nx.DiGraph = nx.DiGraph()

    # ------------- generic helpers -------------

    def add_node(self, node_id: str, node_type: str, **attrs: Any) -> None:
        """Add a node with a required id and type."""
        if not node_id:
            raise ValueError("node_id is required")

        if "type" in attrs and attrs["type"] != node_type:
            raise ValueError(f"type mismatch for node {node_id}")

        attrs.setdefault("type", node_type)
        self.G.add_node(node_id, **attrs)

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        weight: Optional[float] = None,
        **attrs: Any,
    ) -> None:
        """Add an edge with a required type and optional weight."""
        if "type" in attrs and attrs["type"] != edge_type:
            raise ValueError(f"type mismatch for edge {source} -> {target}")

        attrs.setdefault("type", edge_type)
        if weight is not None:
            attrs.setdefault("weight", weight)

        self.G.add_edge(source, target, **attrs)

    # ------------- node builders -------------

    def add_household(
        self,
        household_id: str,
        name: str,
        timezone: str,
        **extra: Any,
    ) -> str:
        self.add_node(
            household_id,
            NODE_HOUSEHOLD,
            name=name,
            timezone=timezone,
            **extra,
        )
        return household_id

    def add_person(
        self,
        person_id: str,
        household_id: str,
        name: str,
        role: str,
        **extra: Any,
    ) -> str:
        self.add_node(
            person_id,
            NODE_PERSON,
            household_id=household_id,
            name=name,
            role=role,
            **extra,
        )
        # household -> person
        self.add_edge(household_id, person_id, EDGE_HAS_MEMBER)
        return person_id

    def add_day(
        self,
        day_id: str,
        person_id: str,
        date: str,
        weekday: int,
        prev_day_id: Optional[str] = None,
        **extra: Any,
    ) -> str:
        self.add_node(
            day_id,
            NODE_DAY,
            person_id=person_id,
            date=date,
            weekday=weekday,
            **extra,
        )
        # person -> day
        self.add_edge(person_id, day_id, EDGE_HAS_DAY)

        # previous day linkage for the same person
        if prev_day_id:
            self.add_edge(prev_day_id, day_id, EDGE_NEXT_DAY)

        return day_id

    def add_meal(
        self,
        meal_id: str,
        person_id: str,
        date: str,
        meal_type: str,
        day_id: Optional[str] = None,
        **extra: Any,
    ) -> str:
        self.add_node(
            meal_id,
            NODE_MEAL,
            person_id=person_id,
            date=date,
            meal_type=meal_type,
            **extra,
        )

        if day_id:
            self.add_edge(day_id, meal_id, EDGE_HAS_MEAL)

        return meal_id

    def add_food(
        self,
        food_id: str,
        name: str,
        **extra: Any,
    ) -> str:
        self.add_node(
            food_id,
            NODE_FOOD,
            name=name,
            **extra,
        )
        return food_id

    def link_food_to_meal(
        self,
        meal_id: str,
        food_id: str,
        **extra: Any,
    ) -> None:
        self.add_edge(meal_id, food_id, EDGE_CONTAINS_FOOD, **extra)

    def add_sleep(
        self,
        sleep_id: str,
        person_id: str,
        date: str,
        start_time: str,
        end_time: str,
        day_id: Optional[str] = None,
        **extra: Any,
    ) -> str:
        self.add_node(
            sleep_id,
            NODE_SLEEP,
            person_id=person_id,
            date=date,
            start_time=start_time,
            end_time=end_time,
            **extra,
        )

        if day_id:
            self.add_edge(day_id, sleep_id, EDGE_HAS_SLEEP)

        return sleep_id

    def add_activity(
        self,
        activity_id: str,
        person_id: str,
        date: str,
        name: str,
        day_id: Optional[str] = None,
        **extra: Any,
    ) -> str:
        self.add_node(
            activity_id,
            NODE_ACTIVITY,
            person_id=person_id,
            date=date,
            name=name,
            **extra,
        )

        if day_id:
            self.add_edge(day_id, activity_id, EDGE_HAS_ACTIVITY)

        return activity_id

    def add_metric(
        self,
        metric_id: str,
        person_id: str,
        metric_type: str,
        value: float,
        unit: str,
        timestamp: str,
        **extra: Any,
    ) -> str:
        self.add_node(
            metric_id,
            NODE_METRIC,
            person_id=person_id,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=timestamp,
            **extra,
        )

        # person -> metric (no day node required)
        self.add_edge(person_id, metric_id, EDGE_HAS_METRIC)

        return metric_id


# ------------- convenience factory -------------

def build_example_graph() -> nx.DiGraph:
    """
    Build the example from docs/graph-schema.md

    Yooni has a household, one day, breakfast with greek yogurt,
    and a fencing activity.
    """
    gb = GraphBuilder()

    household_id = "household:jeong-family"
    person_id = "person:yooni"
    day_id = "day:yooni:2025-01-01"
    meal_id = "meal:yooni:2025-01-01:breakfast"
    food_id = "food:greek-yogurt"
    activity_id = "activity:yooni:2025-01-01:fencing"

    gb.add_household(household_id, name="Jeong family", timezone="America/Los_Angeles")
    gb.add_person(person_id, household_id=household_id, name="Yooni", role="child")

    gb.add_day(
        day_id,
        person_id=person_id,
        date="2025-01-01",
        weekday=2,
        is_school_day=True,
    )

    gb.add_meal(
        meal_id,
        person_id=person_id,
        date="2025-01-01",
        meal_type="breakfast",
        day_id=day_id,
    )

    gb.add_food(
        food_id,
        name="Greek yogurt",
        category="dairy",
    )
    gb.link_food_to_meal(meal_id, food_id)

    gb.add_activity(
        activity_id,
        person_id=person_id,
        date="2025-01-01",
        name="fencing",
        day_id=day_id,
        category="sport",
    )

    return gb.G
