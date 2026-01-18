# Household Health Graph Schema

Version: 0.1.0

This document defines the base graph schema used in the Household Health Graph Engine.
The goals are twofold:

1. Represent a family member's daily life pattern using a graph structure.
2. Make it directly usable for Graph ML or basic graph analysis (NetworkX, PyG, etc).

---

## 1. Graph Basics

The graph follows a **property graph** model.

* All nodes share common attributes:

  * `id: str` unique identifier
  * `type: str` node type
    Examples: `"household"`, `"person"`, `"day"`, `"meal"`, `"food"`, `"sleep"`, `"activity"`, `"metric"`

* All edges share common attributes:

  * `type: str` edge type
    Examples: `"HAS_MEMBER"`, `"HAS_DAY"`, `"HAS_MEAL"`, `"CONTAINS_FOOD"`
  * `weight: float | None` optional

Nodes and edges are expressed using NetworkX:
`G.add_node(id, **attrs)` and `G.add_edge(u, v, **attrs)`.

---

## 2. Node Types

### 2.1 household

Represents a family unit.

Required attributes:

* `id: str`              for example: `"household:smith-family"`
* `type: "household"`
* `name: str`
* `timezone: str`        for example: `"America/Los_Angeles"`

Optional:

* `notes: str`

---

### 2.2 person

A family member.

Required attributes:

* `id: str`              for example: `"person:yooni"`, `"person:jun"`
* `type: "person"`
* `household_id: str`
* `name: str`
* `role: str`            for example: `"child"`, `"parent"`

Optional:

* `birth_year: int`
* `gender: str`
* `notes: str`

---

### 2.3 day

A day node for a specific person.
Each day is one node per person.

Required:

* `id: str`              for example: `"day:yooni:2025-01-01"`
* `type: "day"`
* `person_id: str`
* `date: str`            ISO format `"YYYY-MM-DD"`
* `weekday: int`         Monday=0, Sunday=6

Optional:

* `is_school_day: bool`
* `is_weekend: bool`

---

### 2.4 meal

A meal event.

Required:

* `id: str`              for example: `"meal:yooni:2025-01-01:breakfast"`
* `type: "meal"`
* `person_id: str`
* `date: str`
* `meal_type: str`       `"breakfast"`, `"lunch"`, `"dinner"`, `"snack"`

Optional:

* `start_time: str`
* `total_kcal: float | None`
* `fiber_g: float | None`
* `sugar_g: float | None`
* `notes: str`

---

### 2.5 food

Individual food item within a meal.

Required:

* `id: str`              for example: `"food:banana"`, `"food:greek-yogurt"`
* `type: "food"`
* `name: str`

Optional:

* `category: str`        for example: `"fruit"`, `"dairy"`, `"grain"`
* `kcal: float | None`
* `fiber_g: float | None`
* `sugar_g: float | None`
* `protein_g: float | None`
* `fat_g: float | None`

Food nodes are reusable catalog items.

---

### 2.6 sleep

Sleep session.

Required:

* `id: str`              for example: `"sleep:yooni:2024-12-03"`
* `type: "sleep"`
* `person_id: str`
* `date: str`            typically wake-up date
* `start_time: str`
* `end_time: str`

Optional:

* `duration_hours: float`
* `quality_score: float`
* `notes: str`

---

### 2.7 activity

Activity such as exercise, screen time, or special events.

Required:

* `id: str`              for example: `"activity:yooni:2024-12-03:fencing"`
* `type: "activity"`
* `person_id: str`
* `date: str`
* `name: str`

Optional:

* `category: str`        `"sport"`, `"screen"`, `"study"`, `"other"`
* `duration_min: int | None`
* `intensity: str | None`
* `notes: str`

---

### 2.8 metric

A single numeric measurement such as weight or glucose.

Required:

* `id: str`              example: `"metric:yooni:weight:2024-12-03"`
* `type: "metric"`
* `person_id: str`
* `metric_type: str`
* `value: float`
* `unit: str`
* `timestamp: str`       ISO datetime `"YYYY-MM-DDTHH:MM"`

---

## 3. Edge Types

### 3.1 household relations

* `household` -> `person`

  * `type: "HAS_MEMBER"`

### 3.2 person and day

* `person` -> `day`

  * `type: "HAS_DAY"`

* `day` -> `day`
  for sequential days within one person

  * `type: "NEXT_DAY"`

### 3.3 day and events

* `day` -> `meal`

  * `type: "HAS_MEAL"`

* `meal` -> `food`

  * `type: "CONTAINS_FOOD"`

* `day` -> `sleep`

  * `type: "HAS_SLEEP"`

* `day` -> `activity`

  * `type: "HAS_ACTIVITY"`

* `person` -> `metric`

  * `type: "HAS_METRIC"`

---

## 4. ID Rules

Human readable, no UUID required.

Examples:

* household: `"household:jeong-family"`
* person: `"person:yooni"`
* day: `"day:yooni:2025-01-01"`
* meal: `"meal:yooni:2025-01-01:breakfast"`
* food: `"food:banana"`
* sleep: `"sleep:yooni:2025-01-01"`
* activity: `"activity:yooni:2025-01-01:fencing"`
* metric: `"metric:yooni:weight:2025-01-01"`

This allows simple parsing to extract person, date, and type.

---

## 5. Example Graph Snippet

Example: Yooni eats yogurt and banana for breakfast and does fencing.

```python
G.add_node(
    "household:jeong-family",
    type="household",
    name="Jeong family",
    timezone="America/Los_Angeles",
)

G.add_node(
    "person:yooni",
    type="person",
    household_id="household:jeong-family",
    name="Yooni",
    role="child",
)

G.add_node(
    "day:yooni:2025-01-01",
    type="day",
    person_id="person:yooni",
    date="2025-01-01",
    weekday=2,
    is_school_day=True,
)
```

Edges:

```python
G.add_edge("household:jeong-family", "person:yooni", type="HAS_MEMBER")
G.add_edge("person:yooni", "day:yooni:2025-01-01", type="HAS_DAY")
G.add_edge("day:yooni:2025-01-01", "meal:yooni:2025-01-01:breakfast", type="HAS_MEAL")
G.add_edge("meal:yooni:2025-01-01:breakfast", "food:greek-yogurt", type="CONTAINS_FOOD")
G.add_edge("day:yooni:2025-01-01", "activity:yooni:2025-01-01:fencing", type="HAS_ACTIVITY")
```

---

## 6. Next Actions

1. Open `docs/graph-schema.md` and paste all content above.
2. Save the file so git registers the change.

Next we will implement `graph_builder.py` in `backend/app` following this schema.
