"""
Synthetic data generator for the Household Health Graph Engine.

Builds a fictional four-person household ("Demo family": Mia, Leo, Alex,
Jordan) with daily meals, sleep, activities, and metrics. Every value is
generated; nothing here comes from a real person.

Data for a given (person, date) is seeded from that pair, so the same day
looks identical no matter which window size an endpoint asks for.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional
import random

import networkx as nx

from .graph_builder import (
    GraphBuilder,
)


def _ensure_food_catalog(gb: GraphBuilder) -> None:
    """
    Define a comprehensive reusable food catalog with nutritional details.
    """
    foods = [
        {
            "id": "food:greek-yogurt",
            "name": "Greek yogurt",
            "category": "dairy",
            "protein_g": 10,
            "fat_g": 0,
            "fiber_g": 0,
            "sugar_g": 5,
            "kcal": 100,
        },
        {
            "id": "food:banana",
            "name": "Banana",
            "category": "fruit",
            "protein_g": 1,
            "fat_g": 0,
            "fiber_g": 3,
            "sugar_g": 12,
            "kcal": 90,
        },
        {
            "id": "food:apple",
            "name": "Apple",
            "category": "fruit",
            "protein_g": 0,
            "fat_g": 0,
            "fiber_g": 4,
            "sugar_g": 19,
            "kcal": 95,
        },
        {
            "id": "food:orange",
            "name": "Orange",
            "category": "fruit",
            "protein_g": 1,
            "fat_g": 0,
            "fiber_g": 3,
            "sugar_g": 12,
            "kcal": 62,
        },
        {
            "id": "food:rice",
            "name": "Rice",
            "category": "grain",
            "kcal": 200,
            "protein_g": 4,
            "fat_g": 1,
            "fiber_g": 1,
        },
        {
            "id": "food:whole-wheat-bread",
            "name": "Whole wheat bread",
            "category": "grain",
            "kcal": 150,
            "protein_g": 6,
            "fat_g": 2,
            "fiber_g": 3,
            "sugar_g": 1,
        },
        {
            "id": "food:chicken",
            "name": "Chicken breast",
            "category": "protein",
            "kcal": 160,
            "protein_g": 30,
            "fat_g": 3,
        },
        {
            "id": "food:salmon",
            "name": "Salmon",
            "category": "protein",
            "kcal": 280,
            "protein_g": 25,
            "fat_g": 17,
            "omega3_mg": 2260,
        },
        {
            "id": "food:eggs",
            "name": "Eggs",
            "category": "protein",
            "kcal": 155,
            "protein_g": 13,
            "fat_g": 11,
        },
        {
            "id": "food:broccoli",
            "name": "Broccoli",
            "category": "vegetable",
            "kcal": 30,
            "fiber_g": 3,
            "protein_g": 2,
            "vitamin_c_mg": 90,
        },
        {
            "id": "food:spinach",
            "name": "Spinach",
            "category": "vegetable",
            "kcal": 23,
            "fiber_g": 2,
            "protein_g": 3,
            "iron_mg": 2.7,
        },
        {
            "id": "food:carrot",
            "name": "Carrot",
            "category": "vegetable",
            "kcal": 41,
            "fiber_g": 3,
            "protein_g": 1,
            "vitamin_a_mcg": 961,
        },
        {
            "id": "food:milk",
            "name": "Milk",
            "category": "dairy",
            "kcal": 150,
            "protein_g": 8,
            "fat_g": 8,
            "calcium_mg": 300,
        },
        {
            "id": "food:cheese",
            "name": "Cheese",
            "category": "dairy",
            "kcal": 400,
            "protein_g": 25,
            "fat_g": 33,
            "calcium_mg": 721,
        },
        {
            "id": "food:almonds",
            "name": "Almonds",
            "category": "nuts",
            "kcal": 579,
            "protein_g": 21,
            "fat_g": 50,
            "fiber_g": 12,
        },
        {
            "id": "food:oatmeal",
            "name": "Oatmeal",
            "category": "grain",
            "kcal": 150,
            "protein_g": 5,
            "fat_g": 3,
            "fiber_g": 8,
        },
    ]
    
    for food_data in foods:
        food_id = food_data.pop("id")
        gb.add_food(food_id, **food_data)


DEMO_HOUSEHOLD_ID = "household:demo-family"
DEMO_END_DATE = "2025-07-13"

# Fictional demo members. Baselines differ so cross-person patterns show up.
DEMO_MEMBERS = [
    {
        "key": "leo", "name": "Leo", "role": "child", "age": 11, "emoji": "\U0001F466",
        "sleep_school": (9.4, 0.6), "sleep_weekend": (10.0, 0.5),
        "steps": 9500, "rhr": 82,
        "sport": ("fencing", (1, 3), 60, "high", 350),
    },
    {
        "key": "mia", "name": "Mia", "role": "child", "age": 9, "emoji": "\U0001F467",
        "sleep_school": (8.9, 0.7), "sleep_weekend": (9.6, 0.6),
        "steps": 8800, "rhr": 78,
        "sport": ("swimming", (0, 2, 4), 50, "high", 320),
    },
    {
        "key": "alex", "name": "Alex", "role": "parent", "age": 43, "emoji": "\U0001F468",
        "sleep_school": (6.8, 0.6), "sleep_weekend": (7.6, 0.6),
        "steps": 7200, "rhr": 64,
        "sport": ("strength training", (0, 2, 4), 45, "high", 300),
    },
    {
        "key": "jordan", "name": "Jordan", "role": "parent", "age": 41, "emoji": "\U0001F469",
        "sleep_school": (7.1, 0.5), "sleep_weekend": (7.8, 0.5),
        "steps": 8100, "rhr": 66,
        "sport": ("yoga", (1, 3, 5), 45, "medium", 180),
    },
]

_BREAKFAST = [
    ["food:greek-yogurt", "food:banana"],
    ["food:oatmeal", "food:apple"],
    ["food:eggs", "food:whole-wheat-bread", "food:orange"],
    ["food:milk", "food:whole-wheat-bread", "food:banana"],
]
_LUNCH = [
    ["food:rice", "food:chicken", "food:broccoli"],
    ["food:rice", "food:salmon", "food:spinach"],
    ["food:whole-wheat-bread", "food:chicken", "food:carrot"],
    ["food:rice", "food:eggs", "food:broccoli"],
]
_DINNER = [
    ["food:rice", "food:chicken", "food:spinach"],
    ["food:salmon", "food:broccoli"],
    ["food:rice", "food:eggs", "food:carrot"],
    ["food:chicken", "food:whole-wheat-bread", "food:apple"],
]
_WEEKEND_ACTIVITIES = [
    ("family walk", 30, "low", 100),
    ("bike ride", 45, "medium", 250),
    ("park play", 60, "high", 300),
]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _add_member_day(gb: GraphBuilder, member: dict, d: date, prev_day_id: Optional[str],
                    prev_state: Optional[dict], household_rough_night: bool) -> tuple:
    """
    Add one synthetic day (meals, sleep, activity, metrics) for one member.

    Yesterday's behaviour (`prev_state`) carries over into today on purpose, so
    the lagged-correlation analysis has real cross-day signal to find:
    more sugar -> worse sleep, activity -> longer sleep, better sleep -> better focus.
    """
    prev_state = prev_state or {"quality": 80.0, "sugar": 40.0, "active": False}
    key = member["key"]
    person_id = f"person:{key}"
    iso = d.isoformat()
    rng = random.Random(f"{key}:{iso}")
    is_weekend = d.weekday() >= 5
    day_id = f"day:{key}:{iso}"

    gb.add_day(
        day_id=day_id,
        person_id=person_id,
        date=iso,
        weekday=d.weekday(),
        is_school_day=not is_weekend,
        is_weekend=is_weekend,
        prev_day_id=prev_day_id,
    )

    # Meals
    for meal_type, start_time, options in (
        ("breakfast", "07:30", _BREAKFAST),
        ("lunch", "12:30", _LUNCH),
        ("dinner", "18:30", _DINNER),
    ):
        meal_id = f"meal:{key}:{iso}:{meal_type}"
        gb.add_meal(meal_id, person_id=person_id, date=iso, meal_type=meal_type,
                    day_id=day_id, start_time=start_time)
        for food_id in rng.choice(options):
            gb.link_food_to_meal(meal_id, food_id)
    had_snack = rng.random() < 0.35
    if had_snack:
        snack_id = f"meal:{key}:{iso}:snack"
        gb.add_meal(snack_id, person_id=person_id, date=iso, meal_type="snack",
                    day_id=day_id, start_time="15:30")
        for food_id in rng.choice([["food:almonds"], ["food:cheese", "food:apple"], ["food:banana"]]):
            gb.link_food_to_meal(snack_id, food_id)
    sugar = sum(
        gb.G.nodes[f].get("sugar_g", 0)
        for meal in gb.G.successors(day_id) if gb.G.nodes[meal].get("type") == "meal"
        for f in gb.G.successors(meal)
    )

    # Sleep. A shared "rough night" (for example a sick child) cuts everyone's sleep.
    mean, sd = member["sleep_weekend"] if is_weekend else member["sleep_school"]
    duration = rng.gauss(mean, sd) + (0.5 if prev_state["active"] else 0.0)
    quality = rng.gauss(86 if is_weekend else 81, 4) - (prev_state["sugar"] - 40) * 0.35
    if household_rough_night:
        duration -= rng.uniform(1.5, 2.5)
        quality -= rng.uniform(15, 25)
    duration = _clamp(duration, 4.0, 11.5)
    quality = _clamp(quality, 35, 100)
    gb.add_sleep(sleep_id=f"sleep:{key}:{iso}", person_id=person_id, date=iso,
                 start_time="21:30" if member["role"] == "child" else "23:00",
                 end_time="07:00" if member["role"] == "child" else "06:30",
                 day_id=day_id, duration_hours=round(duration, 2),
                 quality_score=round(quality, 1))

    # Activity
    sport, sport_days, sport_min, sport_intensity, sport_kcal = member["sport"]
    activity_min = 0
    if d.weekday() in sport_days:
        gb.add_activity(activity_id=f"activity:{key}:{iso}:{sport.replace(' ', '-')}", person_id=person_id,
                        date=iso, name=sport, day_id=day_id, category="sport",
                        duration_min=sport_min, intensity=sport_intensity,
                        calories_burned=sport_kcal)
        activity_min = sport_min
    elif is_weekend:
        name, minutes, intensity, kcal = rng.choice(_WEEKEND_ACTIVITIES)
        gb.add_activity(activity_id=f"activity:{key}:{iso}:{name.replace(' ', '-')}",
                        person_id=person_id, date=iso, name=name, day_id=day_id,
                        category="leisure", duration_min=minutes, intensity=intensity,
                        calories_burned=kcal)
        activity_min = minutes
    elif rng.random() > 0.4:
        gb.add_activity(activity_id=f"activity:{key}:{iso}:walk", person_id=person_id,
                        date=iso, name="walk", day_id=day_id, category="commute",
                        duration_min=20, intensity="low", calories_burned=60)
        activity_min = 20

    # Metrics: outcomes depend on sleep, activity, and snacks.
    energy = _clamp(quality - 8 + (duration - mean) * 6 + rng.gauss(0, 5), 10, 100)
    focus = _clamp(rng.gauss(74 if is_weekend else 70, 5) + (prev_state["quality"] - 80) * 0.9, 0, 100)
    mood = _clamp(rng.gauss(74, 10) + activity_min * 0.08 - (6 if had_snack else 0), 0, 100)
    steps = max(800, int(rng.gauss(member["steps"], 1500) + activity_min * 90))
    rhr = round(rng.gauss(member["rhr"], 2.5) + (4 if household_rough_night else 0), 1)

    for suffix, metric_type, value, unit, hour in (
        ("energy", "energy_level", energy, "score_0_100", "09"),
        ("focus", "focus_score", focus, "score_0_100", "14"),
        ("mood", "mood", mood, "score_0_100", "20"),
        ("steps", "steps", steps, "count", "21"),
        ("rhr", "resting_heart_rate", rhr, "bpm", "06"),
    ):
        gb.add_metric(metric_id=f"metric:{key}:{iso}:{suffix}", person_id=person_id,
                      metric_type=metric_type, value=round(float(value), 1), unit=unit,
                      timestamp=f"{iso}T{hour}:00:00Z", day_id=day_id)
    return day_id, {"quality": quality, "sugar": sugar, "active": activity_min >= 40}


def build_demo_family_household(
    num_days: int = 90,
    end_date: str = DEMO_END_DATE,
    members: Optional[list] = None,
) -> nx.DiGraph:
    """
    Build the fictional "Demo family" graph: four members over the
    `num_days` days ending on `end_date` (inclusive).
    """
    if num_days <= 0:
        raise ValueError("num_days must be positive")

    members = members or DEMO_MEMBERS
    gb = GraphBuilder()
    gb.add_household(DEMO_HOUSEHOLD_ID, name="Demo family", timezone="America/Los_Angeles")
    for m in members:
        gb.add_person(f"person:{m['key']}", household_id=DEMO_HOUSEHOLD_ID, name=m["name"],
                      role=m["role"], age=m["age"], emoji=m["emoji"])
    _ensure_food_catalog(gb)

    end = date.fromisoformat(end_date)
    start = end - timedelta(days=num_days - 1)
    prev: dict = {m["key"]: None for m in members}
    state: dict = {m["key"]: None for m in members}
    for offset in range(num_days):
        d = start + timedelta(days=offset)
        # About one household-wide rough night every two to three weeks.
        rough = random.Random(f"household:{d.isoformat()}").random() < 0.06
        for m in members:
            prev[m["key"]], state[m["key"]] = _add_member_day(
                gb, m, d, prev[m["key"]], state[m["key"]], rough)
    return gb.G


def build_single_child_household(
    num_days: int = 60,
    end_date: str = DEMO_END_DATE,
) -> nx.DiGraph:
    """Smaller graph with only Leo (kept for quick experiments and tests)."""
    return build_demo_family_household(num_days=num_days, end_date=end_date,
                                       members=[DEMO_MEMBERS[0]])
