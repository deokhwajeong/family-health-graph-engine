"""
Synthetic data generator for the Household Health Graph Engine.

Builds a realistic household graph for one child over multiple days with
variability to simulate realistic health patterns.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional
import random

import networkx as nx

from .graph_builder import (
    GraphBuilder,
    NODE_FOOD,
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



def build_single_child_household(
    num_days: int = 60,
    start_date: str = "2024-11-20",
) -> nx.DiGraph:
    """
    Build a synthetic graph for one household and one child (Yooni)
    over `num_days` starting from `start_date`.
    
    Includes realistic variation in meals, sleep, activities, and metrics
    to simulate actual health patterns.
    """
    if num_days <= 0:
        raise ValueError("num_days must be positive")

    random.seed(42)  # reproducible
    gb = GraphBuilder()

    household_id = "household:jeong-family"
    person_id = "person:yooni"

    gb.add_household(household_id, name="Jeong family", timezone="America/Los_Angeles")
    gb.add_person(person_id, household_id=household_id, name="Yooni", role="child")

    # prepare food catalog
    _ensure_food_catalog(gb)

    start = date.fromisoformat(start_date)
    prev_day_id: Optional[str] = None
    
    # Food combinations for realistic meals
    breakfast_options = [
        ["food:greek-yogurt", "food:banana"],
        ["food:oatmeal", "food:apple"],
        ["food:eggs", "food:whole-wheat-bread", "food:orange"],
        ["food:milk", "food:whole-wheat-bread", "food:banana"],
    ]
    
    lunch_options = [
        ["food:rice", "food:chicken", "food:broccoli"],
        ["food:rice", "food:salmon", "food:spinach"],
        ["food:whole-wheat-bread", "food:chicken", "food:carrot"],
        ["food:rice", "food:eggs", "food:broccoli"],
    ]
    
    dinner_options = [
        ["food:rice", "food:chicken", "food:spinach"],
        ["food:salmon", "food:broccoli"],
        ["food:rice", "food:eggs", "food:carrot"],
        ["food:chicken", "food:whole-wheat-bread", "food:apple"],
    ]

    for offset in range(num_days):
        d = start + timedelta(days=offset)
        day_id = f"day:yooni:{d.isoformat()}"

        is_weekend = d.weekday() >= 5
        gb.add_day(
            day_id=day_id,
            person_id=person_id,
            date=d.isoformat(),
            weekday=d.weekday(),
            is_school_day=not is_weekend,
            is_weekend=is_weekend,
            prev_day_id=prev_day_id,
        )

        # === MEALS ===
        breakfast_id = f"meal:yooni:{d.isoformat()}:breakfast"
        lunch_id = f"meal:yooni:{d.isoformat()}:lunch"
        dinner_id = f"meal:yooni:{d.isoformat()}:dinner"

        gb.add_meal(
            breakfast_id,
            person_id=person_id,
            date=d.isoformat(),
            meal_type="breakfast",
            day_id=day_id,
            start_time="07:30",
        )
        gb.add_meal(
            lunch_id,
            person_id=person_id,
            date=d.isoformat(),
            meal_type="lunch",
            day_id=day_id,
            start_time="12:30",
        )
        gb.add_meal(
            dinner_id,
            person_id=person_id,
            date=d.isoformat(),
            meal_type="dinner",
            day_id=day_id,
            start_time="18:30",
        )

        # link foods to meals (randomized)
        for food_id in random.choice(breakfast_options):
            gb.link_food_to_meal(breakfast_id, food_id)
        for food_id in random.choice(lunch_options):
            gb.link_food_to_meal(lunch_id, food_id)
        for food_id in random.choice(dinner_options):
            gb.link_food_to_meal(dinner_id, food_id)

        # === SLEEP ===
        # Weekend sleep a bit longer, school nights around 8h
        if is_weekend:
            duration = random.gauss(9, 0.5)
            quality = random.gauss(88, 3)
        else:
            duration = random.gauss(8.25, 0.7)
            quality = random.gauss(82, 5)
        
        duration = max(6, min(11, duration))  # clamp [6, 11]
        quality = max(50, min(100, quality))  # clamp [50, 100]

        sleep_id = f"sleep:yooni:{d.isoformat()}"
        gb.add_sleep(
            sleep_id=sleep_id,
            person_id=person_id,
            date=d.isoformat(),
            start_time="22:30",
            end_time="06:45",
            day_id=day_id,
            duration_hours=duration,
            quality_score=quality,
        )

        # === ACTIVITIES ===
        if d.weekday() in (1, 3):  # Tue, Thu
            act_id = f"activity:yooni:{d.isoformat()}:fencing"
            gb.add_activity(
                activity_id=act_id,
                person_id=person_id,
                date=d.isoformat(),
                name="fencing",
                day_id=day_id,
                category="sport",
                duration_min=60,
                intensity="high",
                calories_burned=350,
            )
        elif is_weekend:
            activities = [
                ("family walk", 30, "low", 100),
                ("bike ride", 45, "medium", 250),
                ("park play", 60, "high", 300),
            ]
            name, duration, intensity, cal = random.choice(activities)
            act_id = f"activity:yooni:{d.isoformat()}:{name.replace(' ', '-')}"
            gb.add_activity(
                activity_id=act_id,
                person_id=person_id,
                date=d.isoformat(),
                name=name,
                day_id=day_id,
                category="sport",
                duration_min=duration,
                intensity=intensity,
                calories_burned=cal,
            )
        else:
            # school days: light activity or nothing
            if random.random() > 0.4:
                act_id = f"activity:yooni:{d.isoformat()}:walk"
                gb.add_activity(
                    activity_id=act_id,
                    person_id=person_id,
                    date=d.isoformat(),
                    name="school walk",
                    day_id=day_id,
                    category="sport",
                    duration_min=20,
                    intensity="low",
                    calories_burned=50,
                )

        # === METRICS (daily measurements) ===
        # Energy level (influenced by sleep quality)
        energy = max(20, min(100, quality - 10 + random.gauss(0, 5)))
        
        # Focus score (affected by sleep & weekend vs weekday)
        if is_weekend:
            focus = random.gauss(75, 10)
        else:
            focus = random.gauss(70, 12)
        focus = max(0, min(100, focus))
        
        # Mood
        mood = random.gauss(75, 15)
        mood = max(0, min(100, mood))

        metric_id_energy = f"metric:yooni:{d.isoformat()}:energy"
        gb.add_metric(
            metric_id=metric_id_energy,
            person_id=person_id,
            metric_type="energy_level",
            value=energy,
            unit="score_0_100",
            timestamp=f"{d.isoformat()}T09:00:00Z",
        )

        metric_id_focus = f"metric:yooni:{d.isoformat()}:focus"
        gb.add_metric(
            metric_id=metric_id_focus,
            person_id=person_id,
            metric_type="focus_score",
            value=focus,
            unit="score_0_100",
            timestamp=f"{d.isoformat()}T14:00:00Z",
        )

        metric_id_mood = f"metric:yooni:{d.isoformat()}:mood"
        gb.add_metric(
            metric_id=metric_id_mood,
            person_id=person_id,
            metric_type="mood",
            value=mood,
            unit="score_0_100",
            timestamp=f"{d.isoformat()}T20:00:00Z",
        )

        prev_day_id = day_id

    return gb.G
