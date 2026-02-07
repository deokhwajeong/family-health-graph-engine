"""
Health Recommendations Engine.

Provides age- and lifestyle-appropriate ideal targets for nutrition and exercise,
compares current data against ideals, and generates personalized recommendations
for food, exercise, and supplements.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# ───────────────────────────────────────────────────────────────────────────────
#  Family member profiles
# ───────────────────────────────────────────────────────────────────────────────
FAMILY_PROFILES: Dict[str, Dict[str, Any]] = {
    "leo": {
        "name": "Leo",
        "age": 11,
        "gender": "male",
        "role": "child",
        "weight_kg": 38,
        "height_cm": 145,
        "activity_level": "moderate",       # moderate, active kid
        "notes": "Demo persona: omnivore, plays soccer on weekends",
    },
    "mia": {
        "name": "Mia",
        "age": 9,
        "gender": "female",
        "role": "child",
        "weight_kg": 30,
        "height_cm": 134,
        "activity_level": "high",           # swim team
        "notes": "Demo persona: likes fruit and pasta, swim practice 3x a week",
    },
    "alex": {
        "name": "Alex",
        "age": 43,
        "gender": "male",
        "role": "parent",
        "weight_kg": 82,
        "height_cm": 180,
        "activity_level": "high",           # regular strength training
        "notes": "Demo persona: high-protein diet, gym most days",
    },
    "jordan": {
        "name": "Jordan",
        "age": 41,
        "gender": "female",
        "role": "parent",
        "weight_kg": 60,
        "height_cm": 166,
        "activity_level": "moderate",       # walking and yoga
        "notes": "Demo persona: balanced diet, mostly home cooking",
    },
}

# ───────────────────────────────────────────────────────────────────────────────
#  Ideal daily values by age group and activity level
# ───────────────────────────────────────────────────────────────────────────────
# Sources: WHO, USDA DRI, Korean Nutrition Society guidelines

def _bmr(age: int, gender: str, weight_kg: float, height_cm: float) -> float:
    """Mifflin-St Jeor BMR formula."""
    if gender == "male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


_ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "high": 1.725,
    "very_high": 1.9,
}


def get_ideal_targets(person_key: str) -> Dict[str, Any]:
    """
    Return evidence-based ideal daily targets for a family member.
    """
    profile = FAMILY_PROFILES.get(person_key)
    if not profile:
        return {"error": f"Unknown person: {person_key}"}

    age = profile["age"]
    gender = profile["gender"]
    weight = profile["weight_kg"]
    height = profile["height_cm"]
    activity = profile["activity_level"]
    role = profile["role"]

    bmr = _bmr(age, gender, weight, height)
    mult = _ACTIVITY_MULTIPLIERS.get(activity, 1.55)
    tdee = round(bmr * mult)

    # ── Macro targets (grams) ──
    if role == "child":
        protein_g = round(weight * 1.2)          # 1.0-1.5 g/kg for growing children
        fat_pct = 0.30                             # 25-35% for kids
        fiber_g = age + 5                          # Age + 5 rule for children
        sugar_max_g = 25                           # WHO: <25g added sugar for kids
    else:
        protein_g = round(weight * 1.6) if activity in ("high", "very_high") else round(weight * 1.0)
        fat_pct = 0.25
        fiber_g = 30 if gender == "male" else 25   # USDA DRI
        sugar_max_g = 36 if gender == "male" else 25

    fat_g = round((tdee * fat_pct) / 9)
    carb_g = round((tdee - protein_g * 4 - fat_g * 9) / 4)

    # ── Micronutrient targets ──
    if role == "child":
        calcium_mg = 1300
        iron_mg = 10
        vitamin_d_iu = 600
        omega3_mg = 700
        vitamin_c_mg = 45
    else:
        calcium_mg = 1000
        iron_mg = 18 if gender == "female" else 8
        vitamin_d_iu = 800
        omega3_mg = 1100 if gender == "male" else 1100
        vitamin_c_mg = 90 if gender == "male" else 75

    # ── Sleep targets (hours) ──
    if age <= 5:
        sleep_hours = 11
    elif age <= 12:
        sleep_hours = 10
    elif age <= 17:
        sleep_hours = 9
    else:
        sleep_hours = 7.5

    # ── Exercise targets ──
    if role == "child":
        exercise_min = 60                           # WHO: 60 min/day for children
        exercise_types = ["outdoor play", "running", "swimming", "cycling", "team sports"]
        exercise_note = "At least 60 min of moderate-to-vigorous activity daily, including bone-strengthening 3x/week"
    elif activity in ("high", "very_high"):
        exercise_min = 60
        exercise_types = ["weight training", "cardio (running/rowing)", "HIIT", "stretching/yoga"]
        exercise_note = "Maintain current active routine; ensure rest days for recovery"
    else:
        exercise_min = 30
        exercise_types = ["brisk walking", "yoga", "swimming", "cycling", "bodyweight exercises"]
        exercise_note = "At least 150 min of moderate aerobic activity per week (WHO)"

    # ── Water intake (ml) ──
    water_ml = round(weight * 35) if role == "parent" else round(weight * 40)

    return {
        "person_key": person_key,
        "profile": profile,
        "daily_targets": {
            "calories_kcal": tdee,
            "protein_g": protein_g,
            "fat_g": fat_g,
            "carbs_g": carb_g,
            "fiber_g": fiber_g,
            "sugar_max_g": sugar_max_g,
            "water_ml": water_ml,
        },
        "micronutrients": {
            "calcium_mg": calcium_mg,
            "iron_mg": iron_mg,
            "vitamin_d_iu": vitamin_d_iu,
            "omega3_mg": omega3_mg,
            "vitamin_c_mg": vitamin_c_mg,
        },
        "sleep": {
            "recommended_hours": sleep_hours,
            "note": f"{'Children' if role == 'child' else 'Adults'} aged {age} need ~{sleep_hours}h of quality sleep",
        },
        "exercise": {
            "recommended_min_per_day": exercise_min,
            "recommended_types": exercise_types,
            "note": exercise_note,
        },
    }


# ───────────────────────────────────────────────────────────────────────────────
#  Compare current data vs. ideal targets
# ───────────────────────────────────────────────────────────────────────────────

def _pct(current: float, target: float) -> float:
    if target == 0:
        return 0.0
    return round((current / target) * 100, 1)


def _status(current: float, target: float, is_max: bool = False) -> str:
    """Return 'low' / 'optimal' / 'high' status."""
    ratio = current / target if target else 0
    if is_max:
        if ratio <= 1.0:
            return "optimal"
        elif ratio <= 1.3:
            return "slightly_high"
        return "high"
    else:
        if ratio < 0.7:
            return "low"
        elif ratio <= 1.15:
            return "optimal"
        return "high"


def compare_current_vs_ideal(
    person_key: str,
    daily_features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare a person's actual daily data against their ideal targets.
    Returns gap analysis with status indicators.
    """
    targets = get_ideal_targets(person_key)
    if "error" in targets:
        return targets

    dt = targets["daily_targets"]
    sleep_target = targets["sleep"]["recommended_hours"]
    exercise_target = targets["exercise"]["recommended_min_per_day"]

    # Extract current values
    cur_cal = daily_features.get("total_calories", 0)
    cur_prot = daily_features.get("total_protein_g", 0)
    cur_fiber = daily_features.get("total_fiber_g", 0)
    cur_sugar = daily_features.get("total_sugar_g", 0)
    cur_sleep = daily_features.get("sleep_duration_h", 0)
    cur_burned = daily_features.get("total_calories_burned", 0)
    cur_activity = daily_features.get("activity_count", 0)

    comparison = {
        "calories": {
            "current": cur_cal, "target": dt["calories_kcal"],
            "pct": _pct(cur_cal, dt["calories_kcal"]),
            "status": _status(cur_cal, dt["calories_kcal"]),
            "unit": "kcal",
        },
        "protein": {
            "current": cur_prot, "target": dt["protein_g"],
            "pct": _pct(cur_prot, dt["protein_g"]),
            "status": _status(cur_prot, dt["protein_g"]),
            "unit": "g",
        },
        "fiber": {
            "current": cur_fiber, "target": dt["fiber_g"],
            "pct": _pct(cur_fiber, dt["fiber_g"]),
            "status": _status(cur_fiber, dt["fiber_g"]),
            "unit": "g",
        },
        "sugar": {
            "current": cur_sugar, "target": dt["sugar_max_g"],
            "pct": _pct(cur_sugar, dt["sugar_max_g"]),
            "status": _status(cur_sugar, dt["sugar_max_g"], is_max=True),
            "unit": "g",
        },
        "sleep": {
            "current": cur_sleep, "target": sleep_target,
            "pct": _pct(cur_sleep, sleep_target),
            "status": _status(cur_sleep, sleep_target),
            "unit": "hours",
        },
        "calories_burned": {
            "current": cur_burned, "target": exercise_target * 8,
            "pct": _pct(cur_burned, exercise_target * 8),
            "status": _status(cur_burned, exercise_target * 8),
            "unit": "kcal",
        },
        "activity_sessions": {
            "current": cur_activity, "target": 2,
            "pct": _pct(cur_activity, 2),
            "status": _status(cur_activity, 2),
            "unit": "sessions",
        },
    }

    # Overall score (0-100)
    scores = []
    for k, v in comparison.items():
        ratio = v["pct"] / 100
        if k == "sugar":
            s = max(0, 100 - max(0, ratio - 1.0) * 200)
        else:
            s = max(0, 100 - abs(ratio - 1.0) * 150)
        scores.append(min(100, s))
    overall = round(sum(scores) / len(scores), 1)

    return {
        "person_key": person_key,
        "comparison": comparison,
        "overall_score": overall,
        "targets": targets,
    }


# ───────────────────────────────────────────────────────────────────────────────
#  Personalized Recommendations
# ───────────────────────────────────────────────────────────────────────────────

# Food recommendation database
FOOD_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    # ── High protein ──
    "chicken breast": {"kcal": 165, "protein_g": 31, "benefit": "Lean protein, low fat", "category": "protein"},
    "salmon": {"kcal": 280, "protein_g": 25, "benefit": "Omega-3 rich, brain health", "category": "protein"},
    "eggs": {"kcal": 155, "protein_g": 13, "benefit": "Complete protein, vitamin D", "category": "protein"},
    "greek yogurt": {"kcal": 100, "protein_g": 17, "benefit": "Probiotics, calcium, protein", "category": "dairy"},
    "tofu": {"kcal": 144, "protein_g": 15, "benefit": "Plant protein, low calorie", "category": "protein"},
    "tuna": {"kcal": 130, "protein_g": 28, "benefit": "High protein, omega-3", "category": "protein"},
    "cottage cheese": {"kcal": 98, "protein_g": 11, "benefit": "Casein protein, calcium", "category": "dairy"},
    "turkey breast": {"kcal": 135, "protein_g": 30, "benefit": "Very lean, high protein", "category": "protein"},
    # ── High fiber ──
    "broccoli": {"kcal": 55, "protein_g": 4, "benefit": "Fiber, vitamin C & K, anti-inflammatory", "category": "vegetable"},
    "spinach": {"kcal": 23, "protein_g": 3, "benefit": "Iron, folate, antioxidants", "category": "vegetable"},
    "oatmeal": {"kcal": 389, "protein_g": 17, "benefit": "Sustained energy, beta-glucan fiber", "category": "grain"},
    "sweet potato": {"kcal": 103, "protein_g": 2, "benefit": "Vitamin A, complex carbs, fiber", "category": "vegetable"},
    "quinoa": {"kcal": 368, "protein_g": 14, "benefit": "Complete plant protein, fiber", "category": "grain"},
    "brown rice": {"kcal": 216, "protein_g": 5, "benefit": "Whole grain, sustained energy", "category": "grain"},
    # ── Anti-inflammatory / Antioxidant ──
    "blueberries": {"kcal": 84, "protein_g": 1, "benefit": "Antioxidants, brain health", "category": "fruit"},
    "avocado": {"kcal": 160, "protein_g": 2, "benefit": "Healthy fats, potassium, fiber", "category": "fruit"},
    "almonds": {"kcal": 579, "protein_g": 21, "benefit": "Vitamin E, healthy fats, fiber", "category": "nuts"},
    "walnuts": {"kcal": 654, "protein_g": 15, "benefit": "Omega-3, brain health", "category": "nuts"},
    # ── Kid-friendly healthy ──
    "banana": {"kcal": 90, "protein_g": 1, "benefit": "Quick energy, potassium", "category": "fruit"},
    "apple": {"kcal": 95, "protein_g": 0, "benefit": "Fiber, vitamin C", "category": "fruit"},
    "carrot sticks": {"kcal": 41, "protein_g": 1, "benefit": "Beta-carotene, vision health", "category": "vegetable"},
    "cheese": {"kcal": 113, "protein_g": 7, "benefit": "Calcium, protein for growth", "category": "dairy"},
}

EXERCISE_RECOMMENDATIONS: List[Dict[str, Any]] = [
    # ── Cardio ──
    {"name": "Brisk walking", "type": "cardio", "intensity": "light", "cal_per_30min": 150, "benefit": "Heart health, low impact", "for": ["all"]},
    {"name": "Jogging / Running", "type": "cardio", "intensity": "moderate", "cal_per_30min": 300, "benefit": "Cardiovascular endurance", "for": ["parent", "child"]},
    {"name": "Cycling", "type": "cardio", "intensity": "moderate", "cal_per_30min": 250, "benefit": "Joint-friendly cardio, leg strength", "for": ["all"]},
    {"name": "Swimming", "type": "cardio", "intensity": "moderate", "cal_per_30min": 250, "benefit": "Full body, zero impact", "for": ["all"]},
    {"name": "Jump rope", "type": "cardio", "intensity": "high", "cal_per_30min": 350, "benefit": "Coordination, bone density", "for": ["child", "parent"]},
    {"name": "Dancing", "type": "cardio", "intensity": "moderate", "cal_per_30min": 200, "benefit": "Fun exercise, coordination", "for": ["child"]},
    # ── Strength ──
    {"name": "Bodyweight exercises", "type": "strength", "intensity": "moderate", "cal_per_30min": 170, "benefit": "Functional strength, no equipment", "for": ["all"]},
    {"name": "Weight training", "type": "strength", "intensity": "high", "cal_per_30min": 220, "benefit": "Muscle mass, bone density, metabolism", "for": ["parent"]},
    {"name": "Resistance bands", "type": "strength", "intensity": "light", "cal_per_30min": 120, "benefit": "Joint-safe strength building", "for": ["all"]},
    # ── Flexibility ──
    {"name": "Yoga", "type": "flexibility", "intensity": "light", "cal_per_30min": 120, "benefit": "Flexibility, stress relief, balance", "for": ["all"]},
    {"name": "Stretching", "type": "flexibility", "intensity": "light", "cal_per_30min": 80, "benefit": "Injury prevention, recovery", "for": ["all"]},
    {"name": "Pilates", "type": "flexibility", "intensity": "moderate", "cal_per_30min": 160, "benefit": "Core strength, posture", "for": ["parent"]},
    # ── Sports (for kids) ──
    {"name": "Fencing", "type": "sport", "intensity": "high", "cal_per_30min": 230, "benefit": "Agility, strategy, discipline", "for": ["child"]},
    {"name": "Soccer / Football", "type": "sport", "intensity": "high", "cal_per_30min": 280, "benefit": "Teamwork, endurance", "for": ["child"]},
    {"name": "Basketball", "type": "sport", "intensity": "high", "cal_per_30min": 260, "benefit": "Coordination, agility", "for": ["child"]},
    {"name": "Martial arts", "type": "sport", "intensity": "high", "cal_per_30min": 300, "benefit": "Discipline, self-defense, full body", "for": ["child", "parent"]},
    # ── Active lifestyle ──
    {"name": "Hiking", "type": "outdoor", "intensity": "moderate", "cal_per_30min": 200, "benefit": "Nature exposure, stamina", "for": ["all"]},
    {"name": "Family bike ride", "type": "outdoor", "intensity": "moderate", "cal_per_30min": 200, "benefit": "Family bonding, cardiovascular", "for": ["all"]},
]

SUPPLEMENT_RECOMMENDATIONS: List[Dict[str, Any]] = [
    # ── Universal ──
    {"name": "Vitamin D3", "dose": "1000-2000 IU/day", "benefit": "Bone health, immune function, mood", "for": ["all"], "priority": "high",
     "when": "Low sun exposure, winter months, indoor lifestyle"},
    {"name": "Omega-3 (Fish oil)", "dose": "500-1000 mg EPA+DHA/day", "benefit": "Brain health, anti-inflammatory, heart health", "for": ["all"], "priority": "high",
     "when": "Not eating fatty fish 2x/week"},
    {"name": "Multivitamin", "dose": "1 tablet/day", "benefit": "Fill micronutrient gaps", "for": ["all"], "priority": "medium",
     "when": "Varied diet may still miss some nutrients"},
    # ── For athletes / high activity ──
    {"name": "Whey protein", "dose": "25-30g post-workout", "benefit": "Muscle recovery, convenient protein", "for": ["parent"], "priority": "medium",
     "when": "Difficulty meeting protein target from food alone"},
    {"name": "Creatine monohydrate", "dose": "3-5g/day", "benefit": "Strength, power output, brain function", "for": ["parent"], "priority": "medium",
     "when": "Doing regular weight training"},
    {"name": "Magnesium", "dose": "200-400 mg/day", "benefit": "Sleep quality, muscle recovery, stress", "for": ["parent"], "priority": "medium",
     "when": "High activity level, poor sleep, muscle cramps"},
    {"name": "BCAA", "dose": "5-10g around training", "benefit": "Reduce muscle soreness, support recovery", "for": ["parent"], "priority": "low",
     "when": "Training fasted or very intense sessions"},
    # ── For children ──
    {"name": "Children's multivitamin", "dose": "1 gummy/day", "benefit": "Growth support, immune function", "for": ["child"], "priority": "medium",
     "when": "Picky eater or limited diet variety"},
    {"name": "Calcium + Vitamin D (chewable)", "dose": "500 mg calcium + 400 IU D3", "benefit": "Bone growth, dental health", "for": ["child"], "priority": "high",
     "when": "Low dairy intake, rapid growth phase"},
    {"name": "Iron supplement", "dose": "Per pediatrician", "benefit": "Prevent anemia, energy, focus", "for": ["child"], "priority": "low",
     "when": "Low red meat intake, fatigue, confirmed deficiency"},
    # ── For women ──
    {"name": "Iron", "dose": "18 mg/day", "benefit": "Prevent anemia, energy", "for": ["female_parent"], "priority": "medium",
     "when": "Menstruating women with low red meat consumption"},
    {"name": "Calcium", "dose": "1000 mg/day", "benefit": "Bone density, osteoporosis prevention", "for": ["female_parent"], "priority": "high",
     "when": "Low dairy diet"},
    {"name": "Probiotics", "dose": "10+ billion CFU/day", "benefit": "Gut health, immune support, digestion", "for": ["all"], "priority": "medium",
     "when": "Digestive issues, after antibiotics, low fermented food intake"},
]


def generate_recommendations(
    person_key: str,
    daily_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate personalized food, exercise, and supplement recommendations
    based on the person's profile and current data gaps.
    """
    profile = FAMILY_PROFILES.get(person_key)
    if not profile:
        return {"error": f"Unknown person: {person_key}"}

    role = profile["role"]
    gender = profile["gender"]
    age = profile["age"]
    activity = profile["activity_level"]

    targets = get_ideal_targets(person_key)
    dt = targets["daily_targets"]

    # Determine gaps from current data
    gaps = []
    if daily_features:
        cur_prot = daily_features.get("total_protein_g", 0)
        cur_fiber = daily_features.get("total_fiber_g", 0)
        cur_sugar = daily_features.get("total_sugar_g", 0)
        cur_cal = daily_features.get("total_calories", 0)
        cur_sleep = daily_features.get("sleep_duration_h", 0)
        cur_activity = daily_features.get("activity_count", 0)

        if cur_prot < dt["protein_g"] * 0.7:
            gaps.append("low_protein")
        if cur_fiber < dt["fiber_g"] * 0.7:
            gaps.append("low_fiber")
        if cur_sugar > dt["sugar_max_g"] * 1.3:
            gaps.append("high_sugar")
        if cur_cal < dt["calories_kcal"] * 0.7:
            gaps.append("low_calories")
        if cur_cal > dt["calories_kcal"] * 1.3:
            gaps.append("high_calories")
        if cur_sleep < targets["sleep"]["recommended_hours"] * 0.8:
            gaps.append("low_sleep")
        if cur_activity < 1:
            gaps.append("low_exercise")
    else:
        gaps = ["no_data"]

    # ── Food recommendations ──
    food_recs = []
    priority_foods = set()

    if "low_protein" in gaps or "no_data" in gaps:
        for name, info in FOOD_RECOMMENDATIONS.items():
            if info.get("category") in ("protein", "dairy") and info.get("protein_g", 0) >= 10:
                food_recs.append({**info, "name": name, "reason": "Boost protein intake"})
                priority_foods.add(name)
    if "low_fiber" in gaps or "no_data" in gaps:
        for name, info in FOOD_RECOMMENDATIONS.items():
            if name not in priority_foods and info.get("category") in ("vegetable", "grain", "fruit"):
                food_recs.append({**info, "name": name, "reason": "Increase fiber intake"})
                priority_foods.add(name)
    if "high_sugar" in gaps:
        food_recs.append({"name": "Reduce sugary snacks", "reason": "Sugar intake is above target",
                          "benefit": "Replace chocolate/candy with fruit, nuts, or dark chocolate (>70%)",
                          "category": "advice", "kcal": 0, "protein_g": 0})

    # Always add some balanced picks
    for name, info in FOOD_RECOMMENDATIONS.items():
        if name not in priority_foods and len(food_recs) < 12:
            # Kid-friendly filter
            if role == "child" and info.get("category") in ("fruit", "dairy", "vegetable"):
                food_recs.append({**info, "name": name, "reason": "Healthy choice for growing kids"})
            elif role == "parent":
                food_recs.append({**info, "name": name, "reason": "Nutritious option"})

    food_recs = food_recs[:12]

    # ── Exercise recommendations ──
    exercise_recs = []
    for ex in EXERCISE_RECOMMENDATIONS:
        targets_role = ex["for"]
        if "all" in targets_role or role in targets_role:
            rec = {**ex}
            # Personalisation
            if person_key == "mia" and ex["name"] == "Fencing":
                rec["note"] = "Continue fencing — great for agility and discipline!"
            elif person_key == "alex" and ex["type"] == "strength":
                rec["note"] = "Aligns with your current training program"
            elif "low_exercise" in gaps:
                rec["note"] = "Activity is below target — try to add this"
            exercise_recs.append(rec)

    exercise_recs = exercise_recs[:10]

    # ── Supplement recommendations ──
    supp_recs = []
    for s in SUPPLEMENT_RECOMMENDATIONS:
        targets_role = s["for"]
        match = False
        if "all" in targets_role:
            match = True
        elif role in targets_role:
            match = True
        elif gender == "female" and role == "parent" and "female_parent" in targets_role:
            match = True

        if match:
            rec = {**s}
            # Personalise relevance
            if person_key == "alex" and s["name"] in ("Whey protein", "Creatine monohydrate", "BCAA"):
                rec["relevance"] = "Already in your routine — keep it up"
            elif person_key == "jordan" and s["name"] == "Iron":
                rec["relevance"] = "Important for women with low red meat intake"
            elif person_key in ("leo", "mia") and s["name"] == "Calcium + Vitamin D (chewable)":
                rec["relevance"] = "Essential during growth years"
            supp_recs.append(rec)

    supp_recs.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority", "low"), 2))

    # ── Lifestyle tips based on gaps ──
    tips = []
    if "low_sleep" in gaps:
        tips.append({"icon": "🌙", "tip": "Aim for consistent bedtime routine. Screen off 1 hour before bed."})
    if "high_sugar" in gaps:
        tips.append({"icon": "🍬", "tip": "Replace sugary drinks with water or milk. Try fruit instead of candy."})
    if "low_exercise" in gaps:
        tips.append({"icon": "🏃", "tip": f"Try to get at least {targets['exercise']['recommended_min_per_day']} min of activity daily."})
    if "low_protein" in gaps:
        tips.append({"icon": "🥚", "tip": "Add a protein source to every meal (eggs, chicken, tofu, yogurt)."})
    if "low_fiber" in gaps:
        tips.append({"icon": "🥦", "tip": "Eat more vegetables and whole grains for digestive health."})
    if "high_calories" in gaps:
        tips.append({"icon": "⚖️", "tip": "Watch portion sizes. Focus on nutrient-dense foods."})
    if not tips:
        tips.append({"icon": "✨", "tip": "Great job! Keep up the balanced lifestyle."})

    return {
        "person_key": person_key,
        "profile": {
            "name": profile["name"],
            "age": age,
            "gender": gender,
            "role": role,
            "activity_level": activity,
        },
        "gaps": gaps,
        "food_recommendations": food_recs,
        "exercise_recommendations": exercise_recs,
        "supplement_recommendations": supp_recs,
        "lifestyle_tips": tips,
        "ideal_targets": targets,
    }
