"""
Apple Health XML Export Parser → Family Health Graph.

Parses Apple Health's `export.xml` (generated via Health → Share → Export All Health Data)
and converts it into the Family Health Graph schema (graph_builder.py).

Supported Apple Health record types:
  ─ Steps (HKQuantityTypeIdentifierStepCount)
  ─ Heart Rate (HKQuantityTypeIdentifierHeartRate)
  ─ Resting Heart Rate (HKQuantityTypeIdentifierRestingHeartRate)
  ─ Body Mass (HKQuantityTypeIdentifierBodyMass)
  ─ Active Energy Burned (HKQuantityTypeIdentifierActiveEnergyBurned)
  ─ Dietary Energy (HKQuantityTypeIdentifierDietaryEnergyConsumed)
  ─ Dietary Protein (HKQuantityTypeIdentifierDietaryProtein)
  ─ Dietary Fiber (HKQuantityTypeIdentifierDietaryFiber)
  ─ Dietary Sugar (HKQuantityTypeIdentifierDietarySugar)
  ─ Sleep Analysis (HKCategoryTypeIdentifierSleepAnalysis)
  ─ Walking + Running Distance (HKQuantityTypeIdentifierDistanceWalkingRunning)
  ─ Workout records
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, BinaryIO

import networkx as nx

from .graph_builder import GraphBuilder


# ─── Apple Health record type → internal mapping ─────────────
QUANTITY_METRICS: Dict[str, Tuple[str, str]] = {
    # Apple Health type → (our metric_type, unit)
    "HKQuantityTypeIdentifierStepCount":                ("steps",               "steps"),
    "HKQuantityTypeIdentifierHeartRate":                 ("heart_rate",          "bpm"),
    "HKQuantityTypeIdentifierRestingHeartRate":          ("resting_heart_rate",  "bpm"),
    "HKQuantityTypeIdentifierBodyMass":                  ("body_weight_kg",      "kg"),
    "HKQuantityTypeIdentifierBodyMassIndex":             ("bmi",                 ""),
    "HKQuantityTypeIdentifierActiveEnergyBurned":        ("active_energy",       "kcal"),
    "HKQuantityTypeIdentifierBasalEnergyBurned":         ("basal_energy",        "kcal"),
    "HKQuantityTypeIdentifierDietaryEnergyConsumed":     ("dietary_energy",      "kcal"),
    "HKQuantityTypeIdentifierDietaryProtein":            ("dietary_protein_g",   "g"),
    "HKQuantityTypeIdentifierDietaryFiber":              ("dietary_fiber_g",     "g"),
    "HKQuantityTypeIdentifierDietarySugar":              ("dietary_sugar_g",     "g"),
    "HKQuantityTypeIdentifierDietaryFatTotal":           ("dietary_fat_g",       "g"),
    "HKQuantityTypeIdentifierDietaryCarbohydrates":      ("dietary_carbs_g",     "g"),
    "HKQuantityTypeIdentifierDietaryWater":              ("water_ml",            "mL"),
    "HKQuantityTypeIdentifierDistanceWalkingRunning":    ("distance_km",         "km"),
    "HKQuantityTypeIdentifierDistanceCycling":           ("cycling_distance_km", "km"),
    "HKQuantityTypeIdentifierFlightsClimbed":            ("flights_climbed",     "flights"),
    "HKQuantityTypeIdentifierVO2Max":                    ("vo2_max",             "mL/kg·min"),
    "HKQuantityTypeIdentifierOxygenSaturation":          ("spo2",                "%"),
    "HKQuantityTypeIdentifierRespiratoryRate":           ("respiratory_rate",    "breaths/min"),
    "HKQuantityTypeIdentifierBodyTemperature":           ("body_temperature",    "°C"),
    "HKQuantityTypeIdentifierBloodPressureSystolic":     ("bp_systolic",         "mmHg"),
    "HKQuantityTypeIdentifierBloodPressureDiastolic":    ("bp_diastolic",        "mmHg"),
    "HKQuantityTypeIdentifierBloodGlucose":              ("blood_glucose",       "mg/dL"),
    "HKQuantityTypeIdentifierAppleExerciseTime":         ("exercise_minutes",    "min"),
    "HKQuantityTypeIdentifierAppleStandTime":            ("stand_minutes",       "min"),
    "HKQuantityTypeIdentifierEnvironmentalAudioExposure":("noise_db",            "dBASPL"),
    "HKQuantityTypeIdentifierHeadphoneAudioExposure":    ("headphone_audio_db",  "dBASPL"),
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":  ("hrv_sdnn",            "ms"),
    "HKQuantityTypeIdentifierWalkingHeartRateAverage":   ("walking_hr_avg",      "bpm"),
}

# Sleep states we map
SLEEP_CATEGORY_VALUES = {
    # HKCategoryValueSleepAnalysis*
    "HKCategoryValueSleepAnalysisInBed":    "in_bed",
    "HKCategoryValueSleepAnalysisAsleepUnspecified": "asleep",
    "HKCategoryValueSleepAnalysisAsleep":   "asleep",
    "HKCategoryValueSleepAnalysisAsleepCore": "core_sleep",
    "HKCategoryValueSleepAnalysisAsleepDeep": "deep_sleep",
    "HKCategoryValueSleepAnalysisAsleepREM":  "rem_sleep",
    "HKCategoryValueSleepAnalysisAwake":    "awake_during_sleep",
}

# Workout type → readable name
WORKOUT_TYPES: Dict[str, str] = {
    "HKWorkoutActivityTypeRunning":           "Running",
    "HKWorkoutActivityTypeWalking":           "Walking",
    "HKWorkoutActivityTypeCycling":           "Cycling",
    "HKWorkoutActivityTypeSwimming":          "Swimming",
    "HKWorkoutActivityTypeYoga":              "Yoga",
    "HKWorkoutActivityTypeHiking":            "Hiking",
    "HKWorkoutActivityTypeFunctionalStrengthTraining": "Strength Training",
    "HKWorkoutActivityTypeTraditionalStrengthTraining": "Strength Training",
    "HKWorkoutActivityTypeDance":             "Dance",
    "HKWorkoutActivityTypePilates":           "Pilates",
    "HKWorkoutActivityTypeFencing":           "Fencing",
    "HKWorkoutActivityTypeSoccer":            "Soccer",
    "HKWorkoutActivityTypeBasketball":        "Basketball",
    "HKWorkoutActivityTypeTennis":            "Tennis",
    "HKWorkoutActivityTypeBadminton":         "Badminton",
    "HKWorkoutActivityTypeTableTennis":       "Table Tennis",
    "HKWorkoutActivityTypeCoreTraining":      "Core Training",
    "HKWorkoutActivityTypeElliptical":        "Elliptical",
    "HKWorkoutActivityTypeStairClimbing":     "Stair Climbing",
    "HKWorkoutActivityTypeHighIntensityIntervalTraining": "HIIT",
    "HKWorkoutActivityTypeCrossTraining":     "Cross Training",
    "HKWorkoutActivityTypeMindAndBody":       "Mind & Body",
    "HKWorkoutActivityTypeOther":             "Other Workout",
}


# ─── Helper structs ──────────────────────────────────────────
@dataclass
class ParsedRecord:
    record_type: str
    value: float
    unit: str
    start_date: datetime
    end_date: datetime
    source: str = ""


@dataclass
class ParsedSleep:
    state: str          # e.g. "asleep", "in_bed", "deep_sleep"
    start: datetime
    end: datetime
    source: str = ""


@dataclass
class ParsedWorkout:
    workout_type: str   # readable name
    start: datetime
    end: datetime
    duration_min: float
    energy_burned: float = 0.0
    distance_km: float = 0.0
    source: str = ""


@dataclass
class DaySummary:
    """Aggregated data for one day per person."""
    date: date
    # Metrics – daily aggregates or last-of-day values
    metrics: Dict[str, Tuple[float, str]] = field(default_factory=dict)  # type → (value, unit)
    # Sleep sessions
    sleep_sessions: List[ParsedSleep] = field(default_factory=list)
    # Workouts
    workouts: List[ParsedWorkout] = field(default_factory=list)
    # Dietary (accumulate per day)
    dietary: Dict[str, float] = field(default_factory=dict)


# ─── Constants for aggregation ────────────────────────────────
# For these metrics, sum the day's samples (steps, energy burned, etc.)
SUM_METRICS = {
    "steps", "active_energy", "basal_energy", "dietary_energy",
    "dietary_protein_g", "dietary_fiber_g", "dietary_sugar_g",
    "dietary_fat_g", "dietary_carbs_g", "water_ml",
    "distance_km", "cycling_distance_km", "flights_climbed",
    "exercise_minutes", "stand_minutes",
}
# For these metrics, average the day's samples
AVG_METRICS = {
    "heart_rate", "resting_heart_rate", "spo2", "respiratory_rate",
    "body_temperature", "blood_glucose", "noise_db", "headphone_audio_db",
    "hrv_sdnn", "walking_hr_avg", "bp_systolic", "bp_diastolic",
    "bmi",
}
# Last-of-day metrics
LAST_METRICS = {"body_weight_kg", "vo2_max"}


def _parse_apple_datetime(s: str) -> datetime:
    """Parse Apple Health datetime format: '2025-01-15 08:23:45 -0700'."""
    # Try the Apple format first
    for fmt in (
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse Apple Health date: {s!r}")


# ─── XML Parser ──────────────────────────────────────────────

class AppleHealthParser:
    """
    Streaming XML parser for Apple Health export.xml.

    Usage:
        parser = AppleHealthParser()
        graph = parser.parse_file("/path/to/export.xml", person_key="mia")

    Apple Health exports can be large (100MB+), so we use iterparse
    for memory efficiency.
    """

    def __init__(
        self,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ):
        self.date_from = date_from
        self.date_to = date_to

        # Intermediate storage
        self._records_by_day: Dict[date, DaySummary] = defaultdict(
            lambda: DaySummary(date=date.min)
        )
        self._record_counts: Dict[str, int] = defaultdict(int)

    def _in_date_range(self, d: date) -> bool:
        if self.date_from and d < self.date_from:
            return False
        if self.date_to and d > self.date_to:
            return False
        return True

    # ─── Core streaming parser ────────────────────────────────

    def _stream_parse(self, source: BinaryIO | str | Path) -> None:
        """Stream-parse the XML, extracting records into _records_by_day."""
        context = ET.iterparse(source, events=("end",))

        for event, elem in context:
            tag = elem.tag

            if tag == "Record":
                self._handle_record(elem)
            elif tag == "Workout":
                self._handle_workout(elem)

            # Free memory – critical for large exports
            elem.clear()

    def _handle_record(self, elem: ET.Element) -> None:
        rec_type = elem.get("type", "")
        start_str = elem.get("startDate", "")
        value_str = elem.get("value", "")
        unit = elem.get("unit", "")
        source = elem.get("sourceName", "")

        if not start_str:
            return

        try:
            start_dt = _parse_apple_datetime(start_str)
        except ValueError:
            return

        rec_date = start_dt.date()
        if not self._in_date_range(rec_date):
            return

        # ── Sleep analysis ────────────────────────────────
        if rec_type == "HKCategoryTypeIdentifierSleepAnalysis":
            state_key = elem.get("value", "")
            state = SLEEP_CATEGORY_VALUES.get(state_key)
            if not state:
                return
            end_str = elem.get("endDate", start_str)
            try:
                end_dt = _parse_apple_datetime(end_str)
            except ValueError:
                end_dt = start_dt

            day = self._get_day(rec_date)
            day.sleep_sessions.append(ParsedSleep(
                state=state, start=start_dt, end=end_dt, source=source,
            ))
            self._record_counts["sleep_analysis"] += 1
            return

        # ── Quantity records ──────────────────────────────
        mapping = QUANTITY_METRICS.get(rec_type)
        if not mapping:
            self._record_counts[f"_skipped:{rec_type}"] += 1
            return

        metric_type, metric_unit = mapping
        try:
            value = float(value_str)
        except (ValueError, TypeError):
            return

        # Unit conversion
        if unit == "lb" and metric_unit == "kg":
            value *= 0.453592
        elif unit == "mi" and metric_unit == "km":
            value *= 1.60934
        elif unit == "°F" and metric_unit == "°C":
            value = (value - 32) * 5 / 9
        elif unit == "Cal" and metric_unit == "kcal":
            pass  # 1 Cal = 1 kcal

        day = self._get_day(rec_date)
        self._record_counts[metric_type] += 1

        # Aggregate by strategy
        if metric_type in SUM_METRICS:
            day.metrics.setdefault(metric_type, (0.0, metric_unit))
            prev_val, _ = day.metrics[metric_type]
            day.metrics[metric_type] = (prev_val + value, metric_unit)
        elif metric_type in AVG_METRICS:
            # Store list for averaging later
            key = f"_avg_{metric_type}"
            if key not in day.dietary:
                day.dietary[key] = 0.0
                day.dietary[f"_cnt_{metric_type}"] = 0.0
            day.dietary[key] += value
            day.dietary[f"_cnt_{metric_type}"] += 1.0
        else:
            # Last-of-day
            day.metrics[metric_type] = (value, metric_unit)

    def _handle_workout(self, elem: ET.Element) -> None:
        wtype = elem.get("workoutActivityType", "")
        start_str = elem.get("startDate", "")
        end_str = elem.get("endDate", "")
        duration_str = elem.get("duration", "0")
        energy_str = elem.get("totalEnergyBurned", "0")
        distance_str = elem.get("totalDistance", "0")
        source = elem.get("sourceName", "")

        if not start_str:
            return

        try:
            start_dt = _parse_apple_datetime(start_str)
            end_dt = _parse_apple_datetime(end_str) if end_str else start_dt
        except ValueError:
            return

        rec_date = start_dt.date()
        if not self._in_date_range(rec_date):
            return

        readable_name = WORKOUT_TYPES.get(wtype, wtype.replace("HKWorkoutActivityType", ""))

        try:
            duration_min = float(duration_str)
        except (ValueError, TypeError):
            duration_min = (end_dt - start_dt).total_seconds() / 60.0

        try:
            energy = float(energy_str)
        except (ValueError, TypeError):
            energy = 0.0

        try:
            dist = float(distance_str)
        except (ValueError, TypeError):
            dist = 0.0

        day = self._get_day(rec_date)
        day.workouts.append(ParsedWorkout(
            workout_type=readable_name,
            start=start_dt,
            end=end_dt,
            duration_min=duration_min,
            energy_burned=energy,
            distance_km=dist,
            source=source,
        ))
        self._record_counts["workout"] += 1

    def _get_day(self, d: date) -> DaySummary:
        if d not in self._records_by_day:
            self._records_by_day[d] = DaySummary(date=d)
        return self._records_by_day[d]

    # ─── Finalize averages ────────────────────────────────

    def _finalize_averages(self) -> None:
        """Convert accumulated sums to averages for AVG_METRICS."""
        for day in self._records_by_day.values():
            for metric_type in AVG_METRICS:
                key = f"_avg_{metric_type}"
                cnt_key = f"_cnt_{metric_type}"
                if key in day.dietary and day.dietary.get(cnt_key, 0) > 0:
                    avg_val = day.dietary[key] / day.dietary[cnt_key]
                    _, unit = QUANTITY_METRICS.get(
                        next(
                            (k for k, v in QUANTITY_METRICS.items() if v[0] == metric_type),
                            ("", ("", ""))
                        ),
                        (metric_type, "")
                    )
                    day.metrics[metric_type] = (round(avg_val, 2), unit)
            # Clean up temp keys
            day.dietary = {
                k: v for k, v in day.dietary.items()
                if not k.startswith("_avg_") and not k.startswith("_cnt_")
            }

    # ─── Build graph ─────────────────────────────────────

    def parse_and_build(
        self,
        source: BinaryIO | str | Path,
        person_key: str,
        household_id: str = "household:demo-family",
        household_name: str = "Demo family",
        person_name: Optional[str] = None,
        person_role: str = "child",
        gb: Optional[GraphBuilder] = None,
    ) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Parse Apple Health export and build the health graph.

        Args:
            source: Path to export.xml, or a file-like bytes object
            person_key: Short key like 'mia', 'alex', etc.
            household_id: Household node ID
            household_name: Household display name
            person_name: Person display name (defaults to person_key.capitalize())
            person_role: 'parent' | 'child'
            gb: Optional existing GraphBuilder to add to (for multi-person)

        Returns:
            (graph, stats) — NetworkX DiGraph and import statistics
        """
        # Reset state
        self._records_by_day.clear()
        self._record_counts.clear()

        # Stream-parse XML
        self._stream_parse(source)
        self._finalize_averages()

        # Build graph
        if gb is None:
            gb = GraphBuilder()
            gb.add_household(household_id, name=household_name, timezone="America/Los_Angeles")

        person_id = f"person:{person_key}"
        if person_id not in gb.G.nodes:
            gb.add_person(
                person_id,
                household_id=household_id,
                name=person_name or person_key.capitalize(),
                role=person_role,
            )

        # Sort days
        sorted_days = sorted(self._records_by_day.keys())
        prev_day_id = None
        nodes_created = 0
        edges_created = 0

        for d in sorted_days:
            day_summary = self._records_by_day[d]
            iso = d.isoformat()
            day_id = f"day:{person_key}:{iso}"

            gb.add_day(
                day_id=day_id,
                person_id=person_id,
                date=iso,
                weekday=d.weekday(),
                is_weekend=d.weekday() >= 5,
                is_school_day=d.weekday() < 5,
                prev_day_id=prev_day_id,
            )
            nodes_created += 1
            edges_created += 2 if prev_day_id else 1

            # ── Metrics ─────────────────────────────────
            for metric_type, (value, unit) in day_summary.metrics.items():
                mid = f"metric:{person_key}:{iso}:{metric_type.replace('_', '-')}"
                gb.add_metric(
                    metric_id=mid,
                    person_id=person_id,
                    metric_type=metric_type,
                    value=round(value, 2),
                    unit=unit,
                    timestamp=f"{iso}T12:00:00Z",
                    day_id=day_id,
                    source="apple_health",
                )
                nodes_created += 1
                edges_created += 2  # person→metric + day→metric

            # ── Sleep ───────────────────────────────────
            sleep_data = self._aggregate_sleep(day_summary.sleep_sessions, d)
            if sleep_data:
                sleep_id = f"sleep:{person_key}:{iso}"
                gb.add_sleep(
                    sleep_id=sleep_id,
                    person_id=person_id,
                    date=iso,
                    start_time=sleep_data["bed_time"],
                    end_time=sleep_data["wake_time"],
                    day_id=day_id,
                    duration_hours=sleep_data["duration_hours"],
                    quality_score=sleep_data["quality_score"],
                    deep_sleep_pct=sleep_data.get("deep_sleep_pct", 0),
                    rem_sleep_pct=sleep_data.get("rem_sleep_pct", 0),
                    source="apple_health",
                )
                nodes_created += 1
                edges_created += 1

            # ── Workouts → Activities ───────────────────
            for i, wo in enumerate(day_summary.workouts):
                cat_key = wo.workout_type.lower().replace(" ", "-")
                act_id = f"activity:{person_key}:{iso}:{cat_key}"
                if i > 0:
                    act_id += f"-{i}"

                intensity = self._infer_intensity(wo)
                gb.add_activity(
                    activity_id=act_id,
                    person_id=person_id,
                    date=iso,
                    name=wo.workout_type,
                    day_id=day_id,
                    category="sport",
                    duration_min=round(wo.duration_min, 1),
                    intensity=intensity,
                    calories_burned=round(wo.energy_burned, 1),
                    distance_km=round(wo.distance_km, 2) if wo.distance_km else None,
                    source="apple_health",
                )
                nodes_created += 1
                edges_created += 1

            prev_day_id = day_id

        stats = {
            "days_imported": len(sorted_days),
            "date_range": {
                "from": sorted_days[0].isoformat() if sorted_days else None,
                "to": sorted_days[-1].isoformat() if sorted_days else None,
            },
            "nodes_created": nodes_created,
            "edges_created": edges_created,
            "record_counts": dict(self._record_counts),
            "total_graph_nodes": len(gb.G.nodes),
            "total_graph_edges": len(gb.G.edges),
        }

        return gb.G, stats

    # ─── Sleep aggregation ────────────────────────────────

    @staticmethod
    def _aggregate_sleep(
        sessions: List[ParsedSleep], target_date: date
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregate Apple Health sleep sessions for one night.

        Apple Health records granular sleep stages (InBed, Asleep, Core, Deep, REM, Awake).
        We aggregate into a single sleep node with duration + quality score.
        """
        if not sessions:
            return None

        # Find the sleep window: earliest start, latest end
        asleep_sessions = [
            s for s in sessions
            if s.state not in ("in_bed", "awake_during_sleep")
        ]

        if not asleep_sessions:
            # Only in_bed records → estimate
            asleep_sessions = [s for s in sessions if s.state == "in_bed"]
            if not asleep_sessions:
                return None

        earliest = min(s.start for s in sessions)
        latest = max(s.end for s in sessions)

        # Total time asleep (excluding awake periods)
        total_asleep_min = sum(
            (s.end - s.start).total_seconds() / 60.0
            for s in sessions
            if s.state not in ("in_bed", "awake_during_sleep")
        )
        total_in_bed_min = (latest - earliest).total_seconds() / 60.0

        # If only in_bed, assume ~90% was asleep
        if total_asleep_min == 0 and total_in_bed_min > 0:
            total_asleep_min = total_in_bed_min * 0.9

        duration_hours = total_asleep_min / 60.0

        # Deep sleep & REM percentages
        deep_min = sum(
            (s.end - s.start).total_seconds() / 60.0
            for s in sessions if s.state == "deep_sleep"
        )
        rem_min = sum(
            (s.end - s.start).total_seconds() / 60.0
            for s in sessions if s.state == "rem_sleep"
        )

        deep_pct = (deep_min / total_asleep_min * 100) if total_asleep_min > 0 else 0
        rem_pct = (rem_min / total_asleep_min * 100) if total_asleep_min > 0 else 0

        # Awake time during sleep
        awake_min = sum(
            (s.end - s.start).total_seconds() / 60.0
            for s in sessions if s.state == "awake_during_sleep"
        )

        # Quality score (0-100) heuristic:
        #   Base: 70
        #   + up to 15 for ideal duration (7-9 hours)
        #   + up to 10 for deep sleep % (ideal ~20%)
        #   + up to 5 for REM % (ideal ~25%)
        #   - penalty for awake time
        quality = 70.0

        # Duration score
        if 7 <= duration_hours <= 9:
            quality += 15
        elif 6 <= duration_hours <= 10:
            quality += 10
        elif duration_hours > 10:
            quality += 5
        else:
            quality += max(0, duration_hours / 6 * 10)

        # Deep sleep bonus
        quality += min(10, deep_pct / 20 * 10)

        # REM bonus
        quality += min(5, rem_pct / 25 * 5)

        # Awake penalty
        if total_in_bed_min > 0:
            awake_ratio = awake_min / total_in_bed_min
            quality -= min(15, awake_ratio * 50)

        quality = max(0, min(100, quality))

        return {
            "bed_time": earliest.strftime("%H:%M"),
            "wake_time": latest.strftime("%H:%M"),
            "duration_hours": round(duration_hours, 2),
            "quality_score": round(quality, 1),
            "deep_sleep_pct": round(deep_pct, 1),
            "rem_sleep_pct": round(rem_pct, 1),
        }

    @staticmethod
    def _infer_intensity(wo: ParsedWorkout) -> str:
        """Infer workout intensity from type, duration, and energy burned."""
        cal_per_min = wo.energy_burned / max(1, wo.duration_min)

        if cal_per_min > 10 or wo.workout_type in ("HIIT", "Running"):
            return "high"
        elif cal_per_min > 5 or wo.workout_type in (
            "Cycling", "Swimming", "Fencing", "Soccer", "Basketball",
        ):
            return "medium"
        else:
            return "low"


# ─── Convenience functions ────────────────────────────────

def parse_apple_health_export(
    source: BinaryIO | str | Path,
    person_key: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    household_id: str = "household:demo-family",
    person_name: Optional[str] = None,
    person_role: str = "child",
    gb: Optional[GraphBuilder] = None,
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    One-line convenience to parse Apple Health export and build graph.

    Args:
        source: Path or file-like object to export.xml
        person_key: 'mia', 'alex', etc.
        date_from: Optional ISO date string to start from
        date_to: Optional ISO date string to end at
        household_id: Graph household node ID
        person_name: Display name
        person_role: 'parent' | 'child'
        gb: Optional existing GraphBuilder to append to

    Returns:
        (graph, stats)
    """
    parser = AppleHealthParser(
        date_from=date.fromisoformat(date_from) if date_from else None,
        date_to=date.fromisoformat(date_to) if date_to else None,
    )
    return parser.parse_and_build(
        source=source,
        person_key=person_key,
        household_id=household_id,
        person_name=person_name,
        person_role=person_role,
        gb=gb,
    )
