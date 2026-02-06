"""
Apple Health integration module.

Provides functionality for:
- Importing Apple Health XML export files
- Processing iOS Shortcuts real-time health data
- Integrating health metrics into the graph
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from lxml import etree
import io

from .graph_builder import GraphBuilder


class AppleHealthParser:
    """Parse Apple Health export.xml files."""
    
    # Apple Health type identifiers we care about
    STEP_COUNT = "HKQuantityTypeIdentifierStepCount"
    DISTANCE_WALKING_RUNNING = "HKQuantityTypeIdentifierDistanceWalkingRunning"
    ACTIVE_ENERGY_BURNED = "HKQuantityTypeIdentifierActiveEnergyBurned"
    HEART_RATE = "HKQuantityTypeIdentifierHeartRate"
    DIETARY_ENERGY = "HKQuantityTypeIdentifierDietaryEnergyConsumed"
    SLEEP_ANALYSIS = "HKCategoryTypeIdentifierSleepAnalysis"
    
    @staticmethod
    def parse_xml(xml_content: bytes) -> Dict[str, Any]:
        """
        Parse Apple Health XML export and extract health metrics.
        
        Args:
            xml_content: Raw XML file content
            
        Returns:
            Dictionary with extracted health data categorized by type
        """
        try:
            tree = etree.parse(io.BytesIO(xml_content))
            root = tree.getroot()
            
            data = {
                "steps": [],
                "distance": [],
                "active_calories": [],
                "heart_rate": [],
                "dietary_calories": [],
                "sleep": [],
            }
            
            # Parse Record elements for quantity types
            for record in root.xpath("//Record"):
                record_type = record.get("type")
                
                if record_type == AppleHealthParser.STEP_COUNT:
                    data["steps"].append({
                        "value": float(record.get("value")),
                        "unit": record.get("unit"),
                        "start_date": record.get("startDate"),
                        "end_date": record.get("endDate"),
                    })
                    
                elif record_type == AppleHealthParser.DISTANCE_WALKING_RUNNING:
                    data["distance"].append({
                        "value": float(record.get("value")),
                        "unit": record.get("unit"),
                        "start_date": record.get("startDate"),
                        "end_date": record.get("endDate"),
                    })
                    
                elif record_type == AppleHealthParser.ACTIVE_ENERGY_BURNED:
                    data["active_calories"].append({
                        "value": float(record.get("value")),
                        "unit": record.get("unit"),
                        "start_date": record.get("startDate"),
                        "end_date": record.get("endDate"),
                    })
                    
                elif record_type == AppleHealthParser.HEART_RATE:
                    data["heart_rate"].append({
                        "value": float(record.get("value")),
                        "unit": record.get("unit"),
                        "start_date": record.get("startDate"),
                        "end_date": record.get("endDate"),
                    })
                    
                elif record_type == AppleHealthParser.DIETARY_ENERGY:
                    data["dietary_calories"].append({
                        "value": float(record.get("value")),
                        "unit": record.get("unit"),
                        "start_date": record.get("startDate"),
                        "end_date": record.get("endDate"),
                    })
            
            # Parse sleep analysis (category type)
            for record in root.xpath("//Record[@type='HKCategoryTypeIdentifierSleepAnalysis']"):
                value = record.get("value")
                # Apple Health sleep values: 0=InBed, 1=Asleep, 2=Awake
                sleep_state = "unknown"
                if value == "0":
                    sleep_state = "in_bed"
                elif value == "1":
                    sleep_state = "asleep"
                elif value == "2":
                    sleep_state = "awake"
                    
                data["sleep"].append({
                    "state": sleep_state,
                    "start_date": record.get("startDate"),
                    "end_date": record.get("endDate"),
                })
            
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to parse Apple Health XML: {str(e)}")
    
    @staticmethod
    def aggregate_daily_steps(steps_data: List[Dict]) -> Dict[str, float]:
        """Aggregate step counts by date."""
        daily_steps = {}
        for entry in steps_data:
            date = entry["start_date"][:10]  # Extract YYYY-MM-DD
            if date not in daily_steps:
                daily_steps[date] = 0
            daily_steps[date] += entry["value"]
        return daily_steps
    
    @staticmethod
    def aggregate_daily_calories(calories_data: List[Dict]) -> Dict[str, float]:
        """Aggregate calories by date."""
        daily_calories = {}
        for entry in calories_data:
            date = entry["start_date"][:10]
            if date not in daily_calories:
                daily_calories[date] = 0
            daily_calories[date] += entry["value"]
        return daily_calories
    
    @staticmethod
    def calculate_average_heart_rate(hr_data: List[Dict]) -> Dict[str, float]:
        """Calculate average heart rate by date."""
        daily_hr = {}
        daily_count = {}
        
        for entry in hr_data:
            date = entry["start_date"][:10]
            if date not in daily_hr:
                daily_hr[date] = 0
                daily_count[date] = 0
            daily_hr[date] += entry["value"]
            daily_count[date] += 1
        
        return {date: daily_hr[date] / daily_count[date] for date in daily_hr}
    
    @staticmethod
    def process_sleep_sessions(sleep_data: List[Dict]) -> List[Dict]:
        """Process sleep data into sleep sessions."""
        # Group consecutive asleep periods
        sessions = []
        current_session = None
        
        for entry in sorted(sleep_data, key=lambda x: x["start_date"]):
            if entry["state"] == "asleep":
                if current_session is None:
                    current_session = {
                        "start_time": entry["start_date"],
                        "end_time": entry["end_date"],
                    }
                else:
                    # Extend current session
                    current_session["end_time"] = entry["end_date"]
            else:
                # End of sleep session
                if current_session is not None:
                    sessions.append(current_session)
                    current_session = None
        
        # Add final session if exists
        if current_session is not None:
            sessions.append(current_session)
        
        return sessions


def add_health_data_to_graph(
    gb: GraphBuilder,
    person_id: str,
    health_data: Dict[str, Any]
) -> Dict[str, int]:
    """
    Add Apple Health data to the graph.
    
    Args:
        gb: GraphBuilder instance
        person_id: Person identifier (e.g., "sei" or "deokhwa")
        health_data: Parsed health data dictionary
        
    Returns:
        Statistics about imported records
    """
    stats = {
        "steps_days": 0,
        "calories_days": 0,
        "heart_rate_days": 0,
        "sleep_sessions": 0,
    }
    
    # Add daily steps as activities
    daily_steps = AppleHealthParser.aggregate_daily_steps(health_data["steps"])
    for date, steps in daily_steps.items():
        activity_id = f"activity:{person_id}:{date}:steps"
        day_id = f"day:{person_id}:{date}"
        
        # Ensure day exists
        try:
            gb.add_day(
                day_id=day_id,
                person_id=person_id,
                date=date,
                weekday=datetime.fromisoformat(date).weekday(),
            )
        except:
            pass  # Day might already exist
        
        gb.add_activity(
            activity_id=activity_id,
            person_id=person_id,
            date=date,
            name="Steps",
            day_id=day_id,
            category="walking",
            steps=int(steps),
        )
        stats["steps_days"] += 1
    
    # Add daily active calories
    daily_calories = AppleHealthParser.aggregate_daily_calories(health_data["active_calories"])
    for date, calories in daily_calories.items():
        activity_id = f"activity:{person_id}:{date}:calories"
        day_id = f"day:{person_id}:{date}"
        
        try:
            gb.add_day(
                day_id=day_id,
                person_id=person_id,
                date=date,
                weekday=datetime.fromisoformat(date).weekday(),
            )
        except:
            pass
        
        gb.add_activity(
            activity_id=activity_id,
            person_id=person_id,
            date=date,
            name="Active Calories",
            day_id=day_id,
            category="calories",
            calories_burned=int(calories),
        )
        stats["calories_days"] += 1
    
    # Add daily average heart rate as metrics
    daily_hr = AppleHealthParser.calculate_average_heart_rate(health_data["heart_rate"])
    for date, avg_hr in daily_hr.items():
        metric_id = f"metric:{person_id}:{date}:heart_rate"
        gb.add_metric(
            metric_id=metric_id,
            person_id=person_id,
            metric_type="heart_rate",
            value=avg_hr,
            unit="bpm",
            timestamp=f"{date}T12:00:00Z",
        )
        stats["heart_rate_days"] += 1
    
    # Add sleep sessions
    sleep_sessions = AppleHealthParser.process_sleep_sessions(health_data["sleep"])
    for session in sleep_sessions:
        start_dt = datetime.fromisoformat(session["start_time"].replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(session["end_time"].replace("Z", "+00:00"))
        
        date = start_dt.date().isoformat()
        duration_hours = (end_dt - start_dt).total_seconds() / 3600
        
        sleep_id = f"sleep:{person_id}:{date}"
        day_id = f"day:{person_id}:{date}"
        
        try:
            gb.add_day(
                day_id=day_id,
                person_id=person_id,
                date=date,
                weekday=start_dt.weekday(),
            )
        except:
            pass
        
        gb.add_sleep(
            sleep_id=sleep_id,
            person_id=person_id,
            date=date,
            start_time=start_dt.time().isoformat(),
            end_time=end_dt.time().isoformat(),
            day_id=day_id,
            duration_hours=duration_hours,
            quality_score=75,  # Default quality score
        )
        stats["sleep_sessions"] += 1
    
    return stats
