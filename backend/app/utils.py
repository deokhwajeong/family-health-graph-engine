"""
Utility functions for the Household Health Graph API.

Includes validation, normalization, and helper functions.
"""

from datetime import datetime, date
from typing import Optional, List
import re


def validate_person_id(person_id: str) -> str:
    """
    Validate and normalize person identifier.
    
    Accepts formats:
    - "yooni" -> "person:yooni"
    - "person:yooni" -> "person:yooni"
    
    Args:
        person_id: Person identifier string
        
    Returns:
        Normalized person identifier
        
    Raises:
        ValueError: If person_id is invalid
    """
    if not person_id or not isinstance(person_id, str):
        raise ValueError("person_id must be a non-empty string")
    
    person_id = person_id.strip()
    
    if not person_id:
        raise ValueError("person_id cannot be empty or whitespace only")
    
    # Normalize format
    if not person_id.startswith("person:"):
        person_id = f"person:{person_id}"
    
    return person_id


def validate_date(date_str: str) -> str:
    """
    Validate date string format (YYYY-MM-DD).
    
    Args:
        date_str: Date string to validate
        
    Returns:
        Normalized date string
        
    Raises:
        ValueError: If date format is invalid
    """
    if not date_str or not isinstance(date_str, str):
        raise ValueError("date must be a non-empty string")
    
    try:
        # Validate format and parse
        parsed = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


def validate_days_range(days: int, min_days: int = 1, max_days: int = 365) -> int:
    """
    Validate days parameter is within acceptable range.
    
    Args:
        days: Number of days
        min_days: Minimum allowed days
        max_days: Maximum allowed days
        
    Returns:
        Validated days value
        
    Raises:
        ValueError: If days is out of range
    """
    if not isinstance(days, int):
        raise ValueError("days must be an integer")
    
    if days < min_days or days > max_days:
        raise ValueError(f"days must be between {min_days} and {max_days}, got {days}")
    
    return days


def validate_confidence(confidence: float) -> float:
    """
    Validate confidence score is between 0 and 1.
    
    Args:
        confidence: Confidence score
        
    Returns:
        Validated confidence value
        
    Raises:
        ValueError: If confidence is out of range
    """
    if not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    
    if confidence < 0 or confidence > 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    
    return float(confidence)


def format_person_id(person_id: str) -> str:
    """Format person ID for display (remove 'person:' prefix if present)."""
    if person_id.startswith("person:"):
        return person_id[7:]  # Remove "person:" prefix
    return person_id


def get_date_range_description(start_date: str, end_date: str) -> str:
    """Create human-readable date range description."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start.year == end.year:
            if start.month == end.month:
                return f"{start.strftime('%B %d-%d, %Y')}"
            return f"{start.strftime('%B %d')} - {end.strftime('%B %d, %Y')}"
        return f"{start.strftime('%B %d, %Y')} - {end.strftime('%B %d, %Y')}"
    except:
        return f"{start_date} to {end_date}"


def extract_features_from_dict(data: dict, keys: List[str], default: float = 0.0) -> dict:
    """
    Safely extract features from dictionary with defaults.
    
    Args:
        data: Dictionary to extract from
        keys: List of keys to extract
        default: Default value for missing keys
        
    Returns:
        Dictionary with extracted features
    """
    return {key: float(data.get(key, default)) for key in keys}


def normalize_confidence_scores(scores: dict) -> dict:
    """
    Normalize confidence scores to sum to 1.0 (softmax-like normalization).
    
    Args:
        scores: Dictionary of score values
        
    Returns:
        Dictionary with normalized scores
    """
    total = sum(abs(v) for v in scores.values())
    if total == 0:
        return {k: 1.0 / len(scores) for k in scores}
    return {k: abs(v) / total for k, v in scores.items()}


def merge_predictions(predictions_list: List[dict], weights: Optional[List[float]] = None) -> dict:
    """
    Merge multiple prediction dictionaries using weighted averaging.
    
    Args:
        predictions_list: List of prediction dictionaries
        weights: Optional list of weights (defaults to equal weights)
        
    Returns:
        Merged prediction dictionary
    """
    if not predictions_list:
        return {}
    
    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)
    
    merged = {}
    for pred, weight in zip(predictions_list, weights):
        for key, value in pred.items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value * weight
    
    return merged
