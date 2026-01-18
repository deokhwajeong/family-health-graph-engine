"""
Pydantic schemas for API request/response validation.
Provides type safety and automatic documentation for all endpoints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import date


class PredictionRequest(BaseModel):
    """Request schema for health metric prediction."""
    person_id: str = Field(..., description="Unique person identifier")
    date: date = Field(..., description="Target date for prediction")
    include_history: bool = Field(default=True, description="Include historical metrics")
    
    @field_validator('person_id')
    @classmethod
    def validate_person_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("person_id cannot be empty")
        return v.strip()


class AnomalyDetectionRequest(BaseModel):
    """Request schema for anomaly detection."""
    person_id: str = Field(..., description="Unique person identifier")
    date: date = Field(..., description="Date to check for anomalies")
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0, description="Anomaly sensitivity (0-1)")
    
    @field_validator('person_id')
    @classmethod
    def validate_person_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("person_id cannot be empty")
        return v.strip()


class LinkPredictionRequest(BaseModel):
    """Request schema for food-health link prediction."""
    source_node_id: str = Field(..., description="Source node (e.g., food)")
    target_node_ids: List[str] = Field(..., description="Target nodes to evaluate")
    top_k: int = Field(default=5, ge=1, le=100, description="Return top K predictions")
    
    @field_validator('source_node_id')
    @classmethod
    def validate_source(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("source_node_id cannot be empty")
        return v.strip()


class ClassificationRequest(BaseModel):
    """Request schema for node classification."""
    node_id: str = Field(..., description="Node to classify")
    node_type: str = Field(..., description="Node type (food/activity/day)")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    
    @field_validator('node_id')
    @classmethod
    def validate_node_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("node_id cannot be empty")
        return v.strip()


class EmbeddingResponse(BaseModel):
    """Response schema for embeddings."""
    node_id: str
    embedding: List[float]
    dimension: int
    confidence: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    person_id: str
    date: str
    predictions: Dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: Optional[str] = None


class AnomalyResponse(BaseModel):
    """Response schema for anomaly detection results."""
    person_id: str
    date: str
    is_anomaly: bool
    anomaly_score: float = Field(ge=0.0, le=1.0)
    reasons: List[str]
    severity: str = Field(description="low | medium | high")


class ClassificationResponse(BaseModel):
    """Response schema for classification results."""
    node_id: str
    node_type: str
    classification: str
    confidence: float = Field(ge=0.0, le=1.0)
    details: Dict[str, Any]


class GraphStatsResponse(BaseModel):
    """Response schema for graph statistics."""
    total_nodes: int
    total_edges: int
    node_types: Dict[str, int]
    density: float
    avg_degree: float
    graph_diameter: Optional[int] = None


class HealthSummaryResponse(BaseModel):
    """Response schema for health summary."""
    person_id: str
    date_range: str
    average_energy: float = Field(ge=0.0, le=100.0)
    average_focus: float = Field(ge=0.0, le=100.0)
    average_mood: float = Field(ge=0.0, le=100.0)
    anomaly_count: int
    key_patterns: List[str]
    recommendations: List[str]
