"""
Household Health Graph Engine - ML-enhanced graph analysis for household health patterns.

Main modules:
- graph_builder: Core graph construction utilities
- synthetic_data: Realistic household data generation
- graph_embeddings: GraphSAGE-based node embeddings
- anomaly_detection: Health pattern anomaly detection
- pattern_analysis: Health pattern discovery and prediction
- link_prediction: Hidden relationship discovery
- node_classifier: Node type classification
- ethics: Privacy, consent, fairness, transparency, and governance
- schemas: Pydantic validation schemas
- config: Application configuration
- logger: Structured logging
- utils: Utility functions
- llm_engine: OpenAI-compatible LLM integration for health Q&A
- rag_engine: RAG pipeline with ChromaDB vector store
- food_ocr: OCR-based food image analysis with LLM vision fallback
"""

__version__ = "0.3.0"
__author__ = "deokhwajeong"

from .config import settings
from .logger import api_logger, ml_logger, data_logger

__all__ = [
    "settings",
    "api_logger",
    "ml_logger",
    "data_logger",
]
