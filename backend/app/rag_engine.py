"""
RAG (Retrieval-Augmented Generation) Engine for Family Health Graph Engine.

Provides:
  1. Health graph → document conversion for indexing
  2. ChromaDB-based vector store for semantic search
  3. Context retrieval and ranking
  4. RAG pipeline combining retrieval + LLM generation
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .logger import ml_logger

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "health_knowledge"
MAX_CONTEXT_CHUNKS = 5
CHUNK_OVERLAP = 50


# ---------------------------------------------------------------------------
#  Graph → Documents Converter
# ---------------------------------------------------------------------------

class GraphDocumentConverter:
    """
    Converts health graph data into text documents suitable for
    embedding and retrieval.
    """

    @staticmethod
    def graph_to_documents(G: nx.DiGraph, person_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert graph nodes and relationships into searchable text documents.

        Args:
            G: NetworkX directed graph
            person_id: Optional filter for specific person

        Returns:
            List of document dicts with 'id', 'text', 'metadata'
        """
        documents = []

        # Normalize person_id
        normalized = None
        if person_id:
            normalized = person_id.split(":")[-1] if ":" in person_id else person_id

        # ── Person summaries ──
        for node_id, data in G.nodes(data=True):
            if data.get("type") != "person":
                continue
            if normalized and not node_id.endswith(f":{normalized}"):
                continue

            name = data.get("name", node_id)
            role = data.get("role", "unknown")
            age = data.get("age", "unknown")
            doc_text = (
                f"{name} is a {age}-year-old {role} in the {data.get('household_id', 'unknown')} household. "
                f"Emoji: {data.get('emoji', '')}."
            )
            documents.append({
                "id": f"doc:person:{node_id}",
                "text": doc_text,
                "metadata": {"type": "person", "person_id": node_id, "name": name},
            })

        # ── Daily summaries ──
        day_nodes = sorted([
            (nid, d) for nid, d in G.nodes(data=True)
            if d.get("type") == "day"
            and (not normalized or nid.startswith(f"day:{normalized}:"))
        ], key=lambda x: x[0])

        for day_id, day_data in day_nodes:
            parts = day_id.split(":")
            person_name = parts[1] if len(parts) >= 3 else "unknown"
            date_str = parts[2] if len(parts) >= 3 else "unknown"
            is_weekend = day_data.get("is_weekend", False)

            # Collect children (meals, sleep, activities, metrics)
            successors = list(G.successors(day_id))
            meals_info = []
            sleep_info = ""
            activities_info = []
            metrics_info = []

            for child_id in successors:
                child = G.nodes[child_id]
                child_type = child.get("type", "")

                if child_type == "meal":
                    meal_type = child.get("meal_type", "meal")
                    foods = []
                    for food_id in G.successors(child_id):
                        food = G.nodes[food_id]
                        if food.get("type") == "food":
                            foods.append(food.get("name", food_id))
                    meals_info.append(f"{meal_type}: {', '.join(foods)}" if foods else meal_type)

                elif child_type == "sleep":
                    duration = child.get("duration_hours", 0)
                    quality = child.get("quality_score", 0)
                    sleep_info = f"Sleep: {duration:.1f}h, quality {quality:.0f}/100"

                elif child_type == "activity":
                    name = child.get("name", "activity")
                    intensity = child.get("intensity", "unknown")
                    duration = child.get("duration_min", 0)
                    cal = child.get("calories_burned", 0)
                    activities_info.append(
                        f"{name} ({intensity}, {duration}min, {cal}cal burned)"
                    )

                elif child_type == "metric":
                    mt = child.get("metric_type", "unknown")
                    val = child.get("value", 0)
                    metrics_info.append(f"{mt}: {val:.0f}")

            day_text = f"Date: {date_str} ({'weekend' if is_weekend else 'weekday'}) for {person_name}.\n"
            if meals_info:
                day_text += f"Meals: {'; '.join(meals_info)}.\n"
            if sleep_info:
                day_text += f"{sleep_info}.\n"
            if activities_info:
                day_text += f"Activities: {'; '.join(activities_info)}.\n"
            if metrics_info:
                day_text += f"Health metrics: {'; '.join(metrics_info)}.\n"

            documents.append({
                "id": f"doc:day:{day_id}",
                "text": day_text.strip(),
                "metadata": {
                    "type": "daily_summary",
                    "person": person_name,
                    "date": date_str,
                    "is_weekend": is_weekend,
                },
            })

        # ── Food catalog ──
        for node_id, data in G.nodes(data=True):
            if data.get("type") != "food":
                continue
            name = data.get("name", node_id)
            cat = data.get("category", "unknown")
            kcal = data.get("kcal", 0)
            protein = data.get("protein_g", 0)
            fiber = data.get("fiber_g", 0)
            sugar = data.get("sugar_g", 0)
            fat = data.get("fat_g", 0)

            food_text = (
                f"{name} ({cat}): {kcal}kcal, protein {protein}g, "
                f"fat {fat}g, fiber {fiber}g, sugar {sugar}g."
            )
            documents.append({
                "id": f"doc:food:{node_id}",
                "text": food_text,
                "metadata": {"type": "food", "name": name, "category": cat},
            })

        # ── General health knowledge ──
        health_knowledge = [
            {
                "id": "doc:knowledge:sleep",
                "text": "Children aged 6-12 need 9-12 hours of sleep per night. Poor sleep quality affects focus, mood, and immune function. Consistent bedtime routines improve sleep quality.",
                "metadata": {"type": "knowledge", "topic": "sleep"},
            },
            {
                "id": "doc:knowledge:nutrition_children",
                "text": "Children need a balanced diet with protein (1.0-1.5g/kg), fiber (age+5g), limited sugar (<25g/day). Key nutrients include calcium for bone growth, iron for energy, and omega-3 for brain development.",
                "metadata": {"type": "knowledge", "topic": "nutrition"},
            },
            {
                "id": "doc:knowledge:exercise",
                "text": "Children should get at least 60 minutes of moderate-to-vigorous physical activity daily. Activities should include aerobic, muscle-strengthening, and bone-strengthening exercises. Sports like fencing build agility and discipline.",
                "metadata": {"type": "knowledge", "topic": "exercise"},
            },
            {
                "id": "doc:knowledge:hydration",
                "text": "Children aged 4-8 need about 1.2L of water daily, 9-13 year olds need 1.6-2.1L. Dehydration impairs cognitive function and physical performance.",
                "metadata": {"type": "knowledge", "topic": "hydration"},
            },
            {
                "id": "doc:knowledge:mental_health",
                "text": "Mental health in children is closely tied to sleep quality, physical activity, and nutrition. Regular exercise releases endorphins, while proper nutrition supports neurotransmitter production.",
                "metadata": {"type": "knowledge", "topic": "mental_health"},
            },
            {
                "id": "doc:knowledge:korean_food",
                "text": "Korean diet staples include rice, kimchi, doenjang jjigae, bibimbap, and bulgogi. Kimchi provides probiotics and fiber. Traditional Korean meals are well-balanced with vegetables, protein, and fermented foods.",
                "metadata": {"type": "knowledge", "topic": "korean_food"},
            },
        ]
        documents.extend(health_knowledge)

        return documents

    @staticmethod
    def generate_person_summary(G: nx.DiGraph, person_id: str) -> str:
        """
        Generate a comprehensive text summary of a person's health data.
        Useful as LLM context for recommendations.
        """
        normalized = person_id.split(":")[-1] if ":" in person_id else person_id

        day_nodes = sorted([
            nid for nid, d in G.nodes(data=True)
            if d.get("type") == "day" and nid.startswith(f"day:{normalized}:")
        ])

        if not day_nodes:
            return f"No data available for {person_id}."

        from .graph_embeddings import extract_day_features

        # Aggregate stats
        total_days = len(day_nodes)
        all_features = []
        for day_id in day_nodes:
            date = day_id.split(":")[-1]
            feat = extract_day_features(G, person_id, date)
            if feat:
                all_features.append(feat)

        if not all_features:
            return f"No detailed data for {person_id} across {total_days} days."

        import numpy as np

        def avg(key):
            vals = [f.get(key, 0) for f in all_features if f.get(key, 0) > 0]
            return round(np.mean(vals), 1) if vals else 0

        summary = f"""Health Summary for {normalized} ({total_days} days analyzed):

Nutrition:
- Average daily calories: {avg('total_calories'):.0f} kcal
- Average protein: {avg('total_protein_g'):.1f}g
- Average fiber: {avg('total_fiber_g'):.1f}g
- Average sugar: {avg('total_sugar_g'):.1f}g
- Average meals per day: {avg('meal_count'):.1f}

Sleep:
- Average sleep duration: {avg('sleep_duration_h'):.1f} hours
- Average sleep quality: {avg('sleep_quality'):.0f}/100

Activity:
- Average activities per day: {avg('activity_count'):.1f}
- Average calories burned: {avg('total_calories_burned'):.0f}

Health Metrics:
- Average energy level: {avg('metric_energy_level'):.0f}/100
- Average focus score: {avg('metric_focus_score'):.0f}/100
- Average mood: {avg('metric_mood'):.0f}/100
"""
        return summary


# ---------------------------------------------------------------------------
#  Vector Store (ChromaDB)
# ---------------------------------------------------------------------------

class HealthVectorStore:
    """
    ChromaDB-based vector store for health knowledge retrieval.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        self._client = None
        self._collection = None
        self._persist_dir = persist_directory
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy-initialize ChromaDB."""
        if self._initialized:
            return

        try:
            import chromadb

            if self._persist_dir:
                self._client = chromadb.PersistentClient(path=self._persist_dir)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
            ml_logger.info("ChromaDB initialized", persist_dir=self._persist_dir)

        except ImportError:
            raise RuntimeError("chromadb not installed. Run: pip install chromadb")
        except Exception as e:
            raise RuntimeError(f"ChromaDB initialization failed: {e}")

    def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Index documents into the vector store.

        Args:
            documents: List of docs with 'id', 'text', 'metadata'

        Returns:
            Stats about indexing
        """
        self._ensure_initialized()

        ids = [d["id"] for d in documents]
        texts = [d["text"] for d in documents]
        metadatas = [d.get("metadata", {}) for d in documents]

        # Upsert in batches
        batch_size = 100
        added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]

            # Ensure metadata values are strings, ints, floats, or bools
            clean_meta = []
            for m in batch_meta:
                clean = {}
                for k, v in m.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean[k] = v
                    else:
                        clean[k] = str(v)
                clean_meta.append(clean)

            self._collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=clean_meta,
            )
            added += len(batch_ids)

        ml_logger.info("Documents indexed", count=added)
        return {"indexed": added, "total_in_store": self._collection.count()}

    def search(
        self,
        query: str,
        n_results: int = MAX_CONTEXT_CHUNKS,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.

        Args:
            query: Search query text
            n_results: Maximum results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of matching documents with scores
        """
        self._ensure_initialized()

        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, self._collection.count() or 1),
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        try:
            results = self._collection.query(**kwargs)
        except Exception as e:
            ml_logger.error("Vector search failed", exc=e)
            return []

        documents = []
        if results and results["documents"]:
            for i, doc_text in enumerate(results["documents"][0]):
                doc = {
                    "text": doc_text,
                    "id": results["ids"][0][i] if results["ids"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                }
                documents.append(doc)

        return documents

    def clear(self) -> None:
        """Clear all documents from the store."""
        self._ensure_initialized()
        try:
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            ml_logger.info("Vector store cleared")
        except Exception as e:
            ml_logger.error("Failed to clear vector store", exc=e)

    @property
    def document_count(self) -> int:
        """Get number of documents in the store."""
        self._ensure_initialized()
        return self._collection.count()


# ---------------------------------------------------------------------------
#  RAG Pipeline
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Full RAG pipeline: retrieval from vector store + LLM generation.
    """

    def __init__(
        self,
        vector_store: Optional[HealthVectorStore] = None,
        llm_engine=None,
    ):
        self.vector_store = vector_store or HealthVectorStore()
        self._llm = llm_engine  # Lazy import to avoid circular deps
        self._indexed = False

    def _get_llm(self):
        """Get LLM engine (lazy import)."""
        if self._llm is None:
            from .llm_engine import get_llm_engine
            self._llm = get_llm_engine()
        return self._llm

    def index_graph(
        self,
        G: nx.DiGraph,
        person_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Index health graph data into the vector store.

        Args:
            G: NetworkX health graph
            person_id: Optional person filter

        Returns:
            Indexing statistics
        """
        converter = GraphDocumentConverter()
        documents = converter.graph_to_documents(G, person_id)
        stats = self.vector_store.index_documents(documents)
        self._indexed = True
        return stats

    def query(
        self,
        question: str,
        person_id: Optional[str] = None,
        n_context: int = MAX_CONTEXT_CHUNKS,
        session_id: str = "rag",
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        RAG query: retrieve relevant context + generate LLM answer.

        Args:
            question: User question
            person_id: Optional person filter for context
            n_context: Number of context chunks to retrieve
            session_id: Conversation session ID
            include_sources: Whether to include source documents

        Returns:
            Dict with answer, sources, and metadata
        """
        # Step 1: Retrieve relevant documents
        filter_meta = None
        if person_id:
            normalized = person_id.split(":")[-1] if ":" in person_id else person_id
            filter_meta = {"person": normalized}

        try:
            search_results = self.vector_store.search(
                query=question,
                n_results=n_context,
                filter_metadata=filter_meta,
            )
        except Exception:
            # Retry without filter if person filter fails
            search_results = self.vector_store.search(
                query=question,
                n_results=n_context,
            )

        # Step 2: Build context string from retrieved docs
        context_parts = []
        sources = []
        for doc in search_results:
            context_parts.append(doc["text"])
            if include_sources:
                sources.append({
                    "id": doc["id"],
                    "text": doc["text"][:200],
                    "relevance": round(1.0 - doc.get("distance", 0), 3),
                    "metadata": doc.get("metadata", {}),
                })

        context = "\n---\n".join(context_parts) if context_parts else ""

        # Step 3: Generate answer with LLM
        llm = self._get_llm()
        if not llm.is_configured:
            # Fallback: return retrieved context without LLM
            return {
                "answer": self._generate_fallback_answer(question, search_results),
                "sources": sources,
                "method": "retrieval_only",
                "llm_configured": False,
                "context_chunks": len(search_results),
                "timestamp": datetime.utcnow().isoformat(),
            }

        result = llm.chat(
            message=question,
            session_id=session_id,
            context=context,
        )

        return {
            "answer": result.get("response", ""),
            "sources": sources,
            "method": "rag",
            "model": result.get("model", ""),
            "usage": result.get("usage", {}),
            "context_chunks": len(search_results),
            "llm_configured": True,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _generate_fallback_answer(
        self,
        question: str,
        search_results: List[Dict[str, Any]],
    ) -> str:
        """
        Generate an answer without LLM, using only retrieved documents.
        """
        if not search_results:
            return (
                "No relevant information found. Please index the graph data first. "
                "(Call POST /rag/index)"
            )

        answer = "🔍 **Search-based answer** (LLM API key not configured — showing search results only)\n\n"
        for i, doc in enumerate(search_results[:3], 1):
            relevance = 1.0 - doc.get("distance", 0)
            answer += f"**[{i}]** (relevance: {relevance:.0%})\n{doc['text']}\n\n"

        answer += "\n💡 Set the OPENAI_API_KEY environment variable for more detailed AI analysis."
        return answer

    def get_health_context(
        self,
        G: nx.DiGraph,
        person_id: str,
    ) -> str:
        """
        Get comprehensive health context for a person from the graph.
        Useful for LLM recommendations.
        """
        return GraphDocumentConverter.generate_person_summary(G, person_id)

    def reindex(self, G: nx.DiGraph, person_id: Optional[str] = None) -> Dict[str, int]:
        """Clear and re-index the vector store."""
        self.vector_store.clear()
        return self.index_graph(G, person_id)


# ---------------------------------------------------------------------------
#  Singleton
# ---------------------------------------------------------------------------

_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get or create the global RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
