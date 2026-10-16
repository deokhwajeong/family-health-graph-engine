"""
LLM Engine for Family Health Graph Engine.

Provides:
  1. Chat-based health Q&A with OpenAI-compatible APIs
  2. Vision-based food image analysis (GPT-4o, etc.)
  3. Health recommendation generation using graph context
  4. Streaming support for real-time responses
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from .logger import ml_logger

# ---------------------------------------------------------------------------
#  LLM Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_VISION_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024

HEALTH_SYSTEM_PROMPT = """You are a knowledgeable family health assistant integrated into the Family Health Graph Engine.
Your role is to:
1. Provide evidence-based health and nutrition advice for families
2. Analyze food images and identify nutritional content
3. Interpret health patterns from graph data
4. Give age-appropriate recommendations for children and adults
5. Flag concerning patterns while being supportive and encouraging

Important guidelines:
- Always clarify that you're not a medical professional and serious concerns should be discussed with a doctor
- Be sensitive to children's health data and privacy
- Provide practical, actionable advice
- Use the provided graph context when available to personalize responses
- Support both English and Korean languages based on user preference
- When analyzing food images, identify specific foods and estimate nutritional content

You have access to a family health knowledge graph that tracks daily meals, sleep, activities, and health metrics.
"""

FOOD_VISION_PROMPT = """Analyze this food image and identify:
1. All visible food items
2. Estimated portion sizes
3. Approximate nutritional content per item (calories, protein, carbs, fat, fiber)
4. Overall meal assessment (healthy/balanced/indulgent)
5. Any dietary concerns or suggestions

Return your analysis in the following JSON format:
{
  "foods": [
    {
      "name": "food name",
      "portion": "estimated portion",
      "calories": 0,
      "protein_g": 0,
      "carbs_g": 0,
      "fat_g": 0,
      "fiber_g": 0,
      "category": "protein/grain/vegetable/fruit/dairy/snack/other"
    }
  ],
  "total_calories": 0,
  "meal_assessment": "healthy/balanced/indulgent",
  "suggestions": ["suggestion1", "suggestion2"],
  "description": "Brief description of the meal"
}
"""


# ---------------------------------------------------------------------------
#  LLM Client
# ---------------------------------------------------------------------------

class LLMEngine:
    """
    OpenAI-compatible LLM engine for health-related queries.
    Supports any OpenAI API-compatible provider (OpenAI, Azure, local models).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        vision_model: str = DEFAULT_VISION_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.model = model
        self.vision_model = vision_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._async_client = None
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        # Conversation histories per session
        self._conversations: Dict[str, List[Dict[str, str]]] = {}

    def _get_client(self):
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                kwargs: Dict[str, Any] = {}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = OpenAI(**kwargs)
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        return self._client

    def _get_async_client(self):
        """Lazy-initialize the async OpenAI client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
                kwargs: Dict[str, Any] = {}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._async_client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._async_client

    @property
    def is_configured(self) -> bool:
        """Check if the LLM engine has an API key configured."""
        return bool(self._api_key)

    # -----------------------------------------------------------------
    #  Chat Completion
    # -----------------------------------------------------------------

    def chat(
        self,
        message: str,
        session_id: str = "default",
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat message and get a response.

        Args:
            message: User message
            session_id: Conversation session identifier
            system_prompt: Override default system prompt
            context: Additional context from RAG or graph data
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Dict with response text, usage stats, and metadata
        """
        client = self._get_client()

        # Build messages
        messages = self._build_messages(
            message, session_id, system_prompt, context
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

            assistant_msg = response.choices[0].message.content or ""

            # Store in conversation history
            self._add_to_history(session_id, "user", message)
            self._add_to_history(session_id, "assistant", assistant_msg)

            ml_logger.info(
                "LLM chat completed",
                model=self.model,
                tokens=response.usage.total_tokens if response.usage else 0,
            )

            return {
                "response": assistant_msg,
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            ml_logger.error("LLM chat failed", exc=e)
            return {
                "response": f"LLM request failed: {str(e)}. Please check your API key configuration.",
                "model": self.model,
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def chat_stream(
        self,
        message: str,
        session_id: str = "default",
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat response token by token.

        Yields:
            Response text chunks
        """
        client = self._get_async_client()
        messages = self._build_messages(message, session_id, system_prompt, context)

        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            full_response = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text

            # Store completed response
            self._add_to_history(session_id, "user", message)
            self._add_to_history(session_id, "assistant", full_response)

        except Exception as e:
            ml_logger.error("LLM stream failed", exc=e)
            yield f"Error: {str(e)}"

    # -----------------------------------------------------------------
    #  Vision / Food Analysis
    # -----------------------------------------------------------------

    def analyze_food_image(
        self,
        image_bytes: bytes,
        additional_context: str = "",
        person_key: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Analyze a food image using vision LLM.

        Args:
            image_bytes: Raw image bytes
            additional_context: Extra context about the person/meal
            person_key: Person identifier for personalization

        Returns:
            Dict with identified foods, nutrition estimates, and suggestions
        """
        client = self._get_client()

        # Encode image
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Detect image type
        if image_bytes[:4] == b'\x89PNG':
            media_type = "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        else:
            media_type = "image/jpeg"  # fallback

        prompt = FOOD_VISION_PROMPT
        if additional_context:
            prompt += f"\n\nAdditional context: {additional_context}"

        messages = [
            {"role": "system", "content": HEALTH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]

        try:
            response = client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                temperature=0.3,  # Lower temp for structured output
                max_tokens=1500,
            )

            raw_text = response.choices[0].message.content or ""

            # Try to parse JSON from response
            analysis = self._parse_food_analysis(raw_text)
            analysis["raw_response"] = raw_text
            analysis["model"] = self.vision_model
            analysis["person_key"] = person_key
            analysis["timestamp"] = datetime.utcnow().isoformat()
            analysis["method"] = "llm_vision"

            ml_logger.info(
                "Food image analysis completed",
                model=self.vision_model,
                foods_found=len(analysis.get("foods", [])),
            )

            return analysis

        except Exception as e:
            ml_logger.error("Food image analysis failed", exc=e)
            return {
                "error": str(e),
                "foods": [],
                "total_calories": 0,
                "meal_assessment": "unknown",
                "suggestions": [],
                "description": "Analysis failed",
                "method": "llm_vision_error",
            }

    def generate_health_recommendation(
        self,
        person_key: str,
        graph_context: str,
        specific_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate personalized health recommendations using LLM with graph context.

        Args:
            person_key: Person identifier
            graph_context: Serialized graph data / health summary
            specific_question: Optional specific question to address

        Returns:
            Dict with recommendation text and metadata
        """
        prompt = f"""Based on the following health data for {person_key}, provide personalized health recommendations.

Health Data Context:
{graph_context}

"""
        if specific_question:
            prompt += f"Specific question: {specific_question}\n"

        prompt += """Please provide:
1. Key observations about current health patterns
2. Top 3 actionable recommendations
3. Any concerns or areas needing attention
4. Encouraging feedback on positive habits

Format your response clearly with sections."""

        return self.chat(
            message=prompt,
            session_id=f"recommendation:{person_key}",
            temperature=0.5,
        )

    # -----------------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------------

    def _build_messages(
        self,
        message: str,
        session_id: str,
        system_prompt: Optional[str],
        context: Optional[str],
    ) -> List[Dict[str, str]]:
        """Build the messages list for a chat completion."""
        messages = [
            {"role": "system", "content": system_prompt or HEALTH_SYSTEM_PROMPT}
        ]

        # Add context if provided (RAG results)
        if context:
            messages.append({
                "role": "system",
                "content": f"Relevant context from the health knowledge base:\n{context}",
            })

        # Add conversation history
        history = self._conversations.get(session_id, [])
        # Keep last 10 turns to avoid token overflow
        for msg in history[-20:]:
            messages.append(msg)

        # Add current message
        messages.append({"role": "user", "content": message})

        return messages

    def _add_to_history(self, session_id: str, role: str, content: str) -> None:
        """Add a message to conversation history."""
        if session_id not in self._conversations:
            self._conversations[session_id] = []
        self._conversations[session_id].append({
            "role": role,
            "content": content,
        })
        # Trim history to prevent memory growth
        if len(self._conversations[session_id]) > 40:
            self._conversations[session_id] = self._conversations[session_id][-20:]

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._conversations.pop(session_id, None)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        return self._conversations.get(session_id, [])

    def _parse_food_analysis(self, raw_text: str) -> Dict[str, Any]:
        """Parse food analysis from LLM response, handling JSON extraction."""
        # Try to find JSON in response
        try:
            # Look for JSON block
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = raw_text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Fallback: try to parse structured text
        return {
            "foods": [],
            "total_calories": 0,
            "meal_assessment": "unknown",
            "suggestions": [],
            "description": raw_text[:500],
        }


# ---------------------------------------------------------------------------
#  Singleton instance
# ---------------------------------------------------------------------------

_llm_engine: Optional[LLMEngine] = None


def get_llm_engine() -> LLMEngine:
    """Get or create the global LLM engine instance."""
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMEngine()
    return _llm_engine


def configure_llm_engine(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    vision_model: str = DEFAULT_VISION_MODEL,
) -> LLMEngine:
    """Configure and return the global LLM engine."""
    global _llm_engine
    _llm_engine = LLMEngine(
        api_key=api_key,
        base_url=base_url,
        model=model,
        vision_model=vision_model,
    )
    return _llm_engine
