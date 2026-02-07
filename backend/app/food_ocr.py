"""
Food Photo OCR & Nutrition Analysis Module.

Provides:
  1. Image-based text extraction (OCR) using pytesseract or regex fallback
  2. Food name recognition from extracted text
  3. Nutritional information lookup from a built-in database
  4. Graph integration — adds recognized foods as meal nodes
"""

from __future__ import annotations

import base64
import io
import re
import hashlib
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
#  Built-in nutrition database  (per serving)
# ---------------------------------------------------------------------------
NUTRITION_DB: Dict[str, Dict[str, Any]] = {
    # Fruits
    "banana": {"kcal": 90, "protein_g": 1.0, "fat_g": 0.3, "fiber_g": 3.0, "sugar_g": 12.0, "category": "fruit"},
    "apple": {"kcal": 95, "protein_g": 0.5, "fat_g": 0.3, "fiber_g": 4.4, "sugar_g": 19.0, "category": "fruit"},
    "orange": {"kcal": 62, "protein_g": 1.2, "fat_g": 0.2, "fiber_g": 3.1, "sugar_g": 12.0, "category": "fruit"},
    "strawberry": {"kcal": 49, "protein_g": 1.0, "fat_g": 0.5, "fiber_g": 3.0, "sugar_g": 7.0, "category": "fruit"},
    "blueberries": {"kcal": 57, "protein_g": 0.7, "fat_g": 0.3, "fiber_g": 2.4, "sugar_g": 10.0, "category": "fruit"},
    "grapes": {"kcal": 69, "protein_g": 0.7, "fat_g": 0.2, "fiber_g": 0.9, "sugar_g": 16.0, "category": "fruit"},
    "watermelon": {"kcal": 30, "protein_g": 0.6, "fat_g": 0.2, "fiber_g": 0.4, "sugar_g": 6.0, "category": "fruit"},
    "mango": {"kcal": 99, "protein_g": 1.4, "fat_g": 0.6, "fiber_g": 2.6, "sugar_g": 23.0, "category": "fruit"},
    "pineapple": {"kcal": 82, "protein_g": 0.9, "fat_g": 0.2, "fiber_g": 2.3, "sugar_g": 16.0, "category": "fruit"},
    "avocado": {"kcal": 160, "protein_g": 2.0, "fat_g": 15.0, "fiber_g": 7.0, "sugar_g": 0.7, "category": "fruit"},

    # Vegetables
    "spinach": {"kcal": 23, "protein_g": 2.9, "fat_g": 0.4, "fiber_g": 2.2, "sugar_g": 0.4, "category": "vegetable"},
    "broccoli": {"kcal": 55, "protein_g": 3.7, "fat_g": 0.6, "fiber_g": 5.1, "sugar_g": 2.5, "category": "vegetable"},
    "carrot": {"kcal": 41, "protein_g": 0.9, "fat_g": 0.2, "fiber_g": 2.8, "sugar_g": 4.7, "category": "vegetable"},
    "tomato": {"kcal": 22, "protein_g": 1.1, "fat_g": 0.2, "fiber_g": 1.5, "sugar_g": 3.2, "category": "vegetable"},
    "cucumber": {"kcal": 16, "protein_g": 0.7, "fat_g": 0.1, "fiber_g": 0.5, "sugar_g": 1.7, "category": "vegetable"},
    "sweet potato": {"kcal": 103, "protein_g": 2.3, "fat_g": 0.1, "fiber_g": 3.8, "sugar_g": 5.4, "category": "vegetable"},
    "kale": {"kcal": 35, "protein_g": 2.9, "fat_g": 0.6, "fiber_g": 3.6, "sugar_g": 0.8, "category": "vegetable"},
    "lettuce": {"kcal": 15, "protein_g": 1.4, "fat_g": 0.2, "fiber_g": 1.3, "sugar_g": 0.8, "category": "vegetable"},
    "onion": {"kcal": 40, "protein_g": 1.1, "fat_g": 0.1, "fiber_g": 1.7, "sugar_g": 4.2, "category": "vegetable"},
    "garlic": {"kcal": 149, "protein_g": 6.4, "fat_g": 0.5, "fiber_g": 2.1, "sugar_g": 1.0, "category": "vegetable"},

    # Grains & Carbs
    "rice": {"kcal": 206, "protein_g": 4.3, "fat_g": 0.4, "fiber_g": 0.6, "sugar_g": 0.0, "category": "grain"},
    "brown rice": {"kcal": 216, "protein_g": 5.0, "fat_g": 1.8, "fiber_g": 3.5, "sugar_g": 0.7, "category": "grain"},
    "bread": {"kcal": 265, "protein_g": 9.4, "fat_g": 3.2, "fiber_g": 2.7, "sugar_g": 5.0, "category": "grain"},
    "pasta": {"kcal": 220, "protein_g": 8.1, "fat_g": 1.3, "fiber_g": 2.5, "sugar_g": 0.6, "category": "grain"},
    "oatmeal": {"kcal": 150, "protein_g": 5.0, "fat_g": 2.5, "fiber_g": 4.0, "sugar_g": 1.0, "category": "grain"},
    "quinoa": {"kcal": 222, "protein_g": 8.1, "fat_g": 3.6, "fiber_g": 5.2, "sugar_g": 1.6, "category": "grain"},
    "tortilla": {"kcal": 218, "protein_g": 5.7, "fat_g": 4.6, "fiber_g": 3.1, "sugar_g": 1.0, "category": "grain"},
    "noodles": {"kcal": 138, "protein_g": 4.5, "fat_g": 2.1, "fiber_g": 1.8, "sugar_g": 0.6, "category": "grain"},

    # Protein
    "chicken breast": {"kcal": 165, "protein_g": 31.0, "fat_g": 3.6, "fiber_g": 0.0, "sugar_g": 0.0, "category": "protein"},
    "chicken": {"kcal": 239, "protein_g": 27.3, "fat_g": 13.6, "fiber_g": 0.0, "sugar_g": 0.0, "category": "protein"},
    "salmon": {"kcal": 208, "protein_g": 20.4, "fat_g": 13.4, "fiber_g": 0.0, "sugar_g": 0.0, "category": "protein"},
    "tuna": {"kcal": 132, "protein_g": 28.2, "fat_g": 1.3, "fiber_g": 0.0, "sugar_g": 0.0, "category": "protein"},
    "beef": {"kcal": 250, "protein_g": 26.0, "fat_g": 15.4, "fiber_g": 0.0, "sugar_g": 0.0, "category": "protein"},
    "steak": {"kcal": 271, "protein_g": 26.0, "fat_g": 18.0, "fiber_g": 0.0, "sugar_g": 0.0, "category": "protein"},
    "pork": {"kcal": 242, "protein_g": 27.3, "fat_g": 14.0, "fiber_g": 0.0, "sugar_g": 0.0, "category": "protein"},
    "tofu": {"kcal": 76, "protein_g": 8.0, "fat_g": 4.8, "fiber_g": 0.3, "sugar_g": 0.7, "category": "protein"},
    "egg": {"kcal": 155, "protein_g": 13.0, "fat_g": 11.0, "fiber_g": 0.0, "sugar_g": 1.1, "category": "protein"},
    "eggs": {"kcal": 155, "protein_g": 13.0, "fat_g": 11.0, "fiber_g": 0.0, "sugar_g": 1.1, "category": "protein"},
    "shrimp": {"kcal": 99, "protein_g": 24.0, "fat_g": 0.3, "fiber_g": 0.0, "sugar_g": 0.0, "category": "protein"},
    "sausage": {"kcal": 301, "protein_g": 12.0, "fat_g": 27.0, "fiber_g": 0.0, "sugar_g": 1.4, "category": "protein"},
    "bacon": {"kcal": 541, "protein_g": 37.0, "fat_g": 42.0, "fiber_g": 0.0, "sugar_g": 1.3, "category": "protein"},

    # Dairy
    "milk": {"kcal": 42, "protein_g": 3.4, "fat_g": 1.0, "fiber_g": 0.0, "sugar_g": 5.0, "category": "dairy"},
    "cheese": {"kcal": 402, "protein_g": 25.0, "fat_g": 33.0, "fiber_g": 0.0, "sugar_g": 1.3, "category": "dairy"},
    "yogurt": {"kcal": 59, "protein_g": 10.0, "fat_g": 0.7, "fiber_g": 0.0, "sugar_g": 3.2, "category": "dairy"},
    "greek yogurt": {"kcal": 59, "protein_g": 10.0, "fat_g": 0.7, "fiber_g": 0.0, "sugar_g": 3.2, "category": "dairy"},
    "butter": {"kcal": 717, "protein_g": 0.9, "fat_g": 81.0, "fiber_g": 0.0, "sugar_g": 0.1, "category": "dairy"},
    "ice cream": {"kcal": 207, "protein_g": 3.5, "fat_g": 11.0, "fiber_g": 0.7, "sugar_g": 21.0, "category": "dairy"},

    # Fast Food / Prepared
    "pizza": {"kcal": 266, "protein_g": 11.0, "fat_g": 10.0, "fiber_g": 2.3, "sugar_g": 3.6, "category": "fast_food"},
    "hamburger": {"kcal": 295, "protein_g": 17.0, "fat_g": 14.0, "fiber_g": 1.3, "sugar_g": 5.0, "category": "fast_food"},
    "hot dog": {"kcal": 290, "protein_g": 10.0, "fat_g": 18.0, "fiber_g": 0.8, "sugar_g": 4.0, "category": "fast_food"},
    "french fries": {"kcal": 312, "protein_g": 3.4, "fat_g": 15.0, "fiber_g": 2.7, "sugar_g": 0.3, "category": "fast_food"},
    "fried chicken": {"kcal": 320, "protein_g": 22.0, "fat_g": 18.0, "fiber_g": 0.5, "sugar_g": 0.5, "category": "fast_food"},
    "taco": {"kcal": 226, "protein_g": 9.4, "fat_g": 11.0, "fiber_g": 2.4, "sugar_g": 1.5, "category": "fast_food"},
    "burrito": {"kcal": 206, "protein_g": 8.7, "fat_g": 6.9, "fiber_g": 3.5, "sugar_g": 1.3, "category": "fast_food"},
    "sandwich": {"kcal": 250, "protein_g": 12.0, "fat_g": 10.0, "fiber_g": 2.0, "sugar_g": 4.0, "category": "fast_food"},
    "ramen": {"kcal": 436, "protein_g": 10.0, "fat_g": 17.0, "fiber_g": 2.0, "sugar_g": 1.0, "category": "fast_food"},
    "instant ramen": {"kcal": 436, "protein_g": 10.0, "fat_g": 17.0, "fiber_g": 2.0, "sugar_g": 1.0, "category": "fast_food"},

    # Korean Food
    "kimchi": {"kcal": 15, "protein_g": 1.1, "fat_g": 0.5, "fiber_g": 1.6, "sugar_g": 1.1, "category": "korean"},
    "bibimbap": {"kcal": 490, "protein_g": 22.0, "fat_g": 14.0, "fiber_g": 5.0, "sugar_g": 6.0, "category": "korean"},
    "bulgogi": {"kcal": 256, "protein_g": 23.0, "fat_g": 14.0, "fiber_g": 1.0, "sugar_g": 10.0, "category": "korean"},
    "japchae": {"kcal": 195, "protein_g": 4.0, "fat_g": 8.0, "fiber_g": 2.0, "sugar_g": 9.0, "category": "korean"},
    "tteokbokki": {"kcal": 330, "protein_g": 7.0, "fat_g": 8.0, "fiber_g": 2.0, "sugar_g": 15.0, "category": "korean"},
    "kimbap": {"kcal": 270, "protein_g": 9.0, "fat_g": 6.0, "fiber_g": 2.0, "sugar_g": 3.0, "category": "korean"},
    "miso soup": {"kcal": 40, "protein_g": 3.0, "fat_g": 1.0, "fiber_g": 0.5, "sugar_g": 2.0, "category": "korean"},
    "doenjang jjigae": {"kcal": 120, "protein_g": 8.0, "fat_g": 5.0, "fiber_g": 2.0, "sugar_g": 3.0, "category": "korean"},
    "sundubu jjigae": {"kcal": 130, "protein_g": 10.0, "fat_g": 6.0, "fiber_g": 1.5, "sugar_g": 2.0, "category": "korean"},
    "samgyeopsal": {"kcal": 330, "protein_g": 18.0, "fat_g": 28.0, "fiber_g": 0.0, "sugar_g": 0.0, "category": "korean"},

    # Japanese
    "sushi": {"kcal": 200, "protein_g": 9.0, "fat_g": 3.0, "fiber_g": 1.0, "sugar_g": 4.0, "category": "japanese"},
    "sashimi": {"kcal": 127, "protein_g": 26.0, "fat_g": 2.0, "fiber_g": 0.0, "sugar_g": 0.0, "category": "japanese"},
    "tempura": {"kcal": 280, "protein_g": 8.0, "fat_g": 16.0, "fiber_g": 1.0, "sugar_g": 2.0, "category": "japanese"},
    "udon": {"kcal": 270, "protein_g": 8.0, "fat_g": 2.0, "fiber_g": 2.0, "sugar_g": 3.0, "category": "japanese"},

    # Snacks / Desserts
    "chocolate": {"kcal": 546, "protein_g": 5.0, "fat_g": 31.0, "fiber_g": 7.0, "sugar_g": 48.0, "category": "snack"},
    "cookies": {"kcal": 502, "protein_g": 5.4, "fat_g": 25.0, "fiber_g": 1.1, "sugar_g": 33.0, "category": "snack"},
    "cake": {"kcal": 257, "protein_g": 3.2, "fat_g": 10.0, "fiber_g": 0.3, "sugar_g": 28.0, "category": "snack"},
    "donut": {"kcal": 452, "protein_g": 5.0, "fat_g": 25.0, "fiber_g": 1.7, "sugar_g": 22.0, "category": "snack"},
    "chips": {"kcal": 536, "protein_g": 7.0, "fat_g": 35.0, "fiber_g": 4.8, "sugar_g": 0.3, "category": "snack"},
    "popcorn": {"kcal": 375, "protein_g": 11.0, "fat_g": 4.5, "fiber_g": 14.5, "sugar_g": 0.9, "category": "snack"},
    "granola bar": {"kcal": 193, "protein_g": 4.0, "fat_g": 7.6, "fiber_g": 1.5, "sugar_g": 14.0, "category": "snack"},
    "candy": {"kcal": 394, "protein_g": 0.1, "fat_g": 0.1, "fiber_g": 0.0, "sugar_g": 98.0, "category": "snack"},

    # Drinks
    "coffee": {"kcal": 2, "protein_g": 0.3, "fat_g": 0.0, "fiber_g": 0.0, "sugar_g": 0.0, "category": "beverage"},
    "green tea": {"kcal": 2, "protein_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0, "sugar_g": 0.0, "category": "beverage"},
    "orange juice": {"kcal": 45, "protein_g": 0.7, "fat_g": 0.2, "fiber_g": 0.2, "sugar_g": 8.4, "category": "beverage"},
    "soda": {"kcal": 41, "protein_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0, "sugar_g": 10.6, "category": "beverage"},
    "smoothie": {"kcal": 135, "protein_g": 3.0, "fat_g": 1.5, "fiber_g": 2.5, "sugar_g": 22.0, "category": "beverage"},
    "protein shake": {"kcal": 160, "protein_g": 25.0, "fat_g": 3.0, "fiber_g": 1.0, "sugar_g": 5.0, "category": "beverage"},

    # Nuts & Seeds
    "almonds": {"kcal": 579, "protein_g": 21.0, "fat_g": 50.0, "fiber_g": 12.0, "sugar_g": 4.4, "category": "nuts"},
    "walnuts": {"kcal": 654, "protein_g": 15.0, "fat_g": 65.0, "fiber_g": 6.7, "sugar_g": 2.6, "category": "nuts"},
    "peanut butter": {"kcal": 588, "protein_g": 25.0, "fat_g": 50.0, "fiber_g": 6.0, "sugar_g": 9.0, "category": "nuts"},
    "chia seeds": {"kcal": 486, "protein_g": 17.0, "fat_g": 31.0, "fiber_g": 34.0, "sugar_g": 0.0, "category": "nuts"},

    # Supplements
    "whey protein": {"kcal": 120, "protein_g": 24.0, "fat_g": 1.0, "fiber_g": 0.0, "sugar_g": 2.0, "category": "supplement"},
    "multivitamin": {"kcal": 5, "protein_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0, "sugar_g": 0.0, "category": "supplement"},
    "fish oil": {"kcal": 25, "protein_g": 0.0, "fat_g": 2.8, "fiber_g": 0.0, "sugar_g": 0.0, "category": "supplement"},

    # Other common
    "salad": {"kcal": 65, "protein_g": 2.5, "fat_g": 3.0, "fiber_g": 2.0, "sugar_g": 2.0, "category": "vegetable"},
    "soup": {"kcal": 80, "protein_g": 4.0, "fat_g": 2.5, "fiber_g": 1.5, "sugar_g": 3.0, "category": "other"},
    "stew": {"kcal": 150, "protein_g": 10.0, "fat_g": 6.0, "fiber_g": 3.0, "sugar_g": 4.0, "category": "other"},
    "curry": {"kcal": 220, "protein_g": 10.0, "fat_g": 12.0, "fiber_g": 3.0, "sugar_g": 5.0, "category": "other"},
    "fried rice": {"kcal": 238, "protein_g": 5.0, "fat_g": 8.0, "fiber_g": 1.0, "sugar_g": 1.0, "category": "other"},
    "pancake": {"kcal": 227, "protein_g": 6.4, "fat_g": 8.1, "fiber_g": 1.0, "sugar_g": 7.0, "category": "other"},
    "waffle": {"kcal": 291, "protein_g": 7.9, "fat_g": 14.0, "fiber_g": 1.0, "sugar_g": 6.0, "category": "other"},
    "cereal": {"kcal": 379, "protein_g": 7.0, "fat_g": 1.6, "fiber_g": 5.0, "sugar_g": 18.0, "category": "other"},
    "olive oil": {"kcal": 884, "protein_g": 0.0, "fat_g": 100.0, "fiber_g": 0.0, "sugar_g": 0.0, "category": "other"},
    "honey": {"kcal": 304, "protein_g": 0.3, "fat_g": 0.0, "fiber_g": 0.2, "sugar_g": 82.0, "category": "other"},
}

# Build aliases for flexible matching
_ALIASES: Dict[str, str] = {}
for _name in NUTRITION_DB:
    _ALIASES[_name] = _name
    _ALIASES[_name.replace(" ", "-")] = _name
    _ALIASES[_name.replace(" ", "_")] = _name
    # Plural forms
    if not _name.endswith("s"):
        _ALIASES[_name + "s"] = _name


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def _fuzzy_match(word: str, threshold: float = 0.7) -> Optional[str]:
    """Simple character-overlap similarity match."""
    word_lower = word.lower().strip()
    if word_lower in _ALIASES:
        return _ALIASES[word_lower]

    best_score, best_match = 0.0, None
    for alias, canonical in _ALIASES.items():
        if len(alias) < 3 or len(word_lower) < 3:
            continue
        # Substring match
        if word_lower in alias or alias in word_lower:
            score = min(len(word_lower), len(alias)) / max(len(word_lower), len(alias))
            if score > best_score:
                best_score, best_match = score, canonical
            continue
        # Character overlap
        common = sum(1 for c in word_lower if c in alias)
        score = common / max(len(word_lower), len(alias))
        if score > best_score:
            best_score, best_match = score, canonical

    return best_match if best_score >= threshold else None


# ---------------------------------------------------------------------------
#  OCR / Text Extraction
# ---------------------------------------------------------------------------

def _try_tesseract(image_bytes: bytes) -> Optional[str]:
    """Attempt OCR via pytesseract. Returns None if unavailable."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang="eng")
        return text
    except Exception:
        return None


def _try_easyocr(image_bytes: bytes) -> Optional[str]:
    """Attempt OCR via easyocr. Returns None if unavailable."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(image_bytes)
        return " ".join([r[1] for r in results])
    except Exception:
        return None


def extract_text_from_image(image_bytes: bytes) -> Tuple[str, str]:
    """
    Extract text from an image using available OCR engines.
    Returns (extracted_text, engine_used).
    Falls back to filename-based detection if no OCR engine available.
    """
    # Try tesseract first (most common)
    text = _try_tesseract(image_bytes)
    if text and text.strip():
        return text.strip(), "tesseract"

    # Try easyocr
    text = _try_easyocr(image_bytes)
    if text and text.strip():
        return text.strip(), "easyocr"

    return "", "none"


# ---------------------------------------------------------------------------
#  Image Color Analysis — suggest foods when OCR finds nothing
# ---------------------------------------------------------------------------

# Dominant-colour → food-category mapping
_COLOR_FOOD_MAP: List[Tuple[Tuple[int, int, int], str, List[str]]] = [
    # (reference_rgb, category_label, representative foods)
    ((200, 60,  50),  "protein",    ["steak", "beef", "pork"]),
    ((180, 120, 70),  "protein",    ["chicken", "grilled chicken"]),
    ((240, 200, 100), "grain",      ["rice", "pasta", "bread"]),
    ((60,  150, 60),  "vegetable",  ["salad", "broccoli", "spinach"]),
    ((250, 220, 50),  "fruit",      ["banana", "mango"]),
    ((220, 50,  50),  "fruit",      ["strawberry", "watermelon", "apple"]),
    ((255, 165, 50),  "fruit",      ["orange", "pineapple"]),
    ((100, 60,  40),  "korean",     ["bulgogi", "bibimbap"]),
    ((200, 180, 150), "dairy",      ["milk", "yogurt", "cheese"]),
    ((80,  50,  30),  "beverage",   ["coffee", "black tea"]),
    ((220, 220, 200), "grain",      ["rice", "tofu"]),
    ((180, 80,  80),  "fast_food",  ["pizza", "burger"]),
]


def _rgb_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def analyze_image_colors(image_bytes: bytes) -> Dict[str, Any]:
    """
    Analyse the dominant colours of an image and suggest likely food categories.
    Returns dict with 'suggestions' list and 'dominant_colors'.
    """
    try:
        from PIL import Image
    except ImportError:
        return {"suggestions": [], "dominant_colors": [], "method": "unavailable"}

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize for speed
        img = img.resize((64, 64), Image.LANCZOS)

        pixels = list(img.getdata())
        if not pixels:
            return {"suggestions": [], "dominant_colors": [], "method": "empty"}

        # K-means-like: pick top 5 dominant colours via simple binning
        from collections import Counter
        # Quantize to 32-level per channel
        quantized = [((r // 32) * 32, (g // 32) * 32, (b // 32) * 32) for r, g, b in pixels]
        top_colors = [c for c, _ in Counter(quantized).most_common(5)]

        # Score each reference colour
        scored: Dict[str, float] = {}
        matched_foods: Dict[str, List[str]] = {}
        for ref_rgb, category, foods in _COLOR_FOOD_MAP:
            min_dist = min(_rgb_distance(ref_rgb, tc) for tc in top_colors)
            # Convert distance to similarity score (0-1)
            sim = max(0.0, 1.0 - min_dist / 300.0)
            if sim > 0.4:
                if category not in scored or sim > scored[category]:
                    scored[category] = sim
                    matched_foods[category] = foods

        # Sort by score
        ranked = sorted(scored.items(), key=lambda x: -x[1])[:3]

        suggestions = []
        for cat, score in ranked:
            for food_name in matched_foods[cat]:
                if food_name in NUTRITION_DB:
                    info = NUTRITION_DB[food_name].copy()
                    info["name"] = food_name
                    info["match_type"] = "image_color"
                    info["confidence"] = round(score, 2)
                    suggestions.append(info)

        return {
            "suggestions": suggestions,
            "dominant_colors": [f"rgb{c}" for c in top_colors[:3]],
            "categories": [cat for cat, _ in ranked],
            "method": "color_analysis",
        }
    except Exception as e:
        return {"suggestions": [], "dominant_colors": [], "method": f"error: {e}"}


def recognize_foods_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Scan extracted text for known food names and return nutrition info.
    Uses greedy longest-match strategy: longer n-gram matches consume their
    word positions so shorter n-grams and fuzzy matches won't duplicate.
    """
    if not text:
        return []

    normalized = _normalize(text)
    words = normalized.split()
    found: Dict[str, Dict[str, Any]] = {}
    consumed: set[int] = set()  # word positions already matched

    # Multi-word matching (up to 3 words), longest first
    for n_gram in (3, 2, 1):
        for i in range(len(words) - n_gram + 1):
            # Skip if any position is already consumed by a longer match
            if any(j in consumed for j in range(i, i + n_gram)):
                continue
            phrase = " ".join(words[i : i + n_gram])
            if phrase in _ALIASES:
                canonical = _ALIASES[phrase]
                if canonical not in found:
                    info = NUTRITION_DB[canonical].copy()
                    info["name"] = canonical
                    info["matched_text"] = phrase
                    info["match_type"] = "exact"
                    found[canonical] = info
                    # Mark positions as consumed
                    for j in range(i, i + n_gram):
                        consumed.add(j)

    # Fuzzy matching for remaining unconsumed words only
    for i, word in enumerate(words):
        if i in consumed or len(word) < 4:
            continue
        match = _fuzzy_match(word, threshold=0.75)
        if match and match not in found:
            info = NUTRITION_DB[match].copy()
            info["name"] = match
            info["matched_text"] = word
            info["match_type"] = "fuzzy"
            found[match] = info
            consumed.add(i)

    return list(found.values())


def recognize_foods_from_filename(filename: str) -> List[Dict[str, Any]]:
    """
    Try to recognize food names from the filename itself.
    Only returns exact alias matches (no fuzzy) to avoid false positives
    from filenames like 'IMG_20250207.jpg'.
    """
    if not filename:
        return []

    # Strip extension and path
    name = filename.rsplit(".", 1)[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    name = name.replace("-", " ").replace("_", " ").lower()

    # Only do exact alias matching — no fuzzy for filenames
    normalized = _normalize(name)
    words = normalized.split()
    found: Dict[str, Dict[str, Any]] = {}
    consumed: set[int] = set()

    for n_gram in (3, 2, 1):
        for i in range(len(words) - n_gram + 1):
            if any(j in consumed for j in range(i, i + n_gram)):
                continue
            phrase = " ".join(words[i : i + n_gram])
            if phrase in _ALIASES:
                canonical = _ALIASES[phrase]
                if canonical not in found:
                    info = NUTRITION_DB[canonical].copy()
                    info["name"] = canonical
                    info["matched_text"] = phrase
                    info["match_type"] = "filename"
                    found[canonical] = info
                    for j in range(i, i + n_gram):
                        consumed.add(j)

    return list(found.values())


def analyze_food_image(
    image_bytes: bytes,
    filename: str = "",
    person_key: str = "unknown",
    meal_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: OCR → color analysis → food recognition → nutrition lookup.

    Returns a dict with:
      - recognized_foods: list of food items with nutrition
      - ocr_text: raw extracted text
      - ocr_engine: which OCR engine was used
      - total_nutrition: aggregated nutrition for all identified foods
      - meal_summary: ready-to-insert meal data
      - image_analysis: color-based food suggestions (when OCR yields nothing)
    """
    if meal_date is None:
        meal_date = date.today().isoformat()

    # Step 1: OCR
    ocr_text, ocr_engine = extract_text_from_image(image_bytes)

    # Step 2: Recognize foods from OCR text
    foods = recognize_foods_from_text(ocr_text)

    # Step 3: Also check filename as fallback
    filename_foods = recognize_foods_from_filename(filename)
    existing_names = {f["name"] for f in foods}
    for ff in filename_foods:
        if ff["name"] not in existing_names:
            ff["match_type"] = "filename"
            foods.append(ff)

    # Step 4: Image colour analysis (always run, especially useful when OCR yields nothing)
    color_analysis = analyze_image_colors(image_bytes)

    # Step 5: Aggregate nutrition
    total = {"kcal": 0, "protein_g": 0, "fat_g": 0, "fiber_g": 0, "sugar_g": 0}
    for f in foods:
        for k in total:
            total[k] += f.get(k, 0)

    # Step 6: Build meal summary
    meal_id = f"meal:{person_key}:{meal_date}:photo-{hashlib.md5(image_bytes[:1024]).hexdigest()[:8]}"

    result = {
        "recognized_foods": foods,
        "food_count": len(foods),
        "ocr_text": ocr_text if ocr_text else "",
        "ocr_engine": ocr_engine,
        "total_nutrition": total,
        "image_analysis": color_analysis,
        "meal_summary": {
            "meal_id": meal_id,
            "person_key": person_key,
            "date": meal_date,
            "meal_type": "photo_analyzed",
            "foods": [f["name"] for f in foods],
            "calories": total["kcal"],
            "protein_g": total["protein_g"],
            "fat_g": total["fat_g"],
            "fiber_g": total["fiber_g"],
            "sugar_g": total["sugar_g"],
        },
    }

    return result


def add_meal_to_graph(graph_builder, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add the analyzed meal to the health graph.
    Returns stats about what was added.
    """
    summary = analysis_result["meal_summary"]
    person_key = summary["person_key"]
    meal_date = summary["date"]
    foods = analysis_result["recognized_foods"]

    nodes_added = 0
    edges_added = 0

    gb = graph_builder
    G = gb.G

    person_id = f"person:{person_key}"
    day_id = f"day:{person_key}:{meal_date}"
    meal_id = summary["meal_id"]

    # Ensure person node exists
    if person_id not in G:
        G.add_node(person_id, type="person", name=person_key)
        nodes_added += 1

    # Ensure day node exists
    if day_id not in G:
        G.add_node(day_id, type="day", date=meal_date, person=person_key)
        nodes_added += 1
        if not G.has_edge(person_id, day_id):
            G.add_edge(person_id, day_id, type="has_day")
            edges_added += 1

    # Add meal node
    G.add_node(
        meal_id,
        type="meal",
        meal_type="photo_analyzed",
        date=meal_date,
        calories=summary["calories"],
        protein_g=summary["protein_g"],
        fat_g=summary["fat_g"],
        fiber_g=summary["fiber_g"],
        sugar_g=summary["sugar_g"],
    )
    nodes_added += 1

    if not G.has_edge(day_id, meal_id):
        G.add_edge(day_id, meal_id, type="has_meal")
        edges_added += 1

    # Add food nodes and edges
    for food in foods:
        food_id = f"food:{food['name'].replace(' ', '-')}"
        if food_id not in G:
            G.add_node(
                food_id,
                type="food",
                name=food["name"].title(),
                kcal=food.get("kcal", 0),
                protein_g=food.get("protein_g", 0),
                fat_g=food.get("fat_g", 0),
                fiber_g=food.get("fiber_g", 0),
                sugar_g=food.get("sugar_g", 0),
                category=food.get("category", "unknown"),
            )
            nodes_added += 1

        if not G.has_edge(meal_id, food_id):
            G.add_edge(meal_id, food_id, type="contains")
            edges_added += 1

    return {
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "meal_id": meal_id,
        "total_graph_nodes": len(G.nodes()),
        "total_graph_edges": len(G.edges()),
    }


# ---------------------------------------------------------------------------
#  Manual food text input (no image needed)
# ---------------------------------------------------------------------------

def analyze_food_text(
    text: str,
    person_key: str = "unknown",
    meal_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a text description of food (e.g. "chicken breast with rice and salad").
    Same output format as analyze_food_image but without OCR step.
    """
    if meal_date is None:
        meal_date = date.today().isoformat()

    foods = recognize_foods_from_text(text)

    total = {"kcal": 0, "protein_g": 0, "fat_g": 0, "fiber_g": 0, "sugar_g": 0}
    for f in foods:
        for k in total:
            total[k] += f.get(k, 0)

    meal_id = f"meal:{person_key}:{meal_date}:text-{hashlib.md5(text.encode()).hexdigest()[:8]}"

    return {
        "recognized_foods": foods,
        "food_count": len(foods),
        "ocr_text": text,
        "ocr_engine": "manual_input",
        "total_nutrition": total,
        "meal_summary": {
            "meal_id": meal_id,
            "person_key": person_key,
            "date": meal_date,
            "meal_type": "text_input",
            "foods": [f["name"] for f in foods],
            "calories": total["kcal"],
            "protein_g": total["protein_g"],
            "fat_g": total["fat_g"],
            "fiber_g": total["fiber_g"],
            "sugar_g": total["sugar_g"],
        },
    }
