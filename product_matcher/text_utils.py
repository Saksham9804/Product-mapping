from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


COLOR_FAMILIES = {
    "black": {"black", "jet black"},
    "white": {"white", "off white", "offwhite", "cream"},
    "blue": {"blue", "navy", "navy blue", "sky blue", "teal", "aqua"},
    "grey": {"grey", "gray", "charcoal"},
    "green": {"green", "olive", "dark green"},
    "red": {"red", "maroon", "burgundy"},
    "pink": {"pink", "peach", "rose"},
    "yellow": {"yellow", "mustard"},
    "brown": {"brown", "coffee", "tan"},
    "beige": {"beige", "khaki"},
    "orange": {"orange"},
    "purple": {"purple", "lavender", "violet"},
}


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def norm_token(value: Any) -> str:
    text = clean_text(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_present(row: pd.Series, names: list[str]) -> Any:
    for name in names:
        if name in row.index:
            value = row.get(name)
            if clean_text(value):
                return value
    return ""


def parse_price(value: Any) -> float:
    text = clean_text(value)
    if not text:
        return math.nan
    match = re.search(r"[\d,]+(?:\.\d+)?", text)
    return float(match.group(0).replace(",", "")) if match else math.nan


def normalize_gender(value: Any, fallback: Any = "") -> str:
    text = f"{norm_token(value)} {norm_token(fallback)}"
    if "women" in text or "female" in text or "womens" in text:
        return "women"
    if "men" in text or "male" in text or "mens" in text:
        return "men"
    return norm_token(value) or "unknown"


def normalize_category(value: Any, fallback: Any = "") -> str:
    text = norm_token(value) or norm_token(fallback)
    replacements = {
        "t shirt": "tshirts",
        "t shirts": "tshirts",
        "tee shirt": "tshirts",
        "tee shirts": "tshirts",
        "polo tshirts": "tshirts",
        "sweatshirt": "sweatshirts",
        "sweat shirts": "sweatshirts",
        "trackpants tracksuits": "trackpants and tracksuits",
        "track pants tracksuits": "trackpants and tracksuits",
        "jackets": "outerwear and jackets",
        "outerwear jackets": "outerwear and jackets",
    }
    for key, mapped in replacements.items():
        if key == text:
            return mapped
    if "tshirt" in text or "t shirt" in text:
        return "tshirts"
    if "sweatshirt" in text:
        return "sweatshirts"
    if "track" in text:
        return "trackpants and tracksuits"
    if "short" in text:
        return "shorts"
    if "jacket" in text:
        return "outerwear and jackets"
    return text or "unknown"


def normalize_pattern(value: Any, fallback: Any = "") -> str:
    text = norm_token(value) or norm_token(fallback)
    if any(t in text for t in ["check", "checked", "checkered"]):
        return "checks"
    if any(t in text for t in ["stripe", "striped", "stripes", "stripped"]):
        return "striped"
    if "print" in text or "graphic" in text:
        return "printed"
    if "solid" in text or "plain" in text:
        return "solid"
    if "self design" in text or "selfdesign" in text:
        return "self design"
    if "color block" in text or "colour block" in text or "colorblock" in text:
        return "colorblock"
    return text or "unknown"


def normalize_fit(value: Any, fallback: Any = "") -> str:
    text = norm_token(value) or norm_token(fallback)
    if "oversized" in text:
        return "oversized"
    if "slim" in text:
        return "slim"
    if "relaxed" in text or "loose" in text:
        return "relaxed"
    if "regular" in text:
        return "regular"
    return text or "unknown"


def normalize_type(value: Any, title: Any = "") -> str:
    text = f"{norm_token(value)} {norm_token(title)}"
    if "polo" in text:
        return "polo"
    if "track suit" in text or "tracksuit" in text:
        return "tracksuits"
    if "track pant" in text or "trackpant" in text:
        return "trackpants"
    if "hood" in text:
        return "hooded"
    return norm_token(value) or "regular"


def normalize_neck(value: Any, title: Any = "") -> str:
    text = f"{norm_token(value)} {norm_token(title)}"
    if "hood" in text:
        return "hooded neck"
    if "round" in text or "crew" in text:
        return "round neck"
    if "polo" in text:
        return "polo neck"
    if "v neck" in text:
        return "v neck"
    if "mandarin" in text:
        return "mandarin collar"
    if "high neck" in text:
        return "high neck"
    return norm_token(value) or "unknown"


def color_family(value: Any) -> str:
    text = norm_token(value)
    if not text:
        return "unknown"
    for family, aliases in COLOR_FAMILIES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return family
    return text.split()[0] if text else "unknown"


def color_score(left: Any, right: Any) -> float:
    left_family = color_family(left)
    right_family = color_family(right)
    if left_family == "unknown" or right_family == "unknown":
        return 0.5
    return 1.0 if left_family == right_family else 0.0


def price_score(left: Any, right: Any) -> float:
    left_price = parse_price(left)
    right_price = parse_price(right)
    if math.isnan(left_price) or math.isnan(right_price) or max(left_price, right_price) <= 0:
        return 0.5
    gap = abs(left_price - right_price) / max(left_price, right_price)
    return max(0.0, 1.0 - min(gap / 0.75, 1.0))


def token_set(*values: Any) -> set[str]:
    text = " ".join(clean_text(value) for value in values)
    stop = {"and", "the", "for", "with", "men", "women", "mens", "womens", "regular", "fit"}
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in stop}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)

