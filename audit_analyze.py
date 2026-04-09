from __future__ import annotations

import re
from collections import Counter
from typing import Any
from uuid import uuid4


PROMO_TERMS = ("best selling", "best seller", "free shipping", "discount", "sale", "limited time")
OFFER_SHIPPING_AVAILABILITY_TERMS = (
    "free shipping",
    "ships free",
    "in stock",
    "limited stock",
    "buy now",
    "save",
    "deal",
    "offer",
    "clearance",
)
RETAILER_TERMS = ("walmart", "amazon", "target", "instacart", "costco", "kroger")
REDIRECT_TERMS = ("http://", "https://", "www.", ".com", "shop at", "visit", "available at")
USE_CASE_TERMS = (
    "for",
    "ideal",
    "perfect",
    "use",
    "usage",
    "designed",
    "great for",
)
OUTCOME_TERMS = (
    "helps",
    "support",
    "improve",
    "benefit",
    "reduce",
    "increase",
    "save",
    "protect",
    "comfort",
)
FORBIDDEN_SPECIAL_CHARS = ("$", "#", "*", "!", "|")
TITLE_SUSPICIOUS_CHARS = ("|", "{", "}", "[", "]", "<", ">", "^", "~", "`")
UNIVERSAL_USE_CASE_TERMS = (
    "use",
    "great for",
    "ideal for",
    "perfect for",
    "designed for",
    "helps",
    "supports",
    "enjoy",
    "serve",
    "apply",
    "clean",
    "protect",
    "organize",
    "store",
    "wear",
    "build",
    "play",
    "pair with",
    "top with",
    "spread on",
)

WALMART_TOP_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "Animals": ("animals", "pet", "pets", "pet supplies"),
    "Arts & Crafts": ("arts & crafts", "arts and crafts", "craft", "crafts"),
    "Baby": ("baby", "infant", "toddler"),
    "Beauty": ("beauty", "cosmetic", "cosmetics", "skin care", "hair care"),
    "Business & Industrial": ("business & industrial", "business and industrial", "industrial"),
    "Electronics & Photography": ("electronics & photography", "electronics and photography", "electronics", "photography"),
    "Everything Else": ("everything else",),
    "Fashion": ("fashion", "apparel", "clothing", "accessories"),
    "Food & Beverage": ("food & beverage", "food and beverage", "food", "beverage", "grocery"),
    "Furniture": ("furniture",),
    "Garden & Patio": ("garden & patio", "garden and patio", "garden", "patio", "outdoor living"),
    "Health & Personal Care": ("health & personal care", "health and personal care", "health", "personal care"),
    "Home": ("home", "kitchen", "housewares"),
    "Home Improvement": ("home improvement", "tools", "hardware"),
    "Household, Industrial Cleaning & Storage": (
        "household, industrial cleaning & storage",
        "household industrial cleaning storage",
        "household cleaning",
        "cleaning",
        "storage",
    ),
    "Media": ("media", "books", "movies", "music"),
    "Musical Instruments": ("musical instruments", "instrument", "instruments"),
    "Office & Stationery": ("office & stationery", "office and stationery", "office", "stationery"),
    "Safety & Emergency": ("safety & emergency", "safety and emergency", "safety", "emergency"),
    "Seasonal & Occasion": ("seasonal & occasion", "seasonal and occasion", "seasonal", "occasion"),
    "Sports, Recreation & Outdoor": (
        "sports, recreation & outdoor",
        "sports recreation outdoor",
        "sports",
        "recreation",
        "outdoor",
    ),
    "Toys": ("toys", "toy"),
    "Vehicle": ("vehicle", "automotive", "auto", "car", "truck"),
}

PRIORITY_CATEGORY_USE_CASE_TERMS: dict[str, tuple[str, ...]] = {
    "Food & Beverage": (
        "serve", "spread on", "pair with", "top with", "enjoy", "snack", "breakfast",
        "dessert", "meal", "recipe", "glaze", "toast", "waffles",
    ),
    "Electronics & Photography": (
        "charge", "connect", "stream", "listen", "work", "gaming", "travel", "protect", "power", "compatible with",
    ),
    "Beauty": (
        "apply", "routine", "daily use", "for skin", "for hair", "massage", "cleanse", "moisturize", "hydrate", "refresh",
    ),
    "Furniture": (
        "sit", "store", "organize", "display", "workspace", "bedroom", "living room", "patio", "comfort", "space-saving",
    ),
    "Home": (
        "organize", "store", "cook", "prep", "serve", "display", "kitchen", "pantry", "countertop", "space-saving",
    ),
    "Toys": (
        "build", "create", "play", "gift", "screen-free", "activity", "learn", "collect", "imagination", "pretend play",
    ),
    "Vehicle": (
        "install", "protect", "repair", "replace", "maintain", "drive", "towing", "storage", "travel", "performance",
    ),
}

LIGHT_CATEGORY_USE_CASE_HINTS: dict[str, tuple[str, ...]] = {
    "Animals": ("feed", "walk", "train"),
    "Arts & Crafts": ("create", "paint", "craft"),
    "Baby": ("daily use", "feeding", "comfort"),
    "Business & Industrial": ("worksite", "operate", "maintain"),
    "Everything Else": ("everyday use", "practical", "convenient"),
    "Fashion": ("wear", "style", "daily wear"),
    "Garden & Patio": ("outdoor use", "plant", "maintain"),
    "Health & Personal Care": ("daily use", "apply", "wellness"),
    "Home Improvement": ("install", "repair", "upgrade"),
    "Household, Industrial Cleaning & Storage": ("clean", "organize", "store"),
    "Media": ("read", "watch", "listen"),
    "Musical Instruments": ("practice", "play", "perform"),
    "Office & Stationery": ("write", "organize", "workspace"),
    "Safety & Emergency": ("protect", "prepare", "emergency use"),
    "Seasonal & Occasion": ("celebrate", "decorate", "gift"),
    "Sports, Recreation & Outdoor": ("train", "outdoor use", "play"),
}
INTERNAL_CATEGORY_BUCKETS = (
    "Food & Beverage",
    "Home",
    "Beauty / Personal Care",
    "Baby / Kids",
    "Pets",
    "Cleaning / Household",
    "Health / Wellness",
    "Electronics / Accessories",
    "General Merchandise / Fallback",
)
WALMART_TOP_TO_INTERNAL_BUCKET: dict[str, str] = {
    "Food & Beverage": "Food & Beverage",
    "Home": "Home",
    "Furniture": "Home",
    "Garden & Patio": "Home",
    "Home Improvement": "Home",
    "Beauty": "Beauty / Personal Care",
    "Baby": "Baby / Kids",
    "Toys": "Baby / Kids",
    "Animals": "Pets",
    "Household, Industrial Cleaning & Storage": "Cleaning / Household",
    "Health & Personal Care": "Health / Wellness",
    "Electronics & Photography": "Electronics / Accessories",
}
INTERNAL_CATEGORY_BUCKET_ALIASES: dict[str, tuple[str, ...]] = {
    "Food & Beverage": ("food", "beverage", "grocery", "snack", "drink", "coffee", "tea"),
    "Home": ("home", "furniture", "decor", "kitchen", "bedroom", "bathroom", "housewares"),
    "Beauty / Personal Care": ("beauty", "cosmetic", "skincare", "skin care", "hair care", "personal care"),
    "Baby / Kids": ("baby", "infant", "toddler", "kids", "children", "child"),
    "Pets": ("pet", "pets", "dog", "cat", "animal"),
    "Cleaning / Household": ("cleaning", "cleaner", "detergent", "household", "surface", "disinfect"),
    "Health / Wellness": ("health", "wellness", "supplement", "vitamin", "nutrition", "personal care"),
    "Electronics / Accessories": ("electronics", "accessories", "charger", "cable", "usb", "battery", "device"),
}
CATEGORY_USE_CASE_HINTS: dict[str, tuple[str, ...]] = {
    "Food & Beverage": ("serve", "serving", "snack", "meal", "breakfast", "recipe", "pair with"),
    "Home": ("living room", "bedroom", "kitchen", "bathroom", "pantry", "closet", "workspace"),
    "Beauty / Personal Care": ("apply", "application", "daily routine", "for skin", "for hair", "for face", "for body"),
    "Baby / Kids": ("for babies", "for toddlers", "for kids", "for children", "for families", "for parents"),
    "Pets": ("for dogs", "for cats", "for pets", "pet care", "feeding", "litter"),
    "Cleaning / Household": ("use on", "surface", "countertop", "bathroom", "kitchen", "stain", "mess", "odor"),
    "Health / Wellness": ("daily routine", "daily use", "for comfort", "for recovery", "for wellness"),
    "Electronics / Accessories": ("compatible with", "works with", "for charging", "for setup", "for travel"),
}
CATEGORY_EXPECTATION_PROFILES: dict[str, dict[str, int]] = {
    "Food & Beverage": {
        "use_case_min_wc": 28,
        "use_case_min_hits": 1,
        "feature_focus_min_wc": 45,
        "feature_high": 3,
        "feature_mid": 3,
        "feature_low": 2,
        "feature_measurement_min": 1,
        "feature_mid_benefit_max": 0,
        "kf_manufacturing_hit_min": 2,
    },
    "Home": {
        "use_case_min_wc": 28,
        "use_case_min_hits": 1,
        "feature_focus_min_wc": 45,
        "feature_high": 3,
        "feature_mid": 3,
        "feature_low": 2,
        "feature_measurement_min": 1,
        "feature_mid_benefit_max": 0,
        "kf_manufacturing_hit_min": 2,
    },
    "Beauty / Personal Care": {
        "use_case_min_wc": 28,
        "use_case_min_hits": 1,
        "feature_focus_min_wc": 45,
        "feature_high": 3,
        "feature_mid": 3,
        "feature_low": 2,
        "feature_measurement_min": 1,
        "feature_mid_benefit_max": 0,
        "kf_manufacturing_hit_min": 2,
    },
    "Baby / Kids": {
        "use_case_min_wc": 28,
        "use_case_min_hits": 1,
        "feature_focus_min_wc": 45,
        "feature_high": 3,
        "feature_mid": 3,
        "feature_low": 2,
        "feature_measurement_min": 1,
        "feature_mid_benefit_max": 0,
        "kf_manufacturing_hit_min": 2,
    },
    "Pets": {
        "use_case_min_wc": 28,
        "use_case_min_hits": 1,
        "feature_focus_min_wc": 45,
        "feature_high": 3,
        "feature_mid": 3,
        "feature_low": 2,
        "feature_measurement_min": 1,
        "feature_mid_benefit_max": 0,
        "kf_manufacturing_hit_min": 2,
    },
    "Cleaning / Household": {
        "use_case_min_wc": 28,
        "use_case_min_hits": 1,
        "feature_focus_min_wc": 45,
        "feature_high": 3,
        "feature_mid": 3,
        "feature_low": 2,
        "feature_measurement_min": 1,
        "feature_mid_benefit_max": 0,
        "kf_manufacturing_hit_min": 2,
    },
    "Health / Wellness": {
        "use_case_min_wc": 30,
        "use_case_min_hits": 1,
        "feature_focus_min_wc": 48,
        "feature_high": 4,
        "feature_mid": 3,
        "feature_low": 2,
        "feature_measurement_min": 1,
        "feature_mid_benefit_max": 0,
        "kf_manufacturing_hit_min": 2,
    },
    "Electronics / Accessories": {
        "use_case_min_wc": 35,
        "use_case_min_hits": 1,
        "feature_focus_min_wc": 55,
        "feature_high": 5,
        "feature_mid": 4,
        "feature_low": 3,
        "feature_measurement_min": 2,
        "feature_mid_benefit_max": 0,
        "kf_manufacturing_hit_min": 3,
    },
    "General Merchandise / Fallback": {
        "use_case_min_wc": 30,
        "use_case_min_hits": 1,
        "feature_focus_min_wc": 45,
        "feature_high": 4,
        "feature_mid": 3,
        "feature_low": 2,
        "feature_measurement_min": 1,
        "feature_mid_benefit_max": 0,
        "kf_manufacturing_hit_min": 2,
    },
}
FEATURE_HEAVY_TERMS = (
    "ingredients",
    "formula",
    "pack",
    "jar",
    "ounce",
    "oz",
    "spec",
    "specification",
    "manufactured",
    "process",
    "compliant",
    "certified",
    "dimensions",
    "size",
)
USE_CASE_SIGNAL_PHRASES = (
    "ideal for",
    "great for",
    "perfect for",
    "designed for",
    "suitable for",
    "intended for",
    "use in",
    "use on",
    "use at",
    "use during",
    "for everyday use",
    "for home",
    "for office",
    "for travel",
    "for kids",
    "for families",
    "for pets",
    "works well in",
    "helps with",
)
USE_CASE_SIGNAL_REGEXES = (
    r"\b(for|at|in|during)\s+(home|office|travel|school|work|outdoor|indoors)\b",
    r"\bfor\s+(kids|children|families|pets|beginners|professionals)\b",
    r"\b(use|using)\s+(in|on|at|during)\b",
)
BENEFIT_SIGNAL_PHRASES = (
    "helps",
    "helps keep",
    "makes it easier",
    "provides",
    "offers",
    "improves",
    "keeps",
    "reduces",
    "allows you to",
    "designed to help",
    "delivers",
    "supports",
    "easy to use",
    "saves time",
    "more comfortable",
    "better organized",
)
FEATURE_SPEC_SIGNAL_PHRASES = FEATURE_HEAVY_TERMS + (
    "made with",
    "constructed with",
    "dimensions",
    "weight",
    "capacity",
    "compatible with",
    "technical",
    "model",
)
MANUFACTURING_DETAIL_TERMS = (
    "manufactured",
    "manufacturing",
    "facility",
    "batch",
    "compliance",
    "specification",
    "pack of",
    "ounces",
    "oz",
    "jar",
    "bottle",
    "count",
)
HTML_CONTAINER_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "ul",
    "ol",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "table",
    "thead",
    "tbody",
    "tr",
    "td",
    "th",
    "br",
}
REPETITION_STOPWORDS = {
    "with",
    "from",
    "that",
    "this",
    "your",
    "have",
    "will",
    "into",
    "over",
    "under",
    "more",
    "most",
    "very",
    "for",
    "and",
    "the",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _lower(text: str) -> str:
    return _norm(text).lower()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _lower(text))


def _strip_html_tags(text: str) -> str:
    if not text:
        return ""
    no_script = re.sub(r"(?is)<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>", " ", text)
    return re.sub(r"(?is)<[^>]+>", " ", no_script)


def _visible_text(text: str) -> str:
    return _norm(_strip_html_tags(text))


def _has_valid_html_container(text: str) -> bool:
    if not text:
        return False
    tag_matches = re.findall(r"(?is)<\s*/?\s*([a-z0-9]+)\b[^>]*>", text)
    return any(tag.lower() in HTML_CONTAINER_TAGS for tag in tag_matches)


def _product_name_anchor(title: str) -> str:
    title_words = [t for t in _tokens(title) if len(t) > 2]
    if len(title_words) >= 2:
        return " ".join(title_words[:2])
    return title_words[0] if title_words else ""


def _is_product_name_repeated_in_text(*, title: str, text: str) -> bool:
    anchor = _product_name_anchor(title)
    if not anchor:
        return False

    text_norm = _lower(text)
    if anchor in text_norm:
        return True

    anchor_tokens = [t for t in _tokens(anchor) if len(t) > 2]
    text_tokens = set(_tokens(text_norm))
    return len(anchor_tokens) >= 2 and sum(1 for t in anchor_tokens if t in text_tokens) >= 2


def _has_substantial_internal_repetition(text: str) -> tuple[bool, dict[str, Any]]:
    tokens = [t for t in _tokens(text) if len(t) >= 4 and t not in REPETITION_STOPWORDS]
    if len(tokens) < 20:
        return False, {"repeated_ngram_count": 0, "repeated_sentence_count": 0}

    repeated_ngrams = 0
    if len(tokens) >= 12:
        ngrams = [" ".join(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
        ngram_counts = Counter(ngrams)
        repeated_ngrams = sum(1 for _ng, count in ngram_counts.items() if count >= 2)

    sentence_parts = re.split(r"[.!?]\s+|\n+", _norm(text))
    normalized_sentences = [
        " ".join([t for t in _tokens(part) if len(t) >= 3 and t not in REPETITION_STOPWORDS])
        for part in sentence_parts
    ]
    sentence_counts = Counter([s for s in normalized_sentences if len(s.split()) >= 6])
    repeated_sentences = sum(1 for _s, count in sentence_counts.items() if count >= 2)

    has_repetition = repeated_ngrams >= 2 or repeated_sentences >= 1
    return has_repetition, {
        "repeated_ngram_count": repeated_ngrams,
        "repeated_sentence_count": repeated_sentences,
    }


def _content_tokens_for_similarity(text: str) -> set[str]:
    return {t for t in _tokens(text) if len(t) >= 4 and t not in REPETITION_STOPWORDS}


def _has_substantial_bullet_repetition(bullets: list[str]) -> tuple[bool, dict[str, Any]]:
    if len(bullets) < 3:
        return False, {"duplicate_pairs": 0, "high_similarity_pairs": 0}

    normalized = [_lower(re.sub(r"[^a-z0-9\s]", " ", b)) for b in bullets]
    exact_duplicates = len(set(normalized)) < len(normalized)

    high_similarity_pairs = 0
    duplicate_pairs = 0
    for i in range(len(normalized)):
        tok_i = _content_tokens_for_similarity(normalized[i])
        for j in range(i + 1, len(normalized)):
            tok_j = _content_tokens_for_similarity(normalized[j])
            if not tok_i or not tok_j:
                continue
            inter = len(tok_i & tok_j)
            union = len(tok_i | tok_j)
            if union == 0:
                continue
            jaccard = inter / union
            if jaccard >= 0.75:
                high_similarity_pairs += 1
            if jaccard >= 0.85:
                duplicate_pairs += 1

    has_repetition = exact_duplicates or duplicate_pairs >= 1 or high_similarity_pairs >= 2
    return has_repetition, {
        "exact_duplicate_detected": exact_duplicates,
        "duplicate_pairs": duplicate_pairs,
        "high_similarity_pairs": high_similarity_pairs,
    }


def _count_bullets_overlapping_description(bullets: list[str], visible_description: str) -> int:
    desc_text = _lower(visible_description)
    desc_tokens = _content_tokens_for_similarity(desc_text)
    overlap_count = 0
    for bullet in bullets:
        b_text = _lower(bullet)
        if not b_text:
            continue
        if len(b_text) >= 20 and b_text in desc_text:
            overlap_count += 1
            continue

        b_tokens = _content_tokens_for_similarity(b_text)
        if len(b_tokens) < 3:
            continue
        coverage = len(b_tokens & desc_tokens) / max(1, len(b_tokens))
        if coverage >= 0.8:
            overlap_count += 1
    return overlap_count


def _count_phrase_hits(text: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for phrase in phrases if phrase and phrase in text)


def _profile_description_language(text: str) -> dict[str, Any]:
    normalized = _lower(text)
    tokens = _tokens(normalized)
    word_count = len(tokens)

    use_case_phrase_hits = _count_phrase_hits(normalized, USE_CASE_SIGNAL_PHRASES)
    use_case_regex_hits = sum(1 for pattern in USE_CASE_SIGNAL_REGEXES if re.search(pattern, normalized))
    use_case_hits = use_case_phrase_hits + use_case_regex_hits

    benefit_hits = _count_phrase_hits(normalized, BENEFIT_SIGNAL_PHRASES)

    feature_phrase_hits = _count_phrase_hits(normalized, FEATURE_SPEC_SIGNAL_PHRASES)
    measurement_hits = len(re.findall(r"\b\d+(\.\d+)?\s?(oz|ml|l|lb|lbs|in|inch|inches|cm|mm|pack|count)\b", normalized))
    count_pack_hits = len(re.findall(r"\b(pack of|set of|count of)\s+\d+\b", normalized))
    feature_hits = feature_phrase_hits + measurement_hits + count_pack_hits

    return {
        "word_count": word_count,
        "use_case_hits": use_case_hits,
        "benefit_hits": benefit_hits,
        "feature_hits": feature_hits,
        "use_case_phrase_hits": use_case_phrase_hits,
        "use_case_regex_hits": use_case_regex_hits,
        "feature_phrase_hits": feature_phrase_hits,
        "measurement_hits": measurement_hits,
        "count_pack_hits": count_pack_hits,
    }


def _word_count(text: str) -> int:
    return len(_tokens(text))


def _contains_promo_or_offer_language(text: str) -> bool:
    lowered = _lower(text)
    return any(term in lowered for term in PROMO_TERMS + OFFER_SHIPPING_AVAILABILITY_TERMS)


def _find_offer_shipping_availability_terms(text: str) -> list[str]:
    lowered = _lower(text)
    return [term for term in OFFER_SHIPPING_AVAILABILITY_TERMS if term in lowered]


def _has_disallowed_or_construction(title: str) -> bool:
    lowered = _lower(title)
    if " or " not in lowered:
        return False
    if any(safe in lowered for safe in ("and/or", "or more", "or less", "or above", "or below")):
        return False
    match = re.search(r"\b([a-z][a-z0-9\-]{2,})\s+or\s+([a-z][a-z0-9\-]{2,})\b", lowered)
    return bool(match)


def _find_suspicious_title_special_patterns(title: str) -> list[str]:
    hits: list[str] = []
    if any(ch in title for ch in TITLE_SUSPICIOUS_CHARS):
        hits.extend([ch for ch in TITLE_SUSPICIOUS_CHARS if ch in title])
    if re.search(r"([#*!|])\1", title):
        hits.append("repeated_special_characters")
    return hits


def _looks_like_bullet_style_description(body_text: str) -> bool:
    if not body_text:
        return False
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", body_text) if ln.strip()]
    if len(lines) < 2:
        return False
    bullet_like = 0
    for ln in lines:
        if re.match(r"^(\-|•|\*|\d+[\.\)])\s+", ln):
            bullet_like += 1
    return bullet_like >= 2


def _has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    t = _lower(text)
    return any(term in t for term in terms)


def _get_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    source_meta = record.get("source_metadata")
    ingest_meta = record.get("ingest_metadata")
    merged: dict[str, Any] = {}
    if isinstance(source_meta, dict):
        merged.update(source_meta)
    if isinstance(ingest_meta, dict):
        merged.update(ingest_meta)
    return merged


def _get_meta_value(meta: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in meta:
            return meta.get(key)
    return None


def _meta_text(meta: dict[str, Any], keys: tuple[str, ...]) -> str:
    value = _get_meta_value(meta, keys)
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "nan", "null"} else text


def _meta_int(meta: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    text = _meta_text(meta, keys)
    if not text:
        return None
    try:
        return int(float(text.replace(",", "")))
    except ValueError:
        return None


def _meta_float(meta: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    text = _meta_text(meta, keys)
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _is_key_feature_notes_context(notes: str) -> bool:
    n = _lower(notes)
    if not n:
        return False
    if "title" in n or "headline" in n:
        return False

    has_description_only_context = "description" in n and not any(
        term in n
        for term in (
            "bullet",
            "bullets",
            "key feature",
            "key-feature",
            "feature",
            "overlap",
            "duplicate",
            "sentence",
            "punctuation",
            "fragment",
        )
    )
    if has_description_only_context:
        return False
    return True


def _has_issue(findings: list[dict[str, Any]], issue_type: str) -> bool:
    return any(str(f.get("issue_type", "")) == issue_type for f in findings)


def _add_universal_finding(
    findings: list[dict[str, Any]],
    *,
    section: str,
    issue_type: str,
    message: str,
    severity: str,
    evidence: dict[str, Any],
    recommendation_theme: str,
) -> None:
    if _has_issue(findings, issue_type):
        return
    findings.append(
        _section_find(
            section=section,
            issue_type=issue_type,
            severity=severity,
            message=message,
            evidence=evidence,
            recommendation_theme=recommendation_theme,
            source="universal_rule",
        )
    )


def _resolve_walmart_top_level_category(record: dict[str, Any]) -> str | None:
    category = _lower(str(record.get("category", "")))
    subcategory = _lower(str(record.get("subcategory", "")))
    product_type = _lower(str(record.get("product_type", "")))
    title = _lower(str(record.get("current_title") or record.get("product_title") or ""))

    weighted_sources = (
        (category, 3),
        (subcategory, 2),
        (product_type, 2),
        (title, 1),
    )

    best_category: str | None = None
    best_score = 0
    for top_category, aliases in WALMART_TOP_CATEGORY_ALIASES.items():
        score = 0
        for source_text, weight in weighted_sources:
            if not source_text:
                continue
            for alias in aliases:
                alias_norm = _lower(alias)
                if alias_norm and alias_norm in source_text:
                    score += weight
                    break
        if score > best_score:
            best_score = score
            best_category = top_category

    return best_category if best_score > 0 else None


def _combined_use_case_term_bank(record: dict[str, Any]) -> tuple[str, ...]:
    top_category = _resolve_walmart_top_level_category(record)
    terms = list(UNIVERSAL_USE_CASE_TERMS)
    if top_category in PRIORITY_CATEGORY_USE_CASE_TERMS:
        terms.extend(PRIORITY_CATEGORY_USE_CASE_TERMS[top_category])
    elif top_category in LIGHT_CATEGORY_USE_CASE_HINTS:
        terms.extend(LIGHT_CATEGORY_USE_CASE_HINTS[top_category])

    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        t = _lower(term)
        if not t or t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return tuple(deduped)


def _category_candidate_texts(record: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("category", "subcategory", "product_type", "department", "item_type", "item_category"):
        val = _norm(str(record.get(key, "")))
        if val:
            candidates.append(val.lower())

    meta = _get_record_metadata(record)
    for key, value in meta.items():
        key_l = str(key).lower()
        if not any(token in key_l for token in ("category", "subcategory", "department", "class", "type")):
            continue
        text = _norm(str(value))
        if text:
            candidates.append(text.lower())

    seen: set[str] = set()
    deduped: list[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        deduped.append(c)
    return deduped


def _resolve_internal_category_bucket(record: dict[str, Any], resolved_top_category: str | None = None) -> str | None:
    if resolved_top_category and resolved_top_category in WALMART_TOP_TO_INTERNAL_BUCKET:
        return WALMART_TOP_TO_INTERNAL_BUCKET[resolved_top_category]

    candidates = _category_candidate_texts(record)
    if not candidates:
        return None

    bucket_scores: dict[str, int] = {bucket: 0 for bucket in INTERNAL_CATEGORY_BUCKET_ALIASES}
    for text in candidates:
        for bucket, aliases in INTERNAL_CATEGORY_BUCKET_ALIASES.items():
            for alias in aliases:
                if alias in text:
                    bucket_scores[bucket] += 1
                    break

    best_bucket = max(bucket_scores, key=bucket_scores.get)
    best_score = bucket_scores.get(best_bucket, 0)
    if best_score >= 2:
        return best_bucket
    return None


def _category_expectation_profile(internal_bucket: str | None) -> dict[str, int]:
    default_profile = CATEGORY_EXPECTATION_PROFILES["General Merchandise / Fallback"]
    if not internal_bucket:
        return dict(default_profile)
    return dict(CATEGORY_EXPECTATION_PROFILES.get(internal_bucket, default_profile))


def _category_use_case_hint_hits(text: str, internal_bucket: str | None) -> int:
    if not internal_bucket:
        return 0
    hints = CATEGORY_USE_CASE_HINTS.get(internal_bucket, ())
    if not hints:
        return 0
    return _count_phrase_hits(_lower(text), hints)


def _section_find(
    *,
    section: str,
    issue_type: str,
    severity: str,
    message: str,
    evidence: dict[str, Any],
    recommendation_theme: str,
    source: str = "rule_check",
    content_source: str | None = None,
) -> dict[str, Any]:
    finding = {
        "finding_id": f"f-{uuid4().hex[:10]}",
        "section": section,
        "issue_type": issue_type,
        "severity": severity,
        "message": message,
        "evidence": evidence,
        "recommendation_theme": recommendation_theme,
        "source": source,
    }
    if content_source:
        finding["content_source"] = content_source
    return finding


def analyze_title(record: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    title = _norm(record.get("current_title") or record.get("product_title") or "")
    title_lower = title.lower()
    meta = _get_record_metadata(record)
    title_count = _meta_int(meta, ("title_count", "Title Count", "titleCount"))
    title_notes = _meta_text(meta, ("title_notes", "Title Notes", "titleNotes")).lower()

    if not title:
        findings.append(
            _section_find(
                section="title",
                issue_type="missing_title",
                severity="high",
                message="No usable product title was extracted.",
                evidence={"title": title},
                recommendation_theme="clarity",
            )
        )
        return findings

    if title_count is not None and title_count > 90 and not _has_issue(findings, "title_too_long"):
        findings.append(
            _section_find(
                section="title",
                issue_type="title_too_long",
                severity="high",
                message="Title exceeds the 90-character baseline.",
                evidence={"title_length": title_count, "max_length": 90},
                recommendation_theme="seo",
                source="sheet_metadata",
                content_source="audit_extract_sheet",
            )
        )

    if len(title) > 90:
        findings.append(
            _section_find(
                section="title",
                issue_type="title_too_long",
                severity="high",
                message="Title exceeds the 90-character baseline.",
                evidence={"title_length": len(title), "max_length": 90},
                recommendation_theme="seo",
            )
        )

    if any(term in title_lower for term in PROMO_TERMS):
        findings.append(
            _section_find(
                section="title",
                issue_type="promo_language",
                severity="high",
                message="Title contains promo language that should be avoided.",
                evidence={"title": title, "promo_terms": [t for t in PROMO_TERMS if t in title_lower]},
                recommendation_theme="formatting",
            )
        )

    offer_terms = _find_offer_shipping_availability_terms(title)
    if offer_terms and not _has_issue(findings, "offer_shipping_availability_language"):
        findings.append(
            _section_find(
                section="title",
                issue_type="offer_shipping_availability_language",
                severity="high",
                message="Title contains offer/shipping/availability language that should be removed.",
                evidence={"title": title, "offer_shipping_terms": offer_terms},
                recommendation_theme="formatting",
                source="heuristic",
            )
        )

    if _has_disallowed_or_construction(title):
        findings.append(
            _section_find(
                section="title",
                issue_type="disallowed_or_construction",
                severity="high",
                message='Title uses an "or" construction that may imply ambiguous product naming.',
                evidence={"title": title},
                recommendation_theme="clarity",
                source="heuristic",
            )
        )

    suspicious_patterns = _find_suspicious_title_special_patterns(title)
    if suspicious_patterns:
        findings.append(
            _section_find(
                section="title",
                issue_type="suspicious_special_characters",
                severity="high",
                message="Title contains suspicious special-character patterns.",
                evidence={"title": title, "suspicious_patterns": suspicious_patterns},
                recommendation_theme="formatting",
                source="heuristic",
            )
        )

    letters = [ch for ch in title if ch.isalpha()]
    if len(letters) >= 8 and all(ch.isupper() for ch in letters):
        findings.append(
            _section_find(
                section="title",
                issue_type="all_caps",
                severity="medium",
                message="Title appears to be in all caps.",
                evidence={"title": title},
                recommendation_theme="clarity",
            )
        )

    title_tokens = [t for t in _tokens(title) if len(t) >= 3]
    if title_tokens:
        counts = Counter(title_tokens)
        most_common_word, most_common_count = counts.most_common(1)[0]
        if most_common_count >= 3:
            findings.append(
                _section_find(
                    section="title",
                    issue_type="keyword_stuffing",
                    severity="low",
                    message="Title shows repeated keyword usage.",
                    evidence={
                        "word": most_common_word,
                        "repetitions": most_common_count,
                        "title": title,
                    },
                    recommendation_theme="seo",
                source="heuristic",
            )
        )

    if title_count is not None and title_count < 20 and not _has_issue(findings, "too_short"):
        findings.append(
            _section_find(
                section="title",
                issue_type="too_short",
                severity="medium",
                message="Title may be too short to be descriptive.",
                evidence={"title_length": title_count, "title": title},
                recommendation_theme="clarity",
                source="sheet_metadata",
                content_source="audit_extract_sheet",
            )
        )

    if len(title) < 20:
        findings.append(
            _section_find(
                section="title",
                issue_type="too_short",
                severity="low",
                message="Title may be too short to be descriptive.",
                evidence={"title_length": len(title), "title": title},
                recommendation_theme="clarity",
                source="heuristic",
            )
        )

    if title_notes:
        if ("too long" in title_notes or "length" in title_notes) and not _has_issue(findings, "title_too_long"):
            findings.append(
                _section_find(
                    section="title",
                    issue_type="title_too_long",
                    severity="medium",
                    message="Extension notes indicate the title should be tightened.",
                    evidence={"title_notes": title_notes},
                    recommendation_theme="seo",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if (
            "too short" in title_notes
            or "not descriptive" in title_notes
            or "missing keyword" in title_notes
        ) and not _has_issue(findings, "too_short"):
            findings.append(
                _section_find(
                    section="title",
                    issue_type="too_short",
                    severity="medium",
                    message="Extension notes indicate the title lacks sufficient detail.",
                    evidence={"title_notes": title_notes},
                    recommendation_theme="clarity",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if "promo" in title_notes and not _has_issue(findings, "promo_language"):
            findings.append(
                _section_find(
                    section="title",
                    issue_type="promo_language",
                    severity="medium",
                    message="Extension notes flag promotional wording in the title.",
                    evidence={"title_notes": title_notes},
                    recommendation_theme="formatting",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("all caps" in title_notes or "capitalization" in title_notes) and not _has_issue(findings, "all_caps"):
            findings.append(
                _section_find(
                    section="title",
                    issue_type="all_caps",
                    severity="medium",
                    message="Extension notes indicate capitalization formatting issues in the title.",
                    evidence={"title_notes": title_notes},
                    recommendation_theme="clarity",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )

    return findings


def analyze_description(record: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    body = _norm(record.get("current_description_body", ""))
    bullets = [str(b).strip() for b in record.get("current_description_bullets", []) if str(b).strip()]
    combined = _norm(record.get("current_description_combined", "") or " ".join([body, *bullets]))
    visible_body = _visible_text(body)
    visible_combined = _visible_text(combined)
    combined_lower = visible_combined.lower()
    wc = _word_count(visible_combined)
    resolved_top_category = _resolve_walmart_top_level_category(record)
    internal_bucket = _resolve_internal_category_bucket(record, resolved_top_category)
    category_profile = _category_expectation_profile(internal_bucket)
    language_profile = _profile_description_language(visible_combined)
    meta = _get_record_metadata(record)
    description_count = _meta_int(meta, ("description_count", "Description Count", "descriptionCount"))
    description_notes = _meta_text(meta, ("description_notes", "Description Notes", "descriptionNotes")).lower()
    content_score = _meta_float(meta, ("content_score", "Content Score", "contentScore"))

    if wc < 40:
        findings.append(
            _section_find(
                section="description",
                issue_type="too_short",
                severity="high",
                message="Description content is below the 40-word minimum baseline.",
                evidence={"word_count": wc, "minimum_words": 40},
                recommendation_theme="seo",
            )
        )
    elif wc < 100:
        findings.append(
            _section_find(
                section="description",
                issue_type="below_recommended_length",
                severity="low",
                message="Description is below the 100-word recommended length.",
                evidence={"word_count": wc, "recommended_words": 100},
                recommendation_theme="clarity",
            )
        )

    if description_count is not None:
        if description_count < 40 and not _has_issue(findings, "too_short"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="too_short",
                    severity="high",
                    message="Description content is below the 40-word minimum baseline.",
                    evidence={"word_count": description_count, "minimum_words": 40},
                    recommendation_theme="seo",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        elif description_count < 100 and not _has_issue(findings, "below_recommended_length"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="below_recommended_length",
                    severity="low",
                    message="Description is below the 100-word recommended length.",
                    evidence={"word_count": description_count, "recommended_words": 100},
                    recommendation_theme="clarity",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )

    title = _norm(record.get("current_title") or record.get("product_title") or "")
    title_anchor = _product_name_anchor(title)
    if title_anchor and not _is_product_name_repeated_in_text(title=title, text=visible_combined):
        findings.append(
            _section_find(
                section="description",
                issue_type="missing_product_name",
                severity="medium",
                message="Description does not appear to repeat the product name.",
                evidence={"product_name_anchor": title_anchor},
                recommendation_theme="seo",
                source="heuristic",
            )
        )

    retailer_hits = [t for t in RETAILER_TERMS if t in combined_lower]
    redirect_hits = [t for t in REDIRECT_TERMS if t in combined_lower]
    if retailer_hits or redirect_hits:
        findings.append(
            _section_find(
                section="description",
                issue_type="retailer_mention",
                severity="high",
                message="Description appears to mention another retailer or external redirect language.",
                evidence={"retailer_terms_detected": retailer_hits, "redirect_terms_detected": redirect_hits},
                recommendation_theme="formatting",
            )
        )

    promo_hits = [t for t in PROMO_TERMS + OFFER_SHIPPING_AVAILABILITY_TERMS if t in combined_lower]
    if promo_hits and not _has_issue(findings, "promo_language"):
        findings.append(
            _section_find(
                section="description",
                issue_type="promo_language",
                severity="high",
                message="Description contains promotional/offer language that should be removed.",
                evidence={"promo_terms_detected": promo_hits},
                recommendation_theme="formatting",
                source="heuristic",
            )
        )

    has_list_html = bool(re.search(r"(?is)<\s*(ul|ol|li)\b", body))
    has_bullet_style_lines = _looks_like_bullet_style_description(_strip_html_tags(body))
    if (has_list_html or has_bullet_style_lines) and not _has_issue(findings, "bullet_style_description"):
        findings.append(
            _section_find(
                section="description",
                issue_type="bullet_style_description",
                severity="medium",
                message="Description appears bullet-formatted instead of paragraph-form.",
                evidence={"has_list_html": has_list_html, "has_bullet_style_lines": has_bullet_style_lines},
                recommendation_theme="formatting",
                source="heuristic",
            )
        )

    if not body and bullets:
        findings.append(
            _section_find(
                section="description",
                issue_type="generic_content",
                severity="medium",
                message="Description appears to be bullet-only with no clear paragraph body.",
                evidence={"body_present": False, "bullet_count": len(bullets)},
                recommendation_theme="clarity",
                source="heuristic",
            )
        )

    category_use_case_hits = _category_use_case_hint_hits(visible_combined, internal_bucket)
    total_use_case_hits = language_profile["use_case_hits"] + category_use_case_hits
    has_use_case_language = total_use_case_hits >= category_profile["use_case_min_hits"]
    if visible_combined and wc >= category_profile["use_case_min_wc"] and not has_use_case_language:
        findings.append(
            _section_find(
                section="description",
                issue_type="missing_use_case",
                severity="medium",
                message="Description lacks clear use-case language.",
                evidence={
                    "word_count": wc,
                    "use_case_hits": total_use_case_hits,
                    "category_use_case_hits": category_use_case_hits,
                    "resolved_top_category": resolved_top_category or "",
                    "internal_category_bucket": internal_bucket or "",
                },
                recommendation_theme="use_case",
                source="heuristic",
            )
        )

    if visible_combined and not _has_any_term(visible_combined, OUTCOME_TERMS):
        findings.append(
            _section_find(
                section="description",
                issue_type="missing_outcome_focus",
                severity="low",
                message="Description does not clearly communicate outcomes or shopper benefit.",
                evidence={"word_count": wc},
                recommendation_theme="conversion",
                source="heuristic",
            )
        )

    has_html_structure = _has_valid_html_container(combined)
    if visible_combined and not has_html_structure:
        findings.append(
            _section_find(
                section="description",
                issue_type="html_structure_missing",
                severity="low",
                message="Description appears plain-text only; consider adding clean HTML structure if channel supports it.",
                evidence={"html_structure_detected": has_html_structure},
                recommendation_theme="formatting",
                source="heuristic",
            )
        )

    if description_notes:
        if ("too short" in description_notes or "thin" in description_notes) and not _has_issue(findings, "too_short"):
            inferred_wc = description_count if description_count is not None else wc
            if inferred_wc < 40:
                findings.append(
                    _section_find(
                        section="description",
                        issue_type="too_short",
                        severity="high",
                        message="Extension notes indicate description depth is below minimum.",
                        evidence={"description_notes": description_notes, "word_count": inferred_wc, "minimum_words": 40},
                        recommendation_theme="seo",
                        source="sheet_metadata",
                        content_source="audit_extract_sheet",
                    )
                )
            elif not _has_issue(findings, "below_recommended_length"):
                findings.append(
                    _section_find(
                        section="description",
                        issue_type="below_recommended_length",
                        severity="low",
                        message="Extension notes indicate description depth could be improved.",
                        evidence={"description_notes": description_notes, "word_count": inferred_wc, "recommended_words": 100},
                        recommendation_theme="clarity",
                        source="sheet_metadata",
                        content_source="audit_extract_sheet",
                    )
                )
        if ("too long" in description_notes or "verbose" in description_notes) and not _has_issue(findings, "too_long"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="too_long",
                    severity="low",
                    message="Description may be overly long and should be tightened for scanability.",
                    evidence={"description_notes": description_notes},
                    recommendation_theme="clarity",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("generic" in description_notes or "vague" in description_notes) and not _has_issue(findings, "generic_content"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="generic_content",
                    severity="medium",
                    message="Extension notes indicate description copy is generic.",
                    evidence={"description_notes": description_notes},
                    recommendation_theme="differentiation",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("use case" in description_notes or "usage" in description_notes) and not _has_issue(findings, "missing_use_case"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="missing_use_case",
                    severity="medium",
                    message="Extension notes indicate missing use-case language.",
                    evidence={"description_notes": description_notes},
                    recommendation_theme="use_case",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("benefit" in description_notes or "outcome" in description_notes) and not _has_issue(findings, "missing_outcome_focus"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="missing_outcome_focus",
                    severity="low",
                    message="Extension notes indicate weak benefit/outcome language.",
                    evidence={"description_notes": description_notes},
                    recommendation_theme="conversion",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("html" in description_notes or "format" in description_notes) and not _has_issue(findings, "html_structure_missing"):
            findings.append(
                _section_find(
                    section="description",
                    issue_type="html_structure_missing",
                    severity="low",
                    message="Extension notes indicate description formatting/HTML structure needs cleanup.",
                    evidence={"description_notes": description_notes},
                    recommendation_theme="formatting",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )

    if content_score is not None and content_score < 75:
        findings.append(
            _section_find(
                section="description",
                issue_type="content_score_low",
                severity="medium" if content_score < 60 else "low",
                message="Content score indicates broader copy quality gaps.",
                evidence={"content_score": content_score, "target_minimum": 75},
                recommendation_theme="conversion",
                source="sheet_metadata",
                content_source="audit_extract_sheet",
            )
        )

    unique_ratio = (len(set(_tokens(visible_combined))) / max(1, len(_tokens(visible_combined)))) if visible_combined else 0.0
    if wc > 0 and wc < 80 and unique_ratio < 0.55:
        findings.append(
            _section_find(
                section="description",
                issue_type="generic_content",
                severity="low",
                message="Description appears thin or generic.",
                evidence={"word_count": wc, "unique_word_ratio": round(unique_ratio, 2)},
                recommendation_theme="differentiation",
                source="heuristic",
            )
        )

    # Additive approved universal statements layer for description recommendations.
    if visible_combined and not has_html_structure:
        _add_universal_finding(
            findings,
            section="description",
            issue_type="universal_desc_not_html_format",
            message="Not in HTML format (<p>...</p>)",
            severity="low",
            evidence={"html_structure_detected": has_html_structure},
            recommendation_theme="formatting",
        )

    effective_wc = wc
    if effective_wc < 60:
        _add_universal_finding(
            findings,
            section="description",
            issue_type="universal_desc_below_60_word_minimum",
            message="Below 60-word minimum",
            severity="low",
            evidence={"word_count": effective_wc},
            recommendation_theme="seo",
        )

    if title_anchor and not _is_product_name_repeated_in_text(title=title, text=visible_body):
        _add_universal_finding(
            findings,
            section="description",
            issue_type="universal_desc_product_name_not_repeated_for_seo",
            message="Product name not repeated for SEO",
            severity="medium",
            evidence={"product_name_anchor": title_anchor},
            recommendation_theme="seo",
        )

    has_use_case = has_use_case_language
    has_outcome = _has_any_term(visible_combined, OUTCOME_TERMS)
    if (
        visible_combined
        and wc >= 40
        and not has_use_case
        and not has_outcome
        and not _has_issue(findings, "missing_shopper_usefulness")
    ):
        findings.append(
            _section_find(
                section="description",
                issue_type="missing_shopper_usefulness",
                severity="low",
                message="Description lacks useful shopper-facing context and outcome framing.",
                evidence={"word_count": wc, "use_case_hits": total_use_case_hits},
                recommendation_theme="conversion",
                source="heuristic",
            )
        )

    if visible_combined and wc >= category_profile["use_case_min_wc"] and not has_use_case:
        _add_universal_finding(
            findings,
            section="description",
            issue_type="universal_desc_missing_clear_use_cases",
            message="Missing clear use cases",
            severity="medium",
            evidence={
                "word_count": wc,
                "use_case_hits": total_use_case_hits,
                "category_use_case_hits": category_use_case_hits,
                "resolved_top_category": resolved_top_category or "",
                "internal_category_bucket": internal_bucket or "",
            },
            recommendation_theme="use_case",
        )

    feature_term_hits = language_profile["feature_hits"]
    benefit_hits = language_profile["benefit_hits"]
    feature_dominant = (
        feature_term_hits >= category_profile["feature_high"]
        or (feature_term_hits >= category_profile["feature_mid"] and benefit_hits <= category_profile["feature_mid_benefit_max"])
        or (
            feature_term_hits >= category_profile["feature_low"]
            and language_profile["measurement_hits"] >= category_profile["feature_measurement_min"]
            and benefit_hits == 0
        )
    )
    if (
        wc >= category_profile["feature_focus_min_wc"]
        and feature_dominant
        and not has_use_case
        and not has_outcome
        and benefit_hits == 0
    ):
        _add_universal_finding(
            findings,
            section="description",
            issue_type="universal_desc_too_feature_focused_not_benefit_use_case_driven",
            message="Too feature-focused, not benefit/use-case driven",
            severity="medium",
            evidence={
                "feature_term_hits": feature_term_hits,
                "benefit_hits": benefit_hits,
                "use_case_hits": total_use_case_hits,
                "category_use_case_hits": category_use_case_hits,
                "measurement_hits": language_profile["measurement_hits"],
                "word_count": wc,
                "internal_category_bucket": internal_bucket or "",
            },
            recommendation_theme="conversion",
        )

    has_duplication, repetition_evidence = _has_substantial_internal_repetition(visible_combined)
    if visible_combined and has_duplication:
        _add_universal_finding(
            findings,
            section="description",
            issue_type="universal_desc_likely_duplicated_content",
            message="Likely duplicated content",
            severity="low",
            evidence={
                "word_count": wc,
                "unique_word_ratio": round(unique_ratio, 2),
                **repetition_evidence,
            },
            recommendation_theme="differentiation",
        )

    return findings


def analyze_key_features(record: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    features = record.get("current_key_features", []) or []
    bullets = [_norm(str(f.get("text", ""))) for f in features if _norm(str(f.get("text", "")))]
    bullet_count = len(bullets)
    full_desc = _visible_text(_norm(record.get("current_description_combined", "")))
    resolved_top_category = _resolve_walmart_top_level_category(record)
    internal_bucket = _resolve_internal_category_bucket(record, resolved_top_category)
    category_profile = _category_expectation_profile(internal_bucket)
    meta = _get_record_metadata(record)
    meta_bullet_count = _meta_int(
        meta,
        ("description_bullet_count", "Description Bullet Count", "descriptionBulletCount"),
    )
    bullet_notes = _meta_text(
        meta,
        ("description_bullet_notes", "Description Bullet Notes", "descriptionBulletNotes"),
    ).lower()

    if bullet_count < 3:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="insufficient_bullets",
                severity="high",
                message="Key features contain fewer than 3 bullets.",
                evidence={"bullet_count": bullet_count, "minimum_bullets": 3},
                recommendation_theme="clarity",
            )
        )
    elif bullet_count == 5:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="recommended_bullets_missing",
                severity="low",
                message="Key features are below the 5-bullet recommendation.",
                evidence={"bullet_count": bullet_count, "recommended_bullets": 5},
                recommendation_theme="conversion",
            )
        )
    elif bullet_count > 6:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="over_recommended_bullets",
                severity="high" if bullet_count >= 10 else "medium",
                message="Key features exceed the 6-bullet recommended limit.",
                evidence={"bullet_count": bullet_count, "recommended_bullets": 6},
                recommendation_theme="clarity",
                source="heuristic",
            )
        )

    if meta_bullet_count is not None:
        if meta_bullet_count < 3 and not _has_issue(findings, "insufficient_bullets"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="insufficient_bullets",
                    severity="high",
                    message="Key features contain fewer than 3 bullets.",
                    evidence={"bullet_count": meta_bullet_count, "minimum_bullets": 3},
                    recommendation_theme="clarity",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        elif meta_bullet_count > 6 and not _has_issue(findings, "over_recommended_bullets"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="over_recommended_bullets",
                    severity="high" if meta_bullet_count >= 10 else "medium",
                    message="Key features exceed the 6-bullet recommended limit.",
                    evidence={"bullet_count": meta_bullet_count, "recommended_bullets": 6},
                    recommendation_theme="clarity",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        elif meta_bullet_count == 5 and not _has_issue(findings, "recommended_bullets_missing"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="recommended_bullets_missing",
                    severity="low",
                    message="Key features are below the 5-bullet recommendation.",
                    evidence={"bullet_count": meta_bullet_count, "recommended_bullets": 5},
                    recommendation_theme="conversion",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )

    punctuated = [b for b in bullets if re.search(r"[.!?;:]\s*$", b)]
    if punctuated:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="ending_punctuation",
                severity="medium",
                message="One or more bullets end with punctuation.",
                evidence={"examples": punctuated[:3], "count": len(punctuated)},
                recommendation_theme="formatting",
            )
        )

    sentence_like = []
    for b in bullets:
        words = _tokens(b)
        if len(words) >= 8 and (re.search(r"[.!?]\s*$", b) or re.search(r"\b(this|it|you|we|they|is|are|will|can)\b", b.lower())):
            sentence_like.append(b)
    if sentence_like:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="full_sentences",
                severity="medium",
                message="One or more bullets read like full sentences rather than fragments.",
                evidence={"examples": sentence_like[:3], "count": len(sentence_like)},
                recommendation_theme="formatting",
                source="heuristic",
            )
        )

    starts_upper = [bool(re.match(r"^[A-Z]", b)) for b in bullets if b]
    if starts_upper and not all(starts_upper):
        findings.append(
            _section_find(
                section="key_features",
                issue_type="inconsistent_formatting",
                severity="low",
                message="Bullet capitalization appears inconsistent.",
                evidence={"starts_with_capital_count": sum(starts_upper), "bullet_count": len(starts_upper)},
                recommendation_theme="formatting",
                source="heuristic",
            )
        )

    joined = " ".join(bullets).lower()
    promo_hits = [t for t in PROMO_TERMS + OFFER_SHIPPING_AVAILABILITY_TERMS if t in joined]
    if promo_hits:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="promo_language",
                severity="high",
                message="Key features contain promotional/offer language.",
                evidence={"promo_terms_detected": promo_hits},
                recommendation_theme="formatting",
                source="heuristic",
            )
        )

    retailer_hits = [t for t in RETAILER_TERMS if t in joined]
    redirect_hits = [t for t in REDIRECT_TERMS if t in joined]
    if retailer_hits or redirect_hits:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="retailer_mention",
                severity="high",
                message="Key features mention another retailer or external redirect language.",
                evidence={"retailer_terms_detected": retailer_hits, "redirect_terms_detected": redirect_hits},
                recommendation_theme="formatting",
            )
        )

    forbidden_hits = [c for c in FORBIDDEN_SPECIAL_CHARS if c in " ".join(bullets)]
    if forbidden_hits:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="forbidden_special_characters",
                severity="medium",
                message="Key features include forbidden special characters.",
                evidence={"forbidden_characters": forbidden_hits},
                recommendation_theme="formatting",
            )
        )

    overlap_count = _count_bullets_overlapping_description(bullets, full_desc)
    if bullet_count > 0 and overlap_count / bullet_count >= 0.5:
        findings.append(
            _section_find(
                section="key_features",
                issue_type="overlap_with_other_section",
                severity="medium",
                message="Key features heavily overlap with the description content.",
                evidence={"overlapping_bullets": overlap_count, "bullet_count": bullet_count},
                recommendation_theme="differentiation",
                source="heuristic",
            )
        )

    if bullets and not _has_any_term(joined, USE_CASE_TERMS + OUTCOME_TERMS):
        findings.append(
            _section_find(
                section="key_features",
                issue_type="missing_use_case",
                severity="low",
                message="Key features have weak use-case or benefit language.",
                evidence={"bullet_count": bullet_count},
                recommendation_theme="use_case",
                source="heuristic",
            )
        )

    if bullet_notes and _is_key_feature_notes_context(bullet_notes):
        if ("full sentence" in bullet_notes or "sentence" in bullet_notes) and not _has_issue(findings, "full_sentences"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="full_sentences",
                    severity="medium",
                    message="Extension notes indicate bullets read as full sentences.",
                    evidence={"description_bullet_notes": bullet_notes},
                    recommendation_theme="formatting",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("punctuation" in bullet_notes or "period" in bullet_notes) and not _has_issue(findings, "ending_punctuation"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="ending_punctuation",
                    severity="medium",
                    message="Extension notes indicate bullet punctuation cleanup is needed.",
                    evidence={"description_bullet_notes": bullet_notes},
                    recommendation_theme="formatting",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("overlap" in bullet_notes or "duplicate" in bullet_notes) and not _has_issue(findings, "overlap_with_other_section"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="overlap_with_other_section",
                    severity="medium",
                    message="Extension notes indicate overlap between bullets and description.",
                    evidence={"description_bullet_notes": bullet_notes},
                    recommendation_theme="differentiation",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )
        if ("benefit" in bullet_notes or "use case" in bullet_notes) and not _has_issue(findings, "missing_use_case"):
            findings.append(
                _section_find(
                    section="key_features",
                    issue_type="missing_use_case",
                    severity="low",
                    message="Extension notes indicate bullets need stronger benefit/use-case language.",
                    evidence={"description_bullet_notes": bullet_notes},
                    recommendation_theme="use_case",
                    source="sheet_metadata",
                    content_source="audit_extract_sheet",
                )
            )

    # Additive approved universal statements layer for key features recommendations.
    sentence_like_trigger = bool(sentence_like) or bool(punctuated)
    if sentence_like_trigger:
        _add_universal_finding(
            findings,
            section="key_features",
            issue_type="universal_kf_bullets_full_sentences_should_be_fragments_no_punctuation",
            message="Bullets are full sentences (should be fragments, no punctuation)",
            severity="medium",
            evidence={"sentence_like_count": len(sentence_like), "punctuated_count": len(punctuated)},
            recommendation_theme="formatting",
        )

    repeated_start_count = 0
    if bullets:
        starts = [" ".join(_tokens(b)[:3]) for b in bullets if _tokens(b)]
        start_counts = Counter([s for s in starts if s])
        repeated_start_count = max(start_counts.values()) if start_counts else 0
    duplicate_ratio = (len(set([b.lower() for b in bullets])) / max(1, len(bullets))) if bullets else 1.0
    has_substantial_bullet_repetition, repetition_evidence = _has_substantial_bullet_repetition(bullets)
    if bullet_count >= 3 and (has_substantial_bullet_repetition or repeated_start_count >= 3 or duplicate_ratio < 0.7):
        _add_universal_finding(
            findings,
            section="key_features",
            issue_type="universal_kf_duplicate_repetitive_content",
            message="Duplicate/repetitive content",
            severity="medium",
            evidence={
                "bullet_count": bullet_count,
                "repeated_start_count": repeated_start_count,
                "unique_ratio": round(duplicate_ratio, 2),
                **repetition_evidence,
            },
            recommendation_theme="differentiation",
        )

    sentence_style_flags = [bool(re.search(r"[.!?]\s*$", b)) for b in bullets]
    if bullets and ((starts_upper and not all(starts_upper)) or (sentence_style_flags and any(sentence_style_flags) and not all(sentence_style_flags))):
        _add_universal_finding(
            findings,
            section="key_features",
            issue_type="universal_kf_inconsistent_formatting_capitalization_structure",
            message="Inconsistent formatting (capitalization/structure)",
            severity="low",
            evidence={
                "starts_upper_mixed": bool(starts_upper and not all(starts_upper)),
                "sentence_style_mixed": bool(sentence_style_flags and any(sentence_style_flags) and not all(sentence_style_flags)),
            },
            recommendation_theme="formatting",
        )

    manufacturing_hits = sum(1 for term in MANUFACTURING_DETAIL_TERMS if term in joined)
    benefit_hits = sum(1 for term in OUTCOME_TERMS + UNIVERSAL_USE_CASE_TERMS if term in joined)
    if (
        bullet_count >= 2
        and manufacturing_hits >= category_profile["kf_manufacturing_hit_min"]
        and benefit_hits <= 1
    ):
        _add_universal_finding(
            findings,
            section="key_features",
            issue_type="universal_kf_too_much_manufacturing_detail",
            message="Too much manufacturing detail",
            severity="low",
            evidence={
                "manufacturing_hits": manufacturing_hits,
                "benefit_hits": benefit_hits,
                "internal_category_bucket": internal_bucket or "",
            },
            recommendation_theme="conversion",
        )

    if bullet_count > 0 and overlap_count / bullet_count >= 0.4:
        _add_universal_finding(
            findings,
            section="key_features",
            issue_type="universal_kf_repeats_description_content",
            message="Repeats description content",
            severity="medium",
            evidence={"overlapping_bullets": overlap_count, "bullet_count": bullet_count},
            recommendation_theme="differentiation",
        )

    return findings


def analyze_images(record: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    images = record.get("images", []) or []
    image_count = len(images)
    has_hero = any(bool(img.get("is_hero")) for img in images)

    if image_count == 0:
        findings.append(
            _section_find(
                section="images",
                issue_type="missing_images",
                severity="high",
                message="No usable product images were extracted.",
                evidence={"image_count": image_count},
                recommendation_theme="imagery",
            )
        )
        return findings

    if image_count < 3:
        findings.append(
            _section_find(
                section="images",
                issue_type="low_image_count",
                severity="medium",
                message="Image stack appears limited.",
                evidence={"image_count": image_count, "recommended_minimum": 3},
                recommendation_theme="imagery",
            )
        )

    if not has_hero:
        findings.append(
            _section_find(
                section="images",
                issue_type="missing_clear_hero",
                severity="low",
                message="No explicit hero flag was detected; first image fallback is being used.",
                evidence={"image_count": image_count, "hero_flag_detected": has_hero},
                recommendation_theme="imagery",
                source="heuristic",
            )
        )

    return findings


def analyze_primary_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    if (record.get("source_type") or "").lower() != "primary":
        return []

    findings: list[dict[str, Any]] = []
    findings.extend(analyze_title(record))
    findings.extend(analyze_description(record))
    findings.extend(analyze_key_features(record))
    findings.extend(analyze_images(record))
    return findings
