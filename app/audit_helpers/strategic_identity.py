"""Guide normalization and identity resolution for the new strategic PPTX path."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.audit_helpers.image_guides import (
    get_image_guide_page,
    load_image_guide,
    normalize_product_type,
    resolve_image_guide_category,
)


STYLE_GUIDE_FILES: dict[str, str] = {
    "garden_patio": "gardenandpatio.json",
    "everything_else": "everythingelse.json",
    "business_industrial": "businessandindustrial.json",
    "food_beverage": "food_beverage.json",
    "beauty": "beauty.json",
    "health_personal_care": "healthpersonal.json",
    "animals": "animals.json",
    "electronics": "electronics.json",
    "photography": "photography.json",
    "media": "media.json",
    "seasonal": "seasonal.json",
    "sports_outdoors": "sportsoutdoors.json",
    "home_improvement": "homeimprovement.json",
    "home": "home.json",
    "musical_instruments": "musicalinstruments.json",
    "office_stationery": "officeandstationery.json",
    "safety_emergency": "safetyemergency.json",
    "vehicle": "vehiclespartsandaccessories.json",
    "arts_crafts": "artsandcrafts.json",
    "baby": "baby.json",
    "fashion": "fashion.json",
    "furniture": "furniture.json",
    "health_general": "healthgeneral.json",
    "household_clean": "householdclean.json",
    "toys": "toys.json"
}

IMAGE_GUIDE_FILES: dict[str, str] = {
    "garden_patio": "gardenandpatioimg.json",
    "everything_else": "everythingelseimg.json",
    "business_industrial": "businessandindustrialimg.json",
    "food_beverage": "food_beverageimg.json",
    "beauty": "beautyimg.json",
    "health_personal_care": "healthpersonalimg.json",
    "animals": "animalsimg.json",
    "electronics": "electronicsimg.json",
    "photography": "photographyimg.json",
    "media": "mediaimg.json",
    "seasonal": "seasonalimg.json",
    "sports_outdoors": "sportsoutdoorsimg.json",
    "home_improvement": "homeimprovementimg.json",
    "home": "homeimg.json",
    "musical_instruments": "musicalinstrumentsimg.json",
    "office_stationery": "officeandstationeryimg.json",
    "safety_emergency": "safetyemergencyimg.json",
    "vehicle": "vehiclespartsandaccessoriesimg.json",
    "arts_crafts": "artsandcraftsimg.json",
    "baby": "babyimg.json",
    "fashion": "fashionimg.json",
    "furniture": "furnitureimg.json",
    "health_general": "healthgeneralimg.json",
    "household_clean": "householdcleanimg.json",
    "toys": "toysimg.json"
}

CATEGORY_VOCABULARY: dict[str, dict[str, str]] = {
    "garden_patio": {
        "benefit": "outdoor comfort and garden care benefits",
        "ingredient": "outdoor materials and weather resistance",
        "usage": "gardening and patio gathering guidance",
        "education": "plant care and outdoor maintenance education",
        "visual": "inviting gardens and outdoor living spaces",
        "navigation": "outdoor space and gardening task navigation",
    },
    "everything_else": {
        "benefit": "practical value and distinctive product benefits",
        "ingredient": "product composition and included contents",
        "usage": "everyday purpose and occasion guidance",
        "education": "product selection and ownership essentials",
        "visual": "clear product context and identifying details",
        "navigation": "product purpose and shopper need navigation",
    },
    "business_industrial": {
        "benefit": "workplace productivity and operational benefits",
        "ingredient": "industrial materials and equipment specifications",
        "usage": "professional tasks and workflow guidance",
        "education": "equipment capacity and operating requirements",
        "visual": "real workplace applications and equipment detail",
        "navigation": "trade, application, and capacity navigation",
    },
    "photography": {
        "benefit": "creative control and image capture benefits",
        "ingredient": "optical specifications and camera compatibility",
        "usage": "shooting situations and creative setup guidance",
        "education": "lens selection and imaging system education",
        "visual": "photography setups and equipment handling detail",
        "navigation": "camera system and shooting style navigation",
    },
    "media": {
        "benefit": "entertainment and discovery value",
        "ingredient": "content, edition, and format details",
        "usage": "reading, viewing, and listening occasions",
        "education": "edition differences and playback requirements",
        "visual": "recognizable cover art and edition presentation",
        "navigation": "genre, creator, and media format navigation",
    },
    "seasonal": {
        "benefit": "celebration and seasonal atmosphere benefits",
        "ingredient": "decorative materials and seasonal design details",
        "usage": "holiday traditions and event styling guidance",
        "education": "seasonal setup, storage, and reuse education",
        "visual": "festive scenes with coordinated seasonal themes",
        "navigation": "holiday, occasion, and decorating theme navigation",
    },
    "sports_outdoors": {
        "benefit": "activity performance and outdoor readiness benefits",
        "ingredient": "sporting materials and equipment construction",
        "usage": "training, recreation, and adventure guidance",
        "education": "equipment fit and activity suitability education",
        "visual": "active demonstrations in relevant outdoor settings",
        "navigation": "sport, terrain, and experience level navigation",
    },
    "home_improvement": {
        "benefit": "repair results and project efficiency benefits",
        "ingredient": "building materials and tool specifications",
        "usage": "installation, repair, and renovation guidance",
        "education": "measurement, surface compatibility, and setup education",
        "visual": "project steps and finished installation detail",
        "navigation": "project, trade, and material navigation",
    },
    "home": {
        "benefit": "everyday home comfort and convenience benefits",
        "ingredient": "household materials, textures, and finishes",
        "usage": "household routines and room styling guidance",
        "education": "home product sizing and care education",
        "visual": "welcoming rooms and everyday living details",
        "navigation": "room, household function, and decor style navigation",
    },
    "musical_instruments": {
        "benefit": "musical expression and playability benefits",
        "ingredient": "instrument materials and sound production details",
        "usage": "practice, rehearsal, and performance guidance",
        "education": "instrument selection, tuning, and care education",
        "visual": "musician interaction and instrument craftsmanship",
        "navigation": "instrument family and playing experience navigation",
    },
    "office_stationery": {
        "benefit": "organization and daily work efficiency benefits",
        "ingredient": "paper, ink, and office supply specifications",
        "usage": "writing, planning, and desk organization guidance",
        "education": "paper sizing and supply compatibility education",
        "visual": "organized workspaces and legible stationery details",
        "navigation": "office task, supply format, and pack size navigation",
    },
    "safety_emergency": {
        "benefit": "hazard awareness and preparedness benefits",
        "ingredient": "protective materials and verified rating details",
        "usage": "safe operation and emergency readiness guidance",
        "education": "intended protection, limitations, and inspection education",
        "visual": "clear safety demonstrations and readable instructions",
        "navigation": "hazard, protection type, and emergency need navigation",
    },
    "vehicle": {
        "benefit": "vehicle upkeep and driving convenience benefits",
        "ingredient": "part specifications and vehicle fitment details",
        "usage": "vehicle maintenance and accessory installation guidance",
        "education": "make, model, year, and part compatibility education",
        "visual": "installed parts and vehicle placement detail",
        "navigation": "vehicle fitment and automotive system navigation",
    },
    "arts_crafts": {
        "benefit": "creative possibilities and project outcome benefits",
        "ingredient": "craft materials, colors, and supply contents",
        "usage": "creative techniques and project inspiration",
        "education": "material selection and crafting skill education",
        "visual": "hands creating and achievable finished projects",
        "navigation": "craft technique, medium, and skill level navigation",
    },
    "baby": {
        "benefit": "caregiver convenience and baby comfort benefits",
        "ingredient": "baby product materials and care requirements",
        "usage": "feeding, changing, and caregiving routine guidance",
        "education": "age suitability and safe product use education",
        "visual": "natural caregiver moments and appropriate product use",
        "navigation": "baby stage and caregiving task navigation",
    },
    "fashion": {
        "benefit": "personal style and wearing comfort benefits",
        "ingredient": "fabric composition and garment construction",
        "usage": "outfit coordination and dressing occasion guidance",
        "education": "sizing, fit, and garment care education",
        "visual": "authentic outfit styling and fabric detail",
        "navigation": "apparel type, size, and personal style navigation",
    },
    "furniture": {
        "benefit": "seating comfort and living space functionality",
        "ingredient": "frame materials, upholstery, and furniture finishes",
        "usage": "room layout and furniture placement guidance",
        "education": "dimensions, assembly, and furniture care education",
        "visual": "furnished room context and construction close-ups",
        "navigation": "furniture type, room size, and finish navigation",
    },
    "health_general": {
        "benefit": "daily health management and support benefits",
        "ingredient": "health product components and specifications",
        "usage": "home health monitoring and care task guidance",
        "education": "intended use and health product limitations",
        "visual": "readable controls and clear care demonstrations",
        "navigation": "health support need and product function navigation",
    },
    "household_clean": {
        "benefit": "cleaning effectiveness and household order benefits",
        "ingredient": "cleaning formulas and surface compatibility",
        "usage": "cleaning tasks and storage routine guidance",
        "education": "dilution, handling, and surface care education",
        "visual": "clear cleaning demonstrations and organized storage",
        "navigation": "surface, cleaning task, and storage need navigation",
    },
    "toys": {
        "benefit": "play enjoyment and discovery benefits",
        "ingredient": "toy materials and included play pieces",
        "usage": "imaginative, shared, and independent play guidance",
        "education": "age suitability, play features, and setup education",
        "visual": "engaging play moments with visible toy interaction",
        "navigation": "age range, play interest, and toy type navigation",
    },
    "food_beverage": {
        "benefit": "benefit communication",
        "ingredient": "ingredient communication",
        "usage": "breakfast and snack storytelling",
        "education": "nutrition and usage education",
        "visual": "appetite-led visual identity",
        "navigation": "pantry shelf navigation",
    },
    "beauty": {
        "benefit": "benefit communication",
        "ingredient": "formula communication",
        "usage": "routine storytelling",
        "education": "regimen education",
        "visual": "cohesive beauty presentation",
        "navigation": "concern-based navigation",
    },
    "health_personal_care": {
        "benefit": "wellness benefit communication",
        "ingredient": "active support communication",
        "usage": "symptom and routine guidance",
        "education": "wellness shopper education",
        "visual": "clinical visual clarity",
        "navigation": "wellness shelf navigation",
    },
    "animals": {
        "benefit": "pet benefit communication",
        "ingredient": "nutrition communication",
        "usage": "feeding and care guidance",
        "education": "life-stage shopper education",
        "visual": "care-led visual identity",
        "navigation": "life-stage navigation",
    },
    "electronics": {
        "benefit": "performance communication",
        "ingredient": "compatibility detail",
        "usage": "setup and use-case guidance",
        "education": "device shopper education",
        "visual": "technical visual clarity",
        "navigation": "device-use navigation",
    }
    
    
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return normalize_product_type(_safe_text(value))


def _slug(value: Any) -> str:
    return _norm(value).replace(" ", "_")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}, []):
        return []
    return [value]


def _unique_text(values: list[Any], limit: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _safe_text(value)
        key = _norm(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _relative_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(_repo_root()))
    except ValueError:
        return str(path)


def _style_path(category_key: str) -> Path | None:
    filename = STYLE_GUIDE_FILES.get(resolve_category_key(category_key))
    if not filename:
        return None
    path = _repo_root() / "config" / "style_guides" / filename
    return path if path.exists() else None


def _image_path(category_key: str) -> Path | None:
    filename = IMAGE_GUIDE_FILES.get(resolve_category_key(category_key))
    if not filename:
        return None
    path = _repo_root() / "config" / "image_guides" / filename
    return path if path.exists() else None


def resolve_category_key(value: str) -> str:
    """Resolve explicit categories, then scan category keywords in order."""
    blob = _norm(value)
    if not blob:
        return ""
    # Exact aliases must not route free text using a single broad category word.
    compact = blob.replace(" ", "")
    if compact.endswith("json"):
        compact = compact[:-4]
    aliases = {
        "gardenandpatio": "garden_patio",
        "gardenpatio": "garden_patio",
        "gardenandpatioimg": "garden_patio",
        "gardenpatioimg": "garden_patio",
        "everythingelse": "everything_else",
        "everythingelseimg": "everything_else",
        "miscellaneous": "everything_else",
        "uncategorized": "everything_else",
        "businessandindustrial": "business_industrial",
        "businessindustrial": "business_industrial",
        "businessandindustrialimg": "business_industrial",
        "businessindustrialsupplies": "business_industrial",
        "foodandbeverage": "food_beverage",
        "foodbeverage": "food_beverage",
        "foodbeverageimg": "food_beverage",
        "foodandbeverages": "food_beverage",
        "grocery": "food_beverage",
        "groceries": "food_beverage",
        "beauty": "beauty",
        "beautyimg": "beauty",
        "beautypersonalcare": "beauty",
        "healthandpersonalcare": "health_personal_care",
        "healthpersonalcare": "health_personal_care",
        "healthpersonal": "health_personal_care",
        "healthpersonalimg": "health_personal_care",
        "healthandwellness": "health_personal_care",
        "animals": "animals",
        "animalsimg": "animals",
        "pet": "animals",
        "pets": "animals",
        "petsupplies": "animals",
        "electronics": "electronics",
        "electronicsandphotography": "electronics",
        "electronicsphotography": "electronics",
        "electronicsimg": "electronics",
        "photgraphyimg": "photography",
        "photography": "photography",
        "photographyimage": "photography",
        "photographyimg": "photography",
        "photographyimageguide": "photography",
        "photographicequipment": "photography",
        "media": "media",
        "mediaimg": "media",
        "booksandmedia": "media",
        "seasonal": "seasonal",
        "seasonalandoccasion": "seasonal",
        "seasonaloccasion": "seasonal",
        "seasonalimg": "seasonal",
        "seasonalandoccasions": "seasonal",
        "seasonaloccasions": "seasonal",
        "sportsandoutdoors": "sports_outdoors",
        "sportsoutdoors": "sports_outdoors",
        "sportsrecreationandoutdoor": "sports_outdoors",
        "sportsrecreationoutdoor": "sports_outdoors",
        "sportsrecreationoutdoors": "sports_outdoors",
        "sportsoutdoorsjson": "sports_outdoors",
        "sportsoutdoorsimg": "sports_outdoors",
        "sportsrecreationandoutdoors": "sports_outdoors",
        "homeimprovement": "home_improvement",
        "homeimprovementimg": "home_improvement",
        "homeimprovementsupplies": "home_improvement",
        "home": "home",
        "homeimageguide": "home",
        "homeimg": "home",
        "homegoods": "home",
        "musicalinstruments": "musical_instruments",
        "musicalinstrumentsimg": "musical_instruments",
        "musicalinstrument": "musical_instruments",
        "officeandstationery": "office_stationery",
        "officestationery": "office_stationery",
        "officestationeryimg": "office_stationery",
        "officestationary": "office_stationery",
        "officeandstationary": "office_stationery",
        "officeandstationaryimg": "office_stationery",
        "safetyandemergency": "safety_emergency",
        "safetyemergency": "safety_emergency",
        "safetyemergencyimg": "safety_emergency",
        "safetyandemergencysupplies": "safety_emergency",
        "vehicle": "vehicle",
        "vehicleimg": "vehicle",
        "vehicles": "vehicle",
        "vehiclepartsandaccessories": "vehicle",
        "vehiclespartsandaccessories": "vehicle",
        "vehiclespartsandaccessoriesimg": "vehicle",
        "artandcrafts": "arts_crafts",
        "artcrafts": "arts_crafts",
        "artsandcrafts": "arts_crafts",
        "artscrafts": "arts_crafts",
        "artscraftsimg": "arts_crafts",
        "artsandcraftsimg": "arts_crafts",
        "baby": "baby",
        "babyimg": "baby",
        "babyproducts": "baby",
        "babycare": "baby",
        "clothingandaccessories": "fashion",
        "apparelandaccessories": "fashion",
        "fashion": "fashion",
        "furniture": "furniture",
        "furnitureimg": "furniture",
        "homefurniture": "furniture",
        "generalhealth": "health_general",
        "healthgeneral": "health_general",
        "healthgeneralimg": "health_general",
        "householdcleaning": "household_clean",
        "householdindustrialcleaningandstorage": "household_clean",
        "householdindustrialcleaningstorage": "household_clean",
        "householdclean": "household_clean",
        "householdcleanimg": "household_clean",
        "householdcleaningandstorage": "household_clean",
        "toys": "toys",
        "toysimg": "toys",
        "toy": "toys",
        "toysandgames": "toys",
    }
    for key, filename in STYLE_GUIDE_FILES.items():
        for label in (key, Path(filename).stem, IMAGE_GUIDE_FILES.get(key, "")):
            normalized = _norm(Path(label).stem).replace(" ", "")
            if normalized:
                aliases[normalized] = key
                aliases[normalized.replace("and", "")] = key
        aliases[key.replace("_", "") + "img"] = key
    if compact in aliases:
        return aliases[compact]

    # Whole phrases keep the original membership checks from matching word fragments.
    words = blob.split()
    phrases = {
        " ".join(words[start:end])
        for start in range(len(words))
        for end in range(start + 1, min(len(words), start + 6) + 1)
    }
    blob = phrases | {phrase[:-1] for phrase in phrases if phrase.endswith("s")} | {
        phrase[:-2] for phrase in phrases if phrase.endswith("es")
    }
    # First matching category wins; keep specific categories before broad ones.
    if any(term in blob for term in ("baby", "infant", "diaper", "stroller", "pacifier", "baby bottle", "baby monitor", "baby food", "car seat", "crib", "bassinet", "teether", "changing pad", "nursing pillow", "baby carrier")):
        return "baby"
    if any(term in blob for term in ("pet", "dog", "cat", "animal", "pet food", "dog food", "cat litter", "dog toy", "pet carrier", "aquarium", "bird feeder", "pet grooming", "dog leash", "cat scratcher")):
        return "animals"
    if any(term in blob for term in ("safety", "emergency", "fire extinguisher", "smoke detector", "protective equipment", "first aid kit", "carbon monoxide detector", "fire blanket", "respirator", "emergency kit", "safety goggles", "hearing protection", "reflective vest", "escape ladder", "emergency radio")):
        return "safety_emergency"
    if any(term in blob for term in ("medical equipment", "mobility aid", "blood pressure monitor", "thermometer", "patient care", "wheelchair", "walker", "medical supplies", "pulse oximeter", "hospital bed", "crutches", "walking cane", "shower chair", "transfer bench", "medical bed")):
        return "health_general"
    if any(term in blob for term in ("health", "wellness", "personal care", "vitamin", "supplement", "symptom", "toothpaste", "toothbrush", "deodorant", "dental floss", "personal hygiene", "mouthwash", "sunscreen", "hand sanitizer", "shaving cream", "contact lens solution")):
        return "health_personal_care"
    if any(term in blob for term in ("photography", "camera", "lens", "tripod", "darkroom", "camera bag", "camera lens", "lens filter", "camera tripod", "photography lighting", "camera flash", "light meter", "camera strap", "photographic film", "camera battery grip")):
        return "photography"
    if any(term in blob for term in ("seasonal", "holiday", "christmas", "halloween", "party decoration", "christmas tree", "holiday lights", "halloween costume", "party decorations", "seasonal decor", "easter", "thanksgiving", "valentine", "ornament", "advent calendar")):
        return "seasonal"
    if any(term in blob for term in ("garden", "patio", "planter", "trellis", "greenhouse", "outdoor furniture", "patio furniture", "lawn mower", "garden hose", "potting soil", "raised garden bed", "watering can", "garden tools", "patio umbrella", "plant stand")):
        return "garden_patio"
    if any(term in blob for term in ("sport", "camping", "hiking", "fitness", "fishing", "camping tent", "sleeping bag", "yoga mat", "hiking boots", "fishing rod", "basketball", "soccer", "tennis", "backpack", "treadmill")):
        return "sports_outdoors"
    if any(term in blob for term in ("industrial", "commercial equipment", "manufacturing", "warehouse", "material handling", "pallet jack", "conveyor", "industrial equipment", "commercial machinery", "warehouse supplies", "forklift", "packaging machine", "workbench", "industrial pump", "shipping supplies")):
        return "business_industrial"
    if any(term in blob for term in ("craft", "yarn", "scrapbooking", "embroidery", "painting supplies", "craft kit", "sewing supplies", "knitting needles", "crochet hooks", "acrylic paint", "watercolor", "beading", "craft paper", "fabric paint", "pottery clay")):
        return "arts_crafts"
    if any(term in blob for term in ("toy", "doll", "puzzle", "playset", "board game", "building blocks", "action figure", "toy car", "doll house", "plush toy", "teddy bear", "remote control car", "play kitchen", "fidget toy", "toy train")):
        return "toys"
    if any(term in blob for term in ("renovation", "plumbing", "drill", "hardware", "power tool", "power tools", "paint roller", "door hardware", "plumbing supplies", "screwdriver", "wrench", "sander", "caulk", "wall anchor", "circuit breaker")):
        return "home_improvement"
    if any(term in blob for term in ("guitar", "piano", "violin", "drum", "instrument", "musical keyboard", "acoustic guitar", "drum kit", "trumpet", "saxophone", "ukulele", "clarinet", "cello", "trombone", "harmonica")):
        return "musical_instruments"
    if any(term in blob for term in ("stationery", "notebook", "binder", "stapler", "envelope", "office supplies", "printer paper", "ballpoint pen", "file folder", "office stationery", "highlighter", "index card", "sticky notes", "paper clip", "document organizer")):
        return "office_stationery"
    if any(term in blob for term in ("vehicle", "automotive", "car part", "motorcycle", "tire", "vehicle parts", "car accessories", "brake pads", "engine oil", "windshield wiper", "spark plug", "car battery", "vehicle floor mat", "steering wheel", "car cover")):
        return "vehicle"
    if any(term in blob for term in ("furniture", "sofa", "chair", "desk", "bookcase", "dining table", "office chair", "bed frame", "coffee table", "dresser", "nightstand", "ottoman", "recliner", "loveseat", "sideboard")):
        return "furniture"
    if any(term in blob for term in ("fashion", "apparel", "clothing", "footwear", "jewelry", "dress", "shirt", "sneaker", "handbag", "jacket", "trousers", "leggings", "sandal", "scarf", "blouse")):
        return "fashion"
    if any(term in blob for term in ("detergent", "disinfectant", "mop", "broom", "cleaning supplies", "laundry detergent", "dish soap", "surface cleaner", "trash bag", "cleaning wipes", "dishwasher tablets", "fabric softener", "scrub brush", "dustpan", "stain remover")):
        return "household_clean"
    if any(term in blob for term in ("bedding", "curtain", "kitchenware", "bath towel", "home decor", "cookware", "dinnerware", "bed sheets", "throw pillow", "window curtains", "bakeware", "cutlery", "tablecloth", "bath mat", "duvet")):
        return "home"
    if any(term in blob for term in ("book", "dvd", "blu ray", "audiobook", "vinyl record", "novel", "textbook", "comic book", "music album", "movie disc", "paperback", "hardcover", "manga", "music cd", "film collection")):
        return "media"
    if any(term in blob for term in ("food", "beverage", "pantry", "spread", "snack", "nutrition", "breakfast", "coffee", "tea", "cereal", "juice", "pasta", "granola", "chocolate", "rice", "soup", "sauce")):
        return "food_beverage"
    if any(term in blob for term in ("skin care", "hair care", "beauty", "makeup", "cosmetic", "shampoo", "conditioner", "lipstick", "moisturizer", "skin serum", "mascara", "eyeliner", "nail polish", "perfume", "face cleanser")):
        return "beauty"
    if any(term in blob for term in ("electronics", "device", "laptop", "phone", "audio", "smartphone", "tablet", "headphones", "computer monitor", "computer keyboard", "router", "television", "smartwatch", "printer", "usb hub")):
        return "electronics"
    if any(term in blob for term in ("everything else", "miscellaneous", "uncategorized", "novelty", "collectible", "memorabilia", "souvenir", "collectibles", "miscellaneous goods", "uncategorised", "keepsake", "commemorative item", "souvenir magnet", "collector item", "novelty gift")):
        return "everything_else"
    return ""


def load_style_title_guide(category_key: str) -> dict[str, Any]:
    """Load one existing style/title guide JSON."""
    path = _style_path(category_key)
    return _load_json(path) if path else {}


def load_image_story_guide(category_key: str) -> dict[str, Any]:
    """Load one existing image guide JSON through the canonical image helper."""
    return load_image_guide(resolve_category_key(category_key))


def normalize_style_title_guide(category_key: str) -> dict[str, Any]:
    """Normalize style/title guide families and product types into one shape."""
    resolved_key = resolve_category_key(category_key)
    guide = load_style_title_guide(resolved_key)
    path = _style_path(resolved_key)
    notes: list[str] = []
    families: dict[str, Any] = {}
    for family_key, family in (guide.get("families") or {}).items():
        if not isinstance(family, dict):
            continue
        product_types: dict[str, Any] = {}
        for product_key, product in (family.get("product_types") or {}).items():
            if not isinstance(product, dict):
                continue
            display = _safe_text(product.get("display_name")) or str(product_key).replace("_", " ").title()
            formula = _unique_text(_as_list(product.get("formula")))
            attributes = _unique_text(_as_list(product.get("attributes")))
            title_keywords = _unique_text(_as_list(product.get("title_keywords")))
            context_keywords = _unique_text(_as_list(product.get("context_keywords")))
            product_types[str(product_key)] = {
                "product_type_key": str(product_key),
                "product_type_display": display,
                "aliases": _unique_text(_as_list(product.get("aliases"))),
                "attribute_cues": attributes,
                "benefit_cues": _unique_text([*attributes, *context_keywords]),
                "usage_occasion_cues": context_keywords,
                "education_cues": _unique_text([*formula, *attributes]),
                "comparison_cues": formula,
                "recommended_title_priorities": formula,
                "title_keywords": title_keywords,
                "context_keywords": context_keywords,
                "negative_keywords": _unique_text(_as_list(product.get("negative_keywords"))),
                "raw": product,
            }
        families[str(family_key)] = {
            "family_key": str(family_key),
            "family_display": _safe_text(family.get("display_name")) or str(family_key).replace("_", " ").title(),
            "product_types": product_types,
        }
    if not guide:
        notes.append("style guide was not available or could not be parsed")
    return {
        "category_key": resolved_key,
        "category_display": _safe_text(guide.get("category")) or resolved_key.replace("_", " ").title(),
        "families": families,
        "source_guides_used": [_relative_path(path)] if path else [],
        "normalization_notes": notes,
    }


def normalize_image_story_guide(category_key: str) -> dict[str, Any]:
    """Normalize image guide pages into visual/story priorities."""
    resolved_key = resolve_category_key(category_key)
    guide = load_image_story_guide(resolved_key)
    path = _image_path(resolved_key)
    notes: list[str] = []
    pages: dict[str, Any] = {}
    slot_definitions = guide.get("slot_definitions") or {}
    for page_key, page in (guide.get("pages") or {}).items():
        if not isinstance(page, dict):
            continue
        required_slots = _unique_text(_as_list(page.get("required_slots")), limit=20)
        visual_priorities: list[str] = []
        story_cues: list[str] = []
        for slot in required_slots:
            slot_data = slot_definitions.get(slot) if isinstance(slot_definitions, dict) else {}
            label = _safe_text((slot_data or {}).get("label")) or slot.replace("_", " ").title()
            guidance = _safe_text((slot_data or {}).get("guidance") or (slot_data or {}).get("description"))
            visual_priorities.append(label)
            if guidance:
                story_cues.append(guidance)
        pages[str(page_key)] = {
            "page_key": str(page_key),
            "page_display": _safe_text(page.get("display_name")) or str(page_key).replace("_", " ").title(),
            "product_types": _unique_text(_as_list(page.get("product_types")), limit=20),
            "required_slots": required_slots,
            "image_story_cues": _unique_text(story_cues, limit=10),
            "recommended_visual_priorities": _unique_text(
                [*visual_priorities, *_as_list(page.get("additional_recommendations"))],
                limit=12,
            ),
            "module_story_cues": _unique_text(_as_list(page.get("additional_recommendations")), limit=8),
        }
    if not guide or not pages:
        notes.append("image guide was not available or did not expose normalized pages")
    return {
        "category_key": resolved_key,
        "category_display": _safe_text(guide.get("category")) or resolved_key.replace("_", " ").title(),
        "guide_key": _safe_text(guide.get("guide_key")),
        "pages": pages,
        "product_type_index": guide.get("product_type_index") or {},
        "source_guides_used": [_relative_path(path)] if path else [],
        "normalization_notes": notes,
    }


def load_normalized_guides(category_key: str) -> dict[str, Any]:
    """Load and combine normalized style/title and image guide data."""
    resolved_key = resolve_category_key(category_key)
    style = normalize_style_title_guide(resolved_key)
    image = normalize_image_story_guide(resolved_key)
    return {
        "category_key": resolved_key,
        "category_display": style.get("category_display") or image.get("category_display") or resolved_key.replace("_", " ").title(),
        "style": style,
        "image": image,
        "source_guides_used": _unique_text(
            [*(style.get("source_guides_used") or []), *(image.get("source_guides_used") or [])],
            limit=8,
        ),
        "normalization_notes": [
            *(style.get("normalization_notes") or []),
            *(image.get("normalization_notes") or []),
        ],
    }


def _record_value(record: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        current: Any = record
        found = True
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if found and current not in (None, "", [], {}):
            return current
    return default


def _record_text(record: dict[str, Any]) -> str:
    parts = [
        _record_value(record, "product_title", "title", "productTitle"),
        _record_value(record, "description_body", "description", "shortDescription"),
        _record_value(record, "category", "categoryPathName"),
        _record_value(record, "product_type", "subcategory", "productType"),
        _record_value(record, "brand", "brandName"),
        *_as_list(_record_value(record, "description_bullets", "Description Bullets", default=[])),
        *_as_list(_record_value(record, "key_features", "Key Features", default=[])),
        *_as_list(record.get("categoryNavigation") if isinstance(record, dict) else []),
    ]
    return " ".join(_safe_text(part) for part in parts if _safe_text(part)).lower()


def _match_style_product_type(
    normalized_style: dict[str, Any],
    product_type: str,
    text_blob: str,
) -> tuple[str, str, dict[str, Any], dict[str, Any], int]:
    best = ("", product_type, {}, {}, 0)
    wanted = _norm(product_type)
    for family_key, family in (normalized_style.get("families") or {}).items():
        for product_key, product in (family.get("product_types") or {}).items():
            score = 0
            display = _safe_text(product.get("product_type_display")) or str(product_key).replace("_", " ").title()
            terms = [
                display,
                *_as_list(product.get("aliases")),
                *_as_list(product.get("title_keywords")),
                *_as_list(product.get("context_keywords")),
            ]
            normalized_terms = [_norm(term) for term in terms if _safe_text(term)]
            if wanted and (wanted == _norm(display) or wanted in normalized_terms):
                score += 10
            for term in normalized_terms:
                if term and term in text_blob:
                    score += 2
            for term in [_norm(value) for value in _as_list(product.get("negative_keywords"))]:
                if term and term in text_blob:
                    score -= 6
            if score > best[4]:
                best = (str(family_key), display, family, product, score)
    return best


def _clean_phrase(value: str, fallback: str) -> str:
    text = _safe_text(value).replace("/", " ")
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text or fallback


def build_identity_phrases(
    *,
    category_display: str,
    family_display: str,
    product_type_display: str,
) -> dict[str, str]:
    """Build display-safe phrase variants for later slide/cue prompts."""
    category = _clean_phrase(category_display, "category")
    family = _clean_phrase(family_display, "")
    product = _clean_phrase(product_type_display, family or category)
    if family and _norm(family) != _norm(product):
        combined = f"the {family} and {product} category"
        combined_alt = f"the {product} segment within {category}"
    else:
        combined = f"the {product} category"
        combined_alt = f"the {product} space within {category}"
    return {
        "combined_category_phrase": combined,
        "combined_category_phrase_alt": combined_alt,
        "shopping_context_phrase": f"{product} shopping journey",
        "product_type_focus_phrase": f"{product} positioning",
    }


def resolve_strategic_identity(
    records: list[dict[str, Any]],
    *,
    fallback_category: str = "",
    fallback_product_type: str = "",
) -> dict[str, Any]:
    """Resolve category/family/product type plus normalized guide context."""
    usable = [record for record in records or [] if isinstance(record, dict)]
    category = next(
        (
            _safe_text(_record_value(record, "category", "categoryPathName"))
            for record in usable
            if _safe_text(_record_value(record, "category", "categoryPathName"))
        ),
        fallback_category,
    )
    product_type = next(
        (
            _safe_text(_record_value(record, "product_type", "subcategory", "productType"))
            for record in usable
            if _safe_text(_record_value(record, "product_type", "subcategory", "productType"))
        ),
        fallback_product_type or category or "category",
    )
    text_blob = " ".join(_record_text(record) for record in usable)
    category_key = resolve_category_key(" ".join([category, product_type, text_blob]))
    guides = load_normalized_guides(category_key or category or product_type)
    family_key, product_display, family_data, product_data, score = _match_style_product_type(
        guides.get("style", {}),
        product_type,
        text_blob,
    )
    product_display = product_display or product_type or category or "category"
    family_display = _safe_text(family_data.get("family_display")) if family_data else ""
    category_display = guides.get("category_display") or _clean_phrase(category, category_key.replace("_", " ").title())

    image_page = get_image_guide_page(category_key, product_display) or get_image_guide_page(category_key, product_type) or {}
    normalized_image_page = {}
    if image_page:
        normalized_image_page = (guides.get("image", {}).get("pages") or {}).get(image_page.get("page_key"), {})

    phrases = build_identity_phrases(
        category_display=category_display,
        family_display=family_display,
        product_type_display=product_display,
    )
    vocab = CATEGORY_VOCABULARY.get(category_key, {})
    source_guides = guides.get("source_guides_used") or []
    notes = list(guides.get("normalization_notes") or [])
    if not family_key:
        notes.append("style product type was resolved from evidence fallback")

    return {
        "category_key": category_key,
        "category_display": category_display,
        "family_key": family_key,
        "family_display": family_display,
        "product_type_key": _slug(product_display),
        "product_type_display": product_display,
        **phrases,
        "attribute_cues": _unique_text(_as_list(product_data.get("attribute_cues")), limit=10),
        "benefit_cues": _unique_text([vocab.get("benefit", ""), *_as_list(product_data.get("benefit_cues"))], limit=10),
        "usage_occasion_cues": _unique_text([vocab.get("usage", ""), *_as_list(product_data.get("usage_occasion_cues"))], limit=10),
        "education_cues": _unique_text([vocab.get("education", ""), *_as_list(product_data.get("education_cues"))], limit=10),
        "comparison_cues": _unique_text(_as_list(product_data.get("comparison_cues")), limit=10),
        "image_story_cues": _unique_text(_as_list(normalized_image_page.get("image_story_cues")), limit=10),
        "module_story_cues": _unique_text(_as_list(normalized_image_page.get("module_story_cues")), limit=10),
        "recommended_visual_priorities": _unique_text(_as_list(normalized_image_page.get("recommended_visual_priorities")), limit=12),
        "recommended_title_priorities": _unique_text(_as_list(product_data.get("recommended_title_priorities")), limit=12),
        "source_guides_used": source_guides,
        "normalization_notes": notes,
        "style_guide_path": next((path for path in source_guides if "style_guides" in path), ""),
        "image_guide_path": next((path for path in source_guides if "image_guides" in path), ""),
        "style_product_type_score": score,
        "style_product_type": product_data.get("raw") or product_data,
        "image_page": image_page,
        "normalized_guides": guides,
    }


def identity_debug_payload(identity: dict[str, Any]) -> dict[str, Any]:
    """Return serializable resolver metadata for export debug structures."""
    hidden = {"style_product_type", "image_page", "normalized_guides"}
    return {key: value for key, value in identity.items() if key not in hidden}
