ON_MODEL_IMAGE_RATE = 240.00
ON_MODEL_DETAIL_RATE = 145.00
LAYDOWN_SILO_SHOES_RATE = 75.00
LAYDOWN_SILO_DEFAULT_RATE = 100.00
COLOR_CORRECTIONS_RATE = 45.00
POST_PRODUCTION_HOURLY_RATE = 175.00
MODEL_KID_HOURLY_RATE = 105.00
MODEL_ADULT_HOURLY_RATE = 230.00
MODEL_FITTING_FLAT_FEE = 50.00
AI_GENERATION_MARKUP_RATE = 150.00

ACCOUNT_MANAGEMENT_UNDER_35_FEE = 175.00
ACCOUNT_MANAGEMENT_35_TO_64_FEE = 350.00
ACCOUNT_MANAGEMENT_65_PLUS_FEE = 750.00


def laydown_silo_rate(selection: str) -> float:
    if selection == "shoes":
        return LAYDOWN_SILO_SHOES_RATE
    return LAYDOWN_SILO_DEFAULT_RATE


def model_hourly_rate(selection: str) -> float:
    if selection == "kid":
        return MODEL_KID_HOURLY_RATE
    return MODEL_ADULT_HOURLY_RATE


def account_management_fee(total_image_count: int) -> float:
    if total_image_count >= 65:
        return ACCOUNT_MANAGEMENT_65_PLUS_FEE
    if total_image_count >= 35:
        return ACCOUNT_MANAGEMENT_35_TO_64_FEE
    return ACCOUNT_MANAGEMENT_UNDER_35_FEE


def account_management_tier_label(total_image_count: int) -> str:
    if total_image_count >= 65:
        return "65+ images"
    if total_image_count >= 35:
        return "35 to 64 images"
    return "Less than 35 images"
