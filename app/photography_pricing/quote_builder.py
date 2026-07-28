from app.photography_pricing.models import ApparelInputs, QuoteLine, QuotePayload
from app.photography_pricing.pricing_rules import (
    AI_GENERATION_MARKUP_RATE,
    COLOR_CORRECTIONS_RATE,
    MODEL_ADULT_HOURLY_RATE,
    MODEL_FITTING_FLAT_FEE,
    MODEL_KID_HOURLY_RATE,
    ON_MODEL_DETAIL_RATE,
    ON_MODEL_IMAGE_RATE,
    POST_PRODUCTION_HOURLY_RATE,
    account_management_fee,
    laydown_silo_rate,
)


def _line(code: str, label: str, quantity: float, unit_price: float) -> QuoteLine:
    return QuoteLine(
        code=code,
        label=label,
        quantity=quantity,
        unit_price=unit_price,
        total=round(quantity * unit_price, 2),
    )


def image_count_for_account_management(inputs: ApparelInputs) -> int:
    return (
        inputs.on_model_image_quantity
        + inputs.on_model_detail_quantity
        + inputs.laydown_silo_quantity
        + inputs.color_corrections_quantity
        + inputs.ai_generation_quantity
    )


def _model_hour_lines(inputs: ApparelInputs) -> tuple[QuoteLine, ...]:
    if inputs.model_type == "both":
        return (
            _line(
                "adult_model_hours",
                "Adult model hours",
                inputs.adult_model_hours,
                MODEL_ADULT_HOURLY_RATE,
            ),
            _line(
                "kid_model_hours",
                "Kid model hours",
                inputs.kid_model_hours,
                MODEL_KID_HOURLY_RATE,
            ),
        )
    if inputs.model_type == "kid":
        quantity = inputs.kid_model_hours if inputs.kid_model_hours else inputs.model_hours
        return (_line("model_hours", "Model hours", quantity, MODEL_KID_HOURLY_RATE),)
    quantity = inputs.adult_model_hours if inputs.adult_model_hours else inputs.model_hours
    return (_line("model_hours", "Model hours", quantity, MODEL_ADULT_HOURLY_RATE),)


def _model_fitting_quantity(inputs: ApparelInputs) -> int:
    if inputs.model_fitting_quantity:
        return max(int(inputs.model_fitting_quantity), 0)
    return 1 if inputs.model_fitting_enabled else 0


def _manual_account_management_amount(inputs: ApparelInputs) -> float:
    return round(max(float(inputs.manual_account_management_amount or 0), 0), 2)


def _account_management_amount(inputs: ApparelInputs, automatic_fee: float) -> float:
    if inputs.account_management_mode == "manual":
        return _manual_account_management_amount(inputs)
    return automatic_fee


def build_apparel_quote(inputs: ApparelInputs) -> QuotePayload:
    total_image_count = image_count_for_account_management(inputs)
    account_fee = account_management_fee(total_image_count)
    account_amount_used = _account_management_amount(inputs, account_fee)

    line_items = (
        _line("on_model_image", "On-model image", inputs.on_model_image_quantity, ON_MODEL_IMAGE_RATE),
        _line("on_model_detail", "On-model detail", inputs.on_model_detail_quantity, ON_MODEL_DETAIL_RATE),
        _line(
            "laydown_silo",
            "Laydown silo",
            inputs.laydown_silo_quantity,
            laydown_silo_rate(inputs.laydown_silo_type),
        ),
        _line(
            "color_corrections",
            "Color corrections from existing images",
            inputs.color_corrections_quantity,
            COLOR_CORRECTIONS_RATE,
        ),
        _line(
            "post_production",
            "Post production hourly time",
            inputs.post_production_hours,
            POST_PRODUCTION_HOURLY_RATE,
        ),
        *_model_hour_lines(inputs),
        _line("model_fitting", "Model fitting", _model_fitting_quantity(inputs), MODEL_FITTING_FLAT_FEE),
        _line("ai_generation", "AI generation markup", inputs.ai_generation_quantity, AI_GENERATION_MARKUP_RATE),
        _line("account_management", "Account management", 1, account_amount_used),
    )
    subtotal = round(sum(line.total for line in line_items), 2)

    return QuotePayload(
        selected_job_type="Apparel",
        apparel_inputs=inputs,
        derived_total_image_count=total_image_count,
        derived_account_management_fee=account_fee,
        account_management_mode=inputs.account_management_mode,
        manual_account_management_amount=_manual_account_management_amount(inputs),
        account_management_amount_used=account_amount_used,
        line_items=line_items,
        subtotal=subtotal,
        total=subtotal,
    )
