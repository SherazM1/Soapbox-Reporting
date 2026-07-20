from dataclasses import asdict, dataclass
from typing import Any, Literal


JobType = Literal["Apparel", "Misc"]
LaydownSiloType = Literal["shoes", "else/default"]
ModelType = Literal["kid", "adult"]


@dataclass(frozen=True)
class ApparelInputs:
    on_model_image_quantity: int = 0
    on_model_detail_quantity: int = 0
    laydown_silo_type: LaydownSiloType = "else/default"
    laydown_silo_quantity: int = 0
    color_corrections_quantity: int = 0
    post_production_hours: float = 0.0
    model_type: ModelType = "adult"
    model_hours: float = 0.0
    model_fitting_enabled: bool = False
    ai_generation_quantity: int = 0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuoteLine:
    code: str
    label: str
    quantity: float
    unit_price: float
    total: float

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuotePayload:
    selected_job_type: JobType
    apparel_inputs: ApparelInputs
    derived_total_image_count: int
    derived_account_management_fee: float
    line_items: tuple[QuoteLine, ...]
    subtotal: float
    total: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "selected_job_type": self.selected_job_type,
            "apparel_inputs": self.apparel_inputs.to_payload(),
            "derived_total_image_count": self.derived_total_image_count,
            "derived_account_management_fee": self.derived_account_management_fee,
            "line_items": [line.to_payload() for line in self.line_items],
            "subtotal": self.subtotal,
            "total": self.total,
        }
