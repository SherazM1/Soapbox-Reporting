from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeliverableInputs:
    pdp_count: int = 0
    brief_count: int = 0
    ad_count: int = 0


@dataclass(frozen=True)
class CreativePricingInputs:
    deliverables: DeliverableInputs = field(default_factory=DeliverableInputs)
    copy_provided: str = "Yes"
    copy_scopes: tuple[str, ...] = ()
    assets_provided: str = "Yes"
    provided_assets: tuple[str, ...] = ()
    missing_assets: tuple[str, ...] = ()
    revision_mode: str = "Fixed"
    fixed_revision_tier: str = "Simple"
    manual_revision_amount: float = 0.0
    internal_work: tuple[str, ...] = ()
    other_internal_work: str = ""
    other_internal_amount: float = 0.0
    tool_platforms: tuple[str, ...] = ()
    other_tool_platform: str = ""
    market_tier: str = "Local"

