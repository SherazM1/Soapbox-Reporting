"""Plain-language presentation copy; findings stay high-level and evidence-qualified."""
from __future__ import annotations

import re


# Fix note: Explain shopper impact without exposing a detailed implementation plan.
def simplify_deck_copy(text: str) -> str:
    if not text:
        return text
    if "educate shoppers through benefit-driven merchandising" in text.lower():
        return "Clear product pages help shoppers understand the product and decide whether it meets their needs."
    if "translating that brand equity" in text.lower():
        return "The opportunity is to make products easier to find, understand, and choose on Walmart."
    # Fix note: Retailer novelty/gag taxonomy is unnecessary in shopper-facing toy copy.
    phrases = {
        "sensory toys and novelty and gag toys": "sensory toys",
        "sensory toys and novelty & gag toys": "sensory toys",
        "novelty and gag toys": "toys",
        "novelty & gag toys": "toys",
        "novelty and gag toy": "toy",
        "novelty & gag toy": "toy",
        "Walmart eCommerce Opportunity": "Walmart Online Opportunity",
        "Search & Discoverability Benchmarking": "Product Search Comparison",
        "PDP Content Benchmarking": "Product Page Comparison",
        "Brand Shop Content Benchmarking": "Brand Shop Comparison",
        "Digital Shelf Ownership": "Online Shopping Presence",
        "Cohesive PDP visual education": "Images tell a consistent product story",
        "Structured shopper education": "Clear product information",
        "conversion-focused educational content": "information that helps shoppers choose",
        "conversion-focused guidance": "guidance that helps shoppers choose",
        "conversion performance": "purchase decisions",
        "omnichannel": "online and in-store",
        "digital shelf ownership": "online shopping presence",
        "category discoverability": "how easily shoppers find products",
        "discoverability": "product visibility",
        "purchase friction": "shopping uncertainty",
        "proof points": "supporting details",
        "regimen": "care routine",
        "merchandising": "product presentation",
        "PDPs": "product pages",
        "PDP": "product page",
        "support conversion": "help shoppers decide",
        "improve conversion": "help shoppers decide",
        "benefit-forward": "benefit-focused",
        "lifestyle-led": "everyday-use",
        "use-case": "product-use",
        "assortment": "product range",
    }
    result = text
    for original, replacement in phrases.items():
        result = re.sub(r"(?<!\w)" + re.escape(original) + r"(?!\w)",
                        lambda match: replacement, result, flags=re.I)
    return result
