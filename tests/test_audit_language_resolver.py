from __future__ import annotations

import unittest

from app.audit_helpers.audit_language_resolver import (
    cue_language_label,
    language_profile,
    recommendation_phrase,
    strategic_bullet_text,
)


class AuditLanguageResolverTests(unittest.TestCase):
    def test_health_groups_and_specific_product_matching(self) -> None:
        from app.audit_helpers.audit_language_resolver import PRODUCT_LANGUAGE_OVERLAYS, _overlay_values

        self.assertGreaterEqual(len(PRODUCT_LANGUAGE_OVERLAYS["health_personal_care"]), 42)
        cases = {
            "Herbal Supplements": "supplement",
            "Toothpaste": "oral care",
            "Ibuprofen": "pain relief",
            "Antihistamines": "cold cough and allergy",
            "Fiber Supplement": "digestive health",
            "Bandages": "first aid and wound care",
            "Menstrual Pads": "feminine care",
            "Men’s Grooming": "mens personal care",
            "Roll-on Deodorant": "deodorant and antiperspirant",
            "Shaving Cream": "shaving and hair removal",
            "Incontinence Pads": "incontinence and adult care",
            "Pregnancy Tests": "sexual wellness",
            "Insoles": "foot care",
            "Contact-Lens Care": "eye and ear care",
            "Pulse Oximeters": "medical monitoring and devices",
            "Melatonin Gummies": "sleep and stress support",
            "Nutrition Shakes": "weight management and nutrition",
            "Hand Sanitizer": "personal hygiene and cleansing",
        }
        for product, context in cases.items():
            with self.subTest(product=product):
                values = _overlay_values({
                    "category_key": "health_personal_care", "product_type_display": product,
                    "family_display": "Vitamins & Supplements",
                })
                self.assertEqual(values["product_context"], context + " shelf")

    def test_specific_health_sections_override_broad_family_language(self) -> None:
        cases = {
            "Denture Adhesive": "denture care",
            "Retainer Cleaner": "orthodontic care",
            "Electric Toothbrush": "powered oral care",
            "Saline Spray": "nasal care",
            "Earplugs": "hearing protection",
            "Hearing Aid Batteries": "hearing aid care",
            "Glucose Test Strips": "diabetes supplies",
            "Walking Cane": "mobility aids",
            "Shower Chair": "bathroom safety",
            "Knee Brace": "braces and supports",
            "Compression Socks": "compression wear",
            "Heating Pad": "hot and cold therapy",
            "Massage Gun": "massage tools",
            "CPAP Mask": "respiratory device supplies",
            "Home Test Kit": "home health tests",
            "Pill Organizer": "medication organization",
            "Sunscreen": "sun protection",
            "Petroleum Jelly": "skin barrier care",
            "Body Wash": "body cleansing",
            "Hand Cream": "hand care",
            "Nail Clippers": "nail care tools",
            "Intimate Wipes": "intimate cleansing",
            "Nicotine Gum": "smoking cessation",
            "Electrolyte Tablets": "hydration support",
        }
        for product, context in cases.items():
            with self.subTest(product=product):
                identity = {
                    "category_key": "health_personal_care",
                    "product_type_display": product,
                    "family_display": "Vitamins & Supplements",
                }
                profile = language_profile(identity)
                self.assertEqual(profile["product_context"], context + " shelf")
                for key in ("navigation", "discovery", "search_intent", "trust"):
                    self.assertIn(context, profile[key])

    def test_common_health_guide_names_resolve_to_relevant_sections(self) -> None:
        cases = {
            "Body Moisturizers": "body moisturizing",
            "Bath Minerals & Salts": "bath and spa products",
            "Cotton Pads": "cotton and cleansing accessories",
            "Body Care Sets": "personal care sets",
            "Teeth Whitening": "teeth whitening",
            "Dental Protectors & Nightguards": "dental guards",
            "Hair Trimmers & Clippers": "electric shaving and trimming",
            "Aftershaves": "aftershave care",
            "Hair Removal Waxes & Waxing Kits": "waxing and epilation",
            "Sleep Masks": "sleep accessories",
            "Reading Glasses": "eyewear and vision accessories",
            "Hearing Aids": "hearing devices",
            "Sock Dressing Aids": "daily living accessories",
            "Wheelchair Cushions": "mobility accessories",
            "Itching & Rash Treatments": "skin treatments",
            "Lice Treatment Kits": "lice care",
            "Disposable Gloves": "protective care supplies",
            "Facial Self-Tanners": "tanning products",
            "Physical Therapy Putty": "physical therapy accessories",
            "Kegel Exercisers": "intimate wellness devices",
            "Powered Toothbrush Replacement Heads": "powered oral care",
            "Blood Glucose Monitors": "diabetes supplies",
            "Sanitary Napkins": "feminine care",
            "Incontinence Underpads & Protectors": "incontinence and adult care",
            "COVID-19 Test Kits": "home health tests",
            "Nutrition Drinks": "weight management and nutrition",
            "Electrolyte Supplements": "hydration support",
            "CPAP Machines": "respiratory device supplies",
            "Shower & Bath Stools & Benches": "bathroom safety",
            "Beard Balms": "mens personal care",
        }
        for product, context in cases.items():
            with self.subTest(product=product):
                profile = language_profile({
                    "category_key": "health_personal_care",
                    "product_type_display": product,
                    "family_display": "Supplements",
                })
                self.assertEqual(profile["product_context"], context + " shelf")
                self.assertTrue(profile["formula"])
                self.assertTrue(profile["education"])

    def test_beauty_groups_and_specific_product_matching(self) -> None:
        from app.audit_helpers.audit_language_resolver import PRODUCT_LANGUAGE_OVERLAYS, _overlay_values

        self.assertGreaterEqual(len(PRODUCT_LANGUAGE_OVERLAYS["beauty"]), 48)
        cases = {
            "Cleansing Foam": "facial cleanser",
            "Face Cream": "moisturizer",
            "Vitamin C Serum": "serums and treatments",
            "Acne Patches": "acne and blemish care",
            "Facial Mist": "toners and essences",
            "Chemical Exfoliants": "masks and exfoliation",
            "Eye Cream": "eye and lip care",
            "Body Lotion": "body care",
            "Body Wash": "bath and shower",
            "Co-Wash": "hair cleansing and conditioning",
            "Hair Serum": "hair styling and treatment",
            "Face Primer": "face makeup",
            "Eyeliner": "eye and brow makeup",
            "Lip Liner": "lip makeup",
            "Body Spray": "fragrance and deodorant",
            "Makeup Brushes": "beauty tools and accessories",
        }
        for product, context in cases.items():
            with self.subTest(product=product):
                values = _overlay_values({
                    "category_key": "beauty", "product_type_display": product,
                    "family_display": "Serums & Treatments",
                })
                self.assertEqual(values["product_context"], context + " shelf")
        self.assertEqual(_overlay_values({"category_key": "beauty", "product_type_display": "glossary"}), {})

    def test_beauty_guide_names_and_specific_product_precedence(self) -> None:
        cases = {
            "Makeup Removers": "makeup removal",
            "Makeup Setting Sprays": "makeup setting and blotting",
            "Makeup Palettes": "makeup palettes and sets",
            "Skin Care Sets": "skin care sets",
            "False Eyelashes": "false lashes",
            "False Eyelash Adhesives": "lash adhesives and applicators",
            "Eyelash Curler Pads": "lash and brow tools",
            "Makeup Brush Cleansers": "brush cleaning and tool care",
            "Cosmetic Pencil Sharpeners": "cosmetic sharpeners",
            "Face Mirrors": "beauty mirrors",
            "Makeup Organizers": "beauty storage",
            "Temporary Tattoos": "body makeup and art",
            "Tattoo & Piercing Aftercare": "tattoo and piercing aftercare",
            "Dry Shampoos": "dry shampoo",
            "Hair Loss Treatments": "hair loss treatments",
            "Hair Color Touch-Up Sticks": "hair color",
            "Hair Relaxers & Straighteners": "hair bleach and chemical treatments",
            "Hair Color Applicators": "hair coloring supplies",
            "Hair Dryer Diffusers": "hair dryers and diffusers",
            "Hair Straightening Brushes": "heated hair styling tools",
            "Hair Brushes": "hair brushes combs and rollers",
            "Hair Barrettes, Pins & Clips": "hair accessories",
            "Wig Liquid Adhesives": "hair extensions and wig supplies",
            "Nail Polish": "nail color and finishes",
            "Nail Polish Removers": "nail polish removal",
            "False Nails": "false nails and adhesives",
            "Nail Art Brushes & Pens": "nail art",
            "Cuticle Creams & Oils": "cuticle and nail treatments",
            "Nail Clippers": "manicure hand tools",
            "Manicure Drill Bits": "powered nail tools",
            "Facial Cleansing Brush Heads": "facial cleansing tools",
            "Acne Clearing Devices": "skin treatment devices",
            "Tinted Moisturizers": "face makeup",
            "Eye Primer": "eye and brow makeup",
            "Eye Creams & Serums": "eye and lip care",
            "Facial Toners & Astringents": "toners and essences",
            "Hair Styling Mousses": "hair styling and treatment",
            "Colognes": "fragrance and deodorant",
        }
        for product, context in cases.items():
            with self.subTest(product=product):
                identity = {
                    "category_key": "beauty",
                    "product_type_display": product,
                    "family_display": "Serums & Treatments",
                }
                profile = language_profile(identity)
                self.assertEqual(profile["product_context"], context + " shelf")
                self.assertIn(context, profile["navigation"])
                self.assertIn(context, profile["trust"])
                identity["product_type_key"] = product.lower().replace(" ", "_")
                identity["product_type_display"] = "Unspecified product"
                self.assertEqual(language_profile(identity)["product_context"], context + " shelf")

    def test_electronics_guide_names_and_specific_product_precedence(self) -> None:
        from app.audit_helpers.audit_language_resolver import _overlay_values

        cases = {
            "Computer Monitors": "computer monitors",
            "Computer Keyboard & Mouse Sets": "keyboards and mice",
            "Webcams": "webcams and conferencing",
            "Laptop Cooling Pads": "computer desk accessories",
            "Laptop Docking Stations": "docks and hubs",
            "USB Wi-Fi Adapters": "network adapters and cabling",
            "Motherboard & CPU Combos": "processors and motherboards",
            "Computer Video Cards": "graphics and expansion cards",
            "RAM Memory": "computer memory",
            "Computer Cases": "computer cases and cooling",
            "Computer Power Supplies": "computer power supplies",
            "Laptop Replacement Screens": "computer replacement parts",
            "Memory Card Readers": "storage readers and adapters",
            "Laptop Charger": "charging adapters",
            "Uninterruptible Power Supplies": "power protection",
            "HDMI Cable": "cables and connectors",
            "Cable Organizers": "cable organization",
            "Cell Phone Cases": "device cases and screen protection",
            "Tablet Computer Stands": "device stands and grips",
            "TV & Monitor Mounts": "tv and speaker mounts",
            "TV Antennas": "remote controls and tv reception",
            "Sound Bars": "sound bars and home speakers",
            "Audio & Video Receivers": "receivers and audio components",
            "Turntable Cartridge Styli": "turntables and accessories",
            "Earbud Tips": "headphone accessories",
            "Mini Projectors": "projectors",
            "Projector Replacement Lamps": "projector screens and accessories",
            "Video Capture Devices": "video playback and capture",
            "Video Game Controllers": "gaming controllers and accessories",
            "Game Controller": "gaming controllers and accessories",
            "VR Headsets": "virtual reality",
            "Graphics Tablets": "graphics tablets and pens",
            "3D Printers": "3d printers",
            "Bluetooth Tracker": "gps and tracking",
            "Cordless Phones": "phones and communication",
            "SIM Cards": "sim cards and phone service",
            "Two-Way Radios": "radio communication",
            "Weather Stations": "security and monitoring electronics",
            "Compressed Air Dusters": "electronics cleaning",
            "Binoculars": "optics",
            "Telescope Eyepieces": "optics accessories",
            "Bluetooth Speakers": "audio device",
            "Laptop Computers": "computer and tablet",
            "Streaming Media Players": "streaming media device",
            "Smart Watches": "wearable technology",
        }
        for product, context in cases.items():
            with self.subTest(product=product):
                identity = {
                    "category_key": "electronics",
                    "product_type_display": product,
                    "family_display": "Computers",
                }
                profile = language_profile(identity)
                self.assertEqual(profile["product_context"], context + " shelf")
                self.assertIn(context, profile["navigation"])
                self.assertIn(context, profile["trust"])
                identity["product_type_key"] = product.lower().replace(" ", "_")
                identity["product_type_display"] = "Unspecified product"
                self.assertEqual(language_profile(identity)["product_context"], context + " shelf")
        for unrelated in ("showcase", "cableway", "rambling"):
            self.assertEqual(_overlay_values({
                "category_key": "electronics", "product_type_display": unrelated,
            }), {})

    def test_business_guide_names_and_specific_product_precedence(self) -> None:
        from app.audit_helpers.audit_language_resolver import _overlay_values

        cases = {
            "Drum & Pail Lids": "drum closures and tools",
            "Drum & Pail Mixers": "drum mixing equipment",
            "Solar Panels": "solar power components",
            "Wind Turbine Motors": "wind energy equipment",
            "Motor Contactors": "motor controls",
            "Printed Circuit Boards": "semiconductors and circuit boards",
            "Air Cylinder Sensors": "industrial sensors and interfaces",
            "Retail Display Cases": "retail display fixtures",
            "Mannequins": "mannequins and jewelry displays",
            "Shopping Carts": "shopping carts and queue barriers",
            "Take-Out Containers": "foodservice packaging",
            "Poly Bag Dispensers": "bags and dispensing supplies",
            "Recycling Bins": "commercial waste receptacles",
            "Hand Dryers": "facility service equipment",
            "Vending Machines": "restaurant seating and vending",
            "Reagent Bottles": "lab bottles and sample containers",
            "Pipette Tips": "pipettes and liquid handling",
            "Lab Scalpels": "lab utensils and sample preparation",
            "Test Tube Racks": "lab racks and supply storage",
            "Lab Heating Mantles": "lab heating and temperature control",
            "Lab Centrifuges": "lab mixing and separation",
            "Autoclave Accessories": "lab sterilization and washing",
            "Vacuum Pumps": "lab vacuum and drying",
            "Lab Chromatography Columns": "lab filtration and chromatography",
            "Lab Condensers": "lab distillation and evaporation",
            "Chemical Standards": "lab reagents and standards",
            "Lab Analytical Balances": "lab balances and calibration",
            "pH Meters": "water and liquid analysis",
            "Gas Monitors": "environmental measurement",
            "Oscilloscopes": "electrical test instruments",
            "Thermal Imaging Cameras": "inspection and dimensional measurement",
            "Pressure Transmitters": "pressure temperature and logging",
            "Spectrophotometers": "optical laboratory analysis",
            "Textile Machinery": "manufacturing machinery",
            "Shims & Shim Stock Raw Materials": "material sheets and stock",
            "Drum & Pail Heaters": "drum and container heaters",
            "Replacement Hydraulic Cylinders": "hydraulics and pneumatics",
            "Pallet Jack": "pallet handling equipment",
            "Shipping Label": "shipping supplies",
        }
        for product, context in cases.items():
            with self.subTest(product=product):
                identity = {
                    "category_key": "business_industrial",
                    "product_type_display": product,
                    "family_display": "Laboratory Equipment",
                }
                profile = language_profile(identity)
                self.assertEqual(profile["product_context"], context + " shelf")
                self.assertIn(context, profile["navigation"])
                self.assertIn(context, profile["trust"])
                identity["product_type_key"] = product.lower().replace(" ", "_")
                identity["product_type_display"] = "Unspecified product"
                self.assertEqual(language_profile(identity)["product_context"], context + " shelf")
        for unrelated in ("safety", "sandwich", "flaskette"):
            self.assertEqual(_overlay_values({
                "category_key": "business_industrial", "product_type_display": unrelated,
            }), {})

    def test_photography_guide_names_and_accessory_precedence(self) -> None:
        from app.audit_helpers.audit_language_resolver import _overlay_values

        cases = {
            "Video Cameras": "video and immersive cameras",
            "Instant Cameras": "film and instant cameras",
            "Underwater Camera Housings": "underwater cameras and housings",
            "Surveillance Camera Lenses": "surveillance and specialty cameras",
            "Camera & Camcorder Mounts": "camera mounting hardware",
            "Camera Stabilizers": "camera stabilizers",
            "Tripod Heads": "tripod heads and accessories",
            "Camera Battery Grips": "battery grips and docking",
            "Camera Rain Covers": "camera protection and partitions",
            "Camera Accessory Bundles": "camera bundles and repair parts",
            "Lens Mount Adapters": "lens adapters and extensions",
            "Lens Hoods": "lens caps hoods and supports",
            "Lens Filter Holders": "filter holders and rings",
            "Camera Flash Diffusers": "flash modifiers and brackets",
            "Camera Flash Slave Trigger Units": "flash triggers and sync",
            "Light Meters": "light meters",
            "Video Camera Microphones": "camera microphones",
            "Camera Film": "camera film and holders",
            "Aerial Drones": "aerial drones",
            "Aerial Drone Controllers": "drone controls and accessories",
            "Photo Studio Backgrounds": "studio backgrounds and props",
            "Photography Monolights": "studio flash lighting",
            "Photography Softboxes": "studio light modifiers",
            "Photography Lighting Stands & Booms": "lighting stands and hardware",
            "Darkroom Tanks & Reels": "darkroom processing supplies",
            "Darkroom Print & Film Washers": "darkroom processors and washing",
            "Darkroom Safelights": "darkroom lighting and storage",
            "Photo Enlarger Lenses": "photo enlargers and accessories",
            "Photography Developer": "darkroom chemicals and paper",
            "Archival Photo & Negative Sleeves": "photo archival and viewing",
            "Camera Lenses": "camera lenses",
            "Camera Bags & Cases": "camera carrying cases",
            "Digital Camera": "digital cameras",
            "Camera Tripod": "camera tripods and supports",
            "Camera Flash": "camera flashes",
            "Camera Battery": "camera batteries and power",
            "Camera Memory Card": "camera memory cards",
            "Camera Remote Controls": "camera remotes and triggers",
        }
        for product, context in cases.items():
            with self.subTest(product=product):
                identity = {
                    "category_key": "photography",
                    "product_type_display": product,
                    "family_display": "Digital Cameras",
                }
                profile = language_profile(identity)
                self.assertEqual(profile["product_context"], context + " shelf")
                self.assertIn(context, profile["navigation"])
                self.assertIn(context, profile["trust"])
                identity["product_type_key"] = product.lower().replace(" ", "_")
                identity["product_type_display"] = "Unspecified product"
                self.assertEqual(language_profile(identity)["product_context"], context + " shelf")
        for unrelated in ("camerahead", "tripodology", "flashlight"):
            self.assertEqual(_overlay_values({
                "category_key": "photography", "product_type_display": unrelated,
            }), {})

    def test_expanded_food_groups_and_overlap(self) -> None:
        groups = (
            "breakfast", "cereal", "snacks", "candy", "cookies", "baked snacks",
            "baking", "cooking ingredients", "sauces and condiments", "dressings",
            "pasta", "rice and grains", "canned and jarred food", "prepackaged meals",
            "coffee", "tea", "soft drinks", "mixers", "juice", "water", "dairy",
            "refrigerated", "frozen foods", "desserts", "protein", "nutrition",
        )
        from app.audit_helpers.audit_language_resolver import _overlay_values

        for group in groups:
            with self.subTest(group=group):
                values = _overlay_values({"category_key": "food_beverage", "product_type_display": group})
                self.assertEqual(values["product_context"], group + " shelf")
        cases = {
            "tonic water": "mixers shelf",
            "ice cream": "desserts shelf",
            "canned & jarred food": "canned and jarred food shelf",
            "peanut butter": "nut butter shelf",
            "hazelnut chocolate spread": "hazelnut spread shelf",
        }
        for product, expected in cases.items():
            with self.subTest(product=product):
                values = _overlay_values({"category_key": "food_beverage", "product_type_display": product})
                self.assertEqual(values["product_context"], expected)
        for product in ("steakhouse signage", "licorice", "teamwork", "ricepaper artwork"):
            self.assertEqual(_overlay_values({"category_key": "food_beverage", "product_type_display": product}), {})

    def test_food_guide_names_and_specific_product_precedence(self) -> None:
        cases = {
            "Breads & Buns": "breads and bakery staples",
            "Tortillas & Wraps": "tortillas and taco shells",
            "Cupcakes": "cakes and pastries",
            "Baking Soda": "baking staples",
            "Solid Baking Chocolate": "baking chocolate and decorations",
            "Sugar Substitutes": "sugars and sweeteners",
            "Maple Syrups": "honey and breakfast syrups",
            "Jams, Jellies & Preserves": "jams and fruit spreads",
            "Cheese Dips & Spreads": "savory dips and spreads",
            "Pickles": "pickles and olives",
            "Mixed Spices & Seasonings": "seasonings and spices",
            "Bouillon, Broths & Stocks": "soups and broths",
            "Dry Beans & Legumes": "beans and legumes",
            "Canned, Jarred & Cup Fruits": "canned fruit and vegetables",
            "Canned & Jarred Poultry": "canned meat and seafood",
            "Steak": "fresh meat and poultry",
            "Sausages & Hot Dogs": "deli meats and sausages",
            "Shrimp & Prawns": "seafood",
            "Bananas": "fresh fruit",
            "Packaged Salads": "fresh vegetables and salads",
            "Frozen Vegetables": "frozen fruit and vegetables",
            "Pizza Bites": "pizza and pizza snacks",
            "Tofu": "plant-based proteins",
            "Liquid Eggs": "eggs",
            "Butter & Margarine": "butter and margarine",
            "Plant-Based Milk": "plant-based milk",
            "Non-Dairy Creamers": "creamers",
            "Hot Cocoa Pods": "drink mixes and cocoa",
            "Energy Drinks": "sports and energy drinks",
            "Trail Mixes": "nuts seeds and trail mixes",
            "Jerky & Dried Meats": "jerky and meat snacks",
            "Snack Chips": "chips popcorn and savory snacks",
            "Fruit Leathers": "fruit snacks and dried produce",
            "Breakfast Bars & Cereal Bars": "snack and cereal bars",
            "Food Gift Assortments": "food gifts and snack boxes",
            "Emergency Food Kits": "emergency food",
            "Hard Ciders": "beer and cider",
            "Wine": "wine",
            "Liquor & Spirits": "spirits and ready-to-drink cocktails",
            "Licorice Candy": "candy",
            "Bottled Drinking Waters": "water",
            "Pasta Sauces": "sauces and condiments",
            "Cooking Wines": "cooking ingredients",
            "Peanut Butter": "nut butter",
            "Baking Chips": "baking chocolate and decorations",
            "Coconut Milk": "baking staples",
        }
        for product, context in cases.items():
            with self.subTest(product=product):
                identity = {
                    "category_key": "food_beverage",
                    "product_type_display": product,
                    "family_display": "Dairy & Eggs",
                }
                profile = language_profile(identity)
                self.assertEqual(profile["product_context"], context + " shelf")
                self.assertIn(context, profile["navigation"])
                self.assertIn(context, profile["trust"])
                identity["product_type_key"] = product.lower().replace(" ", "_")
                identity["product_type_display"] = "Unspecified product"
                self.assertEqual(language_profile(identity)["product_context"], context + " shelf")

    def test_priority_category_profiles_are_commercially_specific(self) -> None:
        cases = {
            "food_beverage": ("pantry shelf", "nutrition", "breakfast"),
            "beauty": ("beauty shelf", "formula", "regimen"),
            "health_personal_care": ("wellness shelf", "dosage", "symptom"),
            "animals": ("pet care shelf", "feeding", "life-stage"),
            "electronics": ("electronics shelf", "compatibility", "setup"),
        }
        for category_key, expected_terms in cases.items():
            with self.subTest(category_key=category_key):
                profile = language_profile(
                    {
                        "category_key": category_key,
                        "category_display": "Category",
                        "product_type_display": "Test Product",
                    }
                )
                joined = " ".join(profile.values()).lower()
                for term in expected_terms:
                    self.assertIn(term, joined)

    def test_slide_aware_translation_changes_expression_by_slide(self) -> None:
        identity = {
            "category_key": "beauty",
            "category_display": "Beauty",
            "family_display": "Skin Care",
            "product_type_display": "Face Moisturizers",
            "shopping_context_phrase": "Face Moisturizers shopping journey",
        }
        candidate = {"cue_key": "shopper_education", "classification": "opportunity"}
        self.assertIn(
            "regimen education",
            strategic_bullet_text(candidate, identity, slide_key="slide2").lower(),
        )
        self.assertIn(
            "discovery",
            strategic_bullet_text({"cue_key": "discoverability"}, identity, slide_key="slide3").lower(),
        )
        self.assertIn(
            "regimen education",
            strategic_bullet_text(candidate, identity, slide_key="slide4").lower(),
        )
        self.assertIn(
            "regimen education",
            strategic_bullet_text(candidate, identity, slide_key="slide5").lower(),
        )

    def test_recommendation_phrases_use_product_and_category_context(self) -> None:
        identity = {
            "category_key": "electronics",
            "product_type_display": "Bluetooth Speakers",
        }
        self.assertIn("Bluetooth Speakers SEO", recommendation_phrase(identity, "seo"))
        self.assertIn("compatibility", recommendation_phrase(identity, "attributes"))
        self.assertIn("device discovery", recommendation_phrase(identity, "discovery"))
        self.assertIn("compatibility", cue_language_label(identity, "pack_or_spec_detail"))
        self.assertIn("spec detail", cue_language_label(identity, "pack_or_spec_detail"))

    def test_product_type_overlays_make_language_more_specific(self) -> None:
        cleanser = language_profile(
            {
                "category_key": "beauty",
                "category_display": "Beauty",
                "family_display": "Skin Care",
                "product_type_display": "Facial Cleansers",
            }
        )
        peanut_butter = language_profile(
            {
                "category_key": "food_beverage",
                "category_display": "Food",
                "product_type_display": "Peanut Butter",
            }
        )
        speakers = language_profile(
            {
                "category_key": "electronics",
                "category_display": "Electronics",
                "product_type_display": "Bluetooth Speakers",
            }
        )
        self.assertIn("facial cleanser shelf", cleanser["product_context"])
        self.assertIn("sensitive-skin", cleanser["formula"])
        self.assertIn("nut butter shelf", peanut_butter["product_context"])
        self.assertIn("allergen", peanut_butter["formula"])
        self.assertIn("audio device shelf", speakers["product_context"])
        self.assertIn("battery", speakers["formula"])

    def test_category_invalid_language_guards_block_cross_category_leakage(self) -> None:
        beauty = {
            "category_key": "beauty",
            "product_type_display": "Facial Cleansers",
        }
        food = {
            "category_key": "food_beverage",
            "product_type_display": "Peanut Butter",
        }
        electronics = {
            "category_key": "electronics",
            "product_type_display": "Bluetooth Speakers",
        }
        self.assertNotIn(
            "nutrition detail",
            strategic_bullet_text(
                {"cue_key": "ingredient_or_formula_communication"},
                beauty,
                slide_key="slide4",
                evidence_terms={"detail": "Clear nutrition detail"},
            ).lower(),
        )
        self.assertNotIn("regimen", recommendation_phrase(food, "education").lower())
        self.assertNotIn(
            "ingredient detail",
            strategic_bullet_text(
                {"cue_key": "ingredient_or_formula_communication"},
                electronics,
                slide_key="slide4",
                evidence_terms={"detail": "Clear ingredient detail"},
            ).lower(),
        )

    def test_expanded_guards_replace_phrases_without_damaging_valid_terms(self) -> None:
        from app.audit_helpers.audit_language_resolver import _guard_language

        cases = {
            "food_beverage": ("BEAUTY SHELF; dosage guidance", "food and beverage shelf; serving guidance"),
            "beauty": ("ingredient and nutrition detail; serving size", "formula and ingredient detail; application amount"),
            "electronics": ("nutrition and usage education; shade and undertone", "specification and setup education; color and finish"),
            "business_industrial": ("nutrition and usage education; shade matching", "specification and operating education; specification matching"),
            "health_personal_care": ("pet feeding guidance; species suitability", "label-directed use guidance; intended-user suitability"),
        }
        for category, (original, expected) in cases.items():
            with self.subTest(category=category):
                profile = {"category_key": category}
                self.assertEqual(_guard_language(original, profile), expected)
                self.assertEqual(_guard_language(expected, profile), expected)
        preserved = {
            "food_beverage": "nutrition, flavor, brewer compatibility, and serving size",
            "beauty": "preserving color, device compatibility, lip flavor, and feeding tube",
            "electronics": "nutrition tracking app, ingredient scanner, formula editor, and feeding mechanism",
            "business_industrial": "reagent concentration, chemical formulation, foodservice capacity, and material feeding",
            "health_personal_care": "dosage, serving size, nutrition detail, sensitive-skin, and device compatibility",
        }
        for category, original in preserved.items():
            with self.subTest(preserved=category):
                self.assertEqual(_guard_language(original, {"category_key": category}), original)
        self.assertEqual(_guard_language("nutrition details", {"category_key": "beauty"}), "nutrition details")
        self.assertEqual(_guard_language("beauty shelf", {"category_key": "unknown"}), "beauty shelf")

    def test_cue_language_families_are_separated(self) -> None:
        identity = {
            "category_key": "beauty",
            "product_type_display": "Facial Cleansers",
        }
        labels = {
            cue: cue_language_label(identity, cue)
            for cue in (
                "keyword_alignment",
                "discoverability",
                "assortment_segmentation",
                "category_grouping",
                "discovery_pathways",
                "cross_category_navigation",
                "review_or_trust_signals",
                "conversion_guidance",
            )
        }
        self.assertEqual(len(set(labels.values())), len(labels))
        self.assertIn("search intent", labels["keyword_alignment"])
        self.assertIn("shelf visibility", labels["discoverability"])
        self.assertIn("segmentation", labels["assortment_segmentation"])
        self.assertIn("cross-shopping", labels["cross_category_navigation"])

    def test_recommendation_phrases_are_commercial_actions(self) -> None:
        identity = {
            "category_key": "beauty",
            "product_type_display": "Facial Cleansers",
        }
        self.assertIn("titles and PDP language", recommendation_phrase(identity, "search_intent"))
        self.assertIn("Brand Shop modules", recommendation_phrase(identity, "brand_shop"))
        self.assertIn("priority attributes", recommendation_phrase(identity, "attributes"))
        self.assertIn("decision points", recommendation_phrase(identity, "conversion"))


if __name__ == "__main__":
    unittest.main()
