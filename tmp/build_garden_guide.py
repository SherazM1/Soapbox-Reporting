import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = json.loads((root / 'config/image_guides/businessandindustrialimg.json').read_text(encoding='utf-8'))
style = json.loads((root / 'config/style_guides/gardenandpatio.json').read_text(encoding='utf-8-sig'))
norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
known = {norm(p['display_name']) for f in style['families'].values() for p in f['product_types'].values()}
known.update(map(norm, ['Grills, Fire Pits & Outdoor Cooking', 'Pond & Pool Pumps']))
groups = []
for section in (root / 'tmp/pdfs/garden.txt').read_text(encoding='utf-8').split('Main Image\n')[1:]:
    lines = section.split('Product types included:')[0].strip().splitlines()
    names = []
    i = 0
    while i < len(lines):
        for count in range(1, 4):
            name = ' '.join(x.strip() for x in lines[i:i+count])
            if norm(name) in known:
                names.append(name)
                i += count
                break
        else:
            raise ValueError(lines[i])
    groups.append(names)
g = {k: base[k] for k in ['category', 'category_key', 'guide_key', 'source', 'last_updated', 'status', 'global_guidance']}
g.update(category='Garden & Patio', category_key='garden_patio', guide_key='gardenandpatioimg', source='FY27_Category_StyleGuide_Garden-Patio.pdf')
g['slot_definitions'] = {'main_image': base['slot_definitions']['main_image']}
def slot(key, label, guidance, recommendation, hints):
    g['slot_definitions'][key] = dict(label=label, guidance=guidance, recommendation=recommendation, detection_hints=hints)
slot('silo_front', 'Silo, Front', 'Photograph the product straight-on on a pure white background. Show the full assembled unit for large equipment; follow the product section for plant or seed packaging requirements.', 'Add a straight-on front silo on pure white, showing the full assembled equipment or the plant/seed packaging specified by the section.', ['front', 'silo', 'main', 'hero', 'white background', 'assembled'])
slot('lifestyle_in_use', 'Lifestyle, In-Use', 'Show the product in a realistic outdoor or patio setting that feels welcoming and achievable.', 'Add a lifestyle image showing the product in a realistic outdoor or patio setting.', ['lifestyle', 'in-use', 'in use', 'outdoor', 'patio', 'garden'])
slot('feature_graphic', 'Feature Graphic', 'Use the section-specific feature direction for each graphic, preserving separate primary and secondary feature images.', 'Add the feature graphic specified by the product section, with clear callouts or the planting-process image where directed.', ['feature', 'benefit', 'callout', 'graphic', 'specifications', 'performance'])
slot('silo_front_in_pack', 'Silo, Front, In Pack', 'Show the front of product packaging straight-on on a white background with all branding and key label details legible.', 'Add a straight-on front packaging image on white with branding and key label details legible.', ['front in pack', 'package front', 'retail packaging', 'in pack', 'branding'])
slot('silo_back_in_pack', 'Silo, Back, In Pack', 'Show the back of product packaging straight-on on a pure white background with all label details legible.', 'Add a straight-on back packaging image on pure white with all label details legible.', ['back in pack', 'package back', 'back packaging', 'rear packaging', 'label details'])
slot('dimensions', 'Dimensions', 'Show product dimensions on the silo using the section-specific cutting width, folded, assembled, inflated, or installed measurements and footprint diagram where applicable.', 'Add a dimensions graphic with the measurements and footprint specified for the product section.', ['dimensions', 'height', 'width', 'depth', 'footprint', 'cutting width', 'folded dimensions', 'inflated dimensions'])
slot('lifestyle_result', 'Lifestyle, Result', 'Show the plant at peak bloom or harvest in a vibrant, true-to-color garden setting that reflects the end result.', 'Add a result lifestyle image showing the plant at peak bloom or harvest in a true-to-color garden setting.', ['result', 'peak bloom', 'harvest', 'bloom', 'grown plant', 'end result'])
recommendations = [
    'Coverage / Lawn Size Guide: Show a coverage area chart or recommended lawn size range for this product.',
    'Assembly Overview: Provide a simplified assembly or setup diagram for powered equipment or furniture.',
    'Compatible Accessories: Show compatible attachments, accessories, or replacement parts for this product.',
    "Warranty / Brand Guarantee: Include the brand's warranty graphic or satisfaction guarantee badge if applicable.",
    'Seasonal Use Graphic: Show the seasons or growing zones this product is ideal for using a map or icon chart.',
]
sections = [
    ('lawn_mowers_grills_power_equipment_parts', 'Lawn Mowers, Grills, Power Equipment & Parts', ['silo_front', 'lifestyle_in_use', 'feature_graphic', 'feature_graphic', 'silo_front_in_pack', 'dimensions'], [
        'Show the full assembled unit for large equipment in the front silo.',
        'The first feature graphic highlights the primary performance specification such as cutting width, power, voltage, cord length, or airflow using bold callouts.',
        'The second feature graphic highlights secondary features such as self-propelled drive, mulching, adjustable height, or foldable design using icon callouts.',
        'Dimensions should overlay cutting deck width or product dimensions on the silo, including cutting-path width or folded dimensions where applicable.',
        'The source product list includes the broad entry Grills, Fire Pits & Outdoor Cooking; preserve it as printed.',
    ]),
    ('gardening_tools_supplies_plant_care', 'Gardening Tools, Supplies & Plant Care', ['silo_front', 'lifestyle_in_use', 'feature_graphic', 'feature_graphic', 'silo_front_in_pack'], [
        'Show the full assembled unit for large equipment in the front silo.',
        'The first feature graphic highlights the primary tool benefit such as ergonomic grip, durability, coverage rate, or non-toxic formula using callout graphics.',
        'The second feature graphic shows coverage area or product dimensions using clean graphics to communicate scale.',
        'The source has an unlabeled empty image box after the packaging view; it does not define an additional required slot.',
    ]),
    ('plants_seeds_bulbs', 'Plants, Seeds & Bulbs', ['silo_front', 'lifestyle_in_use', 'feature_graphic', 'feature_graphic', 'silo_front_in_pack', 'silo_back_in_pack', 'lifestyle_result'], [
        'Photograph the front of the plant or seed packaging straight-on on pure white with variety name, bloom color, and growing information clearly legible.',
        'The first feature graphic displays sun, water, spacing, and zone requirements using icon callouts.',
        'The second feature graphic shows the planting process such as seeds, bulbs, or seedlings in a natural garden setting with diverse representation.',
        'The final Lifestyle Result image appears on the second row and shows peak bloom or harvest in a vibrant, true-to-color garden setting.',
    ]),
    ('outdoor_decor_patio_shade_structures', 'Outdoor Decor, Patio, Shade & Structures', ['silo_front', 'lifestyle_in_use', 'feature_graphic', 'feature_graphic', 'dimensions'], [
        'The front silo shows the full assembled unit for large equipment and front packaging with variety and growing information for plants or seeds, as directed by the source.',
        'The first feature graphic highlights material quality such as UV-resistant, weather-proof, or durable construction and includes warranty or durability claims using callouts.',
        'The second feature graphic highlights secondary features such as easy assembly, fold-flat storage, weight capacity, or wind resistance.',
        'Overlay assembled dimensions on the silo and include a footprint diagram for coverage or surface area where applicable.',
        'Front and back packaging views are additional recommendations when applicable, rather than required carousel slots on this page.',
    ]),
    ('pools_spas_water_features_seasonal', 'Pools, Spas, Water Features & Seasonal', ['silo_front', 'lifestyle_in_use', 'feature_graphic', 'feature_graphic', 'silo_front_in_pack', 'silo_back_in_pack', 'dimensions'], [
        'The front silo shows the full assembled unit for large equipment and front packaging with variety and growing information for plants or seeds, as directed by the source.',
        'The first feature graphic highlights pool or spa capacity, dimensions, or flow rate using callout graphics and includes quick-inflate features where applicable.',
        'The second feature graphic calls out safety features, filtration, included accessories, or setup details and includes coverage or protection claims where relevant.',
        'Dimensions appear on the second row and show assembled dimensions with a footprint diagram, including inflated or installed dimensions where relevant.',
    ]),
]
g['pages'] = {}
g['product_type_index'] = {}
for i, ((key, title, required, notes), names) in enumerate(zip(sections, groups)):
    additional = recommendations if i != 3 else ['Silos, Front & Back of Pack: Show the front or back of product packaging, if applicable, straight-on on white with all branding and key label details legible.'] + recommendations[:4]
    notes = [f'Source PDF page {i+3}. Preserve the carousel order shown on the source page.', 'Preserve repeated feature_graphic because the source page shows two separate feature graphics.'] + notes
    if i != 2:
        notes.append('Individual brand guidelines should take precedence.')
    g['pages'][key] = dict(display_name=title, format='General', product_types=names, required_slots=required, additional_recommendations=additional, notes=notes)
    for name in names:
        assert name not in g['product_type_index']
        g['product_type_index'][name] = key
assert list(map(len, groups)) == [76, 77, 10, 48, 45]
(root / 'config/image_guides/gardenandpatioimg.json').write_text(json.dumps(g, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('Created Garden & Patio: five sections and 256 product mappings.')
