"""
shelf_sequence_checker.py — Visicooler shelf pack-size ordering check.

IMPORTANT — how "shelf number" is actually determined
-------------------------------------------------------
Each row in orgi.coolermetricsmaster (one iterationid + iterationtranid) is
ONE PHOTO OF ONE SHELF, not a photo of the whole cooler. The real shelf
position of an image is NOT the `shelfnumber` value stored on its
orgi.coolermetricstransaction detection rows — it is the RANK of that
image's iterationtranid among all the images belonging to the same cooler
(same iterationid + storeid + caserid), sorted ascending:

    lowest iterationtranid for that (iterationid, storeid, caserid)  -> shelf 1 (TOP-most)
    next lowest                                                     -> shelf 2
    ... and so on, down to the highest iterationtranid -> the BOTTOM-most shelf.

Example (orgi.coolermetricsmaster):
    iterationid=285  iterationtranid=1  storeid=1130110008  caserid=3   -> shelf 1 (top)
    iterationid=285  iterationtranid=2  storeid=1130110008  caserid=3   -> shelf 2
    iterationid=285  iterationtranid=3  storeid=1130110008  caserid=3   -> shelf 3
    iterationid=285  iterationtranid=4  storeid=1130110008  caserid=3   -> shelf 4 (bottom)
    iterationid=285  iterationtranid=5  storeid=1130110016  caserid=5   -> shelf 1 (different cooler)
    ...

The total number of images in that (iterationid, storeid, caserid) group is
the cooler's actual shelf count (N), and that N is what decides which rule
set applies — NOT any single row's `shelfnumber` value.

Rules (as specified by the business)
-------------------------------------
  "Other" / no-product / structural classes (Other CAN, Other PET, Other
  RGB, Other TPK, Other Water Bottle, Others, No Coke Products,
  Shelf-detection, Cooler, Shelf) and any class whose brand can't be
  determined at all are ignored entirely — not real, size-resolved SKUs
  and not considered by either condition below.

  "Unidentified" bottles (brand known, Large/Small bucket known, exact
  pack size NOT known — the 9000-series "(unresolved size)" classes and
  combined-size placeholders like "1500/2000/2250") are NO LONGER simply
  ignored or force-assigned a guessed minimum size. They are redistributed
  per-image, per-brand, per-bucket against that same image's real,
  identified SKUs of the same brand+bucket (or a Virtual SKU if none
  exist), so their bottles still count toward Condition 1/2 with a
  realistic size instead of vanishing or skewing the size comparison with
  a fixed minimum. See load_product_maps() and
  redistribute_unidentified_bottles() for the full algorithm ("Unidentified
  Bottle Redistribution Logic").

  Shelf 1 (the TOP-most shelf, rank 1) is ALWAYS compliant ('Y') — there
  is no shelf above it to compare against, so no check is applied to it.

  For every shelf N > 1, two independent conditions are checked. If
  Condition 1 passes, the shelf is 'Y' and Condition 2 is never evaluated.
  Only if Condition 1 fails is Condition 2 checked. The shelf is 'Y' if
  EITHER condition passes, 'N' only if BOTH fail.

  Condition 1 — top-to-bottom size growth:
      Take the largest resolved SKU size on shelf N-1 (the shelf immediately
      above, i.e. ONLY the previous shelf — no skipping through empty
      shelves) and the smallest resolved SKU size on shelf N (the shelf
            below it). Passes iff min(shelf N) >= max(shelf N-1) — i.e. sizes must
      grow as you go DOWN the cooler, from shelf 1 (top) to shelf N
      (bottom). If shelf N-1 has zero resolved SKU detections, Condition 1
      automatically fails for shelf N (no carrying-forward to an earlier
      shelf).

  Condition 2 — left-to-right size growth, evaluated on shelf N alone:
      Sort shelf N's resolved SKUs by their bounding box left edge (x1)
            ascending. Passes iff sizes are non-decreasing left-to-right. Any SKU
      missing x1/y1/x2/y2 is dropped from this check entirely (not counted
      either way). A shelf left with 0 usable SKUs after that filtering
      fails Condition 2; a shelf with exactly 1 usable SKU trivially
      passes it (nothing to compare).

Output: one row per (iterationid, iterationtranid) is upserted into
orgi.shelfsequencecompliance — see shelf_sequence_compliance_ddl.sql.
  - maxshelfnumber : total shelf count for this image's cooler (same value
                      for every image in the group)
  - shelfposition  : this specific image's rank within its cooler (1..N) —
                      this is the number actually used to pick/apply the rule
  - skusconsidered : count of resolved (non-"Other") SKUs detected on this
                      shelf, regardless of whether they had bbox coordinates
  - compliance_flag: 'Y' if shelf 1, or if Condition 1 or Condition 2
                      passed for this shelf; 'N' if both conditions failed

Assumptions made explicit (flag these if they don't match your intent):
  1. A "cooler" = one (iterationid, storeid, caserid) group. If two images
     for the same store in the same iteration can legitimately have
     different caserid values but still be the same physical cooler, this
     will split them into separate coolers — let me know and I'll change
     the grouping key.
  2. Shelf rank is assigned purely by sorting iterationtranid ascending
     within the group, with the LOWEST iterationtranid = shelf 1 = the
     TOP-most shelf (confirmed: business spec numbers shelves 1->2->3->4
     top to bottom). If ordering should instead follow `modelrun`
     timestamp (in case iterationtranid isn't allocated in capture order),
     or if lowest-tranid is actually the bottom shelf rather than the top
     in your capture pipeline, say so and I'll adjust.
  3. "Left-to-right" is determined by sorting on x1 (bounding box left
     edge) ascending. If your coordinate system's origin/axis direction
     means x1 doesn't correspond to physical left-to-right, say so.
    4. "Increase" in both conditions means greater than or equal (ties pass).
  5. orgi.coolermetricsmaster has no imagefilename column, so an image with
     literally zero detection rows (nothing detected on it at all) still
     counts as a shelf position (affecting N), but its imagefilename will
     show as NULL — that's a schema limitation, not a bug in this script.
"""

import json
import logging
import os
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from app.db_handler import initialize_db_connection, close_db_connection
    from app.db_retry import retry_on_network_error, verify_connection
    from app.cap_sku_mapper import extract_pack_size, extract_package_type, extract_brand
except ImportError:
    # Allows the module to be imported/tested outside the `app` package layout.
    from db_handler import initialize_db_connection, close_db_connection
    from db_retry import retry_on_network_error, verify_connection
    from cap_sku_mapper import extract_pack_size, extract_package_type, extract_brand


# ─────────────────────────────────────────────────────────────────────────────
# Product-class → pack size resolution
# ─────────────────────────────────────────────────────────────────────────────

# Fallback snapshot (used only if orgi.productmaster can't be reached / is
# empty) built from the current productclassid -> productname mapping,
# already filtered down to real, size-resolved, non-"Other" SKUs.
# Format: classid -> (packtype, packsize_ml)
#
# NOTE: the 9000-series "(unresolved size)" classes (9020-9024, 9028, 9029,
# 9040-9043) are deliberately NOT in this table — they no longer get a
# fixed size at all. They're "Unidentified" bottles now handled by the
# redistribution logic below (see UNIDENTIFIED_BUCKET_SOURCE), not treated
# as directly-resolved SKUs.
STATIC_CLASS_SIZE_MAP = {
    0: ('PET', 1000.0), 1: ('PET', 1500.0), 2: ('PET', 2000.0), 3: ('PET', 2250.0),
    4: ('RGB', 250.0), 5: ('PET', 250.0), 6: ('PET', 500.0), 7: ('PET', 1000.0),
    8: ('PET', 2000.0), 9: ('PET', 2250.0), 10: ('PET', 250.0), 11: ('PET', 1000.0),
    12: ('PET', 1500.0), 13: ('PET', 2000.0), 14: ('PET', 2250.0), 15: ('RGB', 250.0),
    16: ('PET', 250.0), 17: ('PET', 500.0), 18: ('RGB', 250.0), 19: ('PET', 250.0),
    20: ('PET', 500.0), 21: ('PET', 1000.0), 22: ('PET', 500.0), 29: ('PET', 1000.0),
    30: ('PET', 1500.0), 31: ('PET', 2000.0), 32: ('PET', 2250.0), 33: ('RGB', 250.0),
    34: ('PET', 250.0), 35: ('PET', 500.0), 36: ('PET', 250.0), 37: ('PET', 500.0),
    56: ('RGB', 175.0), 57: ('RGB', 175.0), 58: ('RGB', 175.0), 60: ('PET', 250.0),
    61: ('PET', 500.0),
    # Cap-mapping bucket classes (9000+) that DO already have one exact,
    # unambiguous size — genuinely resolved, unlike the ones below.
    9001: ('PET', 1000.0), 9002: ('PET', 1000.0),
    9010: ('RGB', 250.0), 9011: ('RGB', 250.0), 9012: ('RGB', 250.0), 9013: ('RGB', 250.0),
}

# ── Unidentified-bottle redistribution ──────────────────────────────────────
# Business logic: an "unidentified" bottle is one where the AI could tell
# the brand and whether it's Large (>=1L) or Small (<1L), but NOT the exact
# SKU/pack size. Rather than assigning it a guessed fixed size (the old
# behaviour — see the removed CAP_BUCKET_SIZE_OVERRIDES table), these
# classes now feed a per-image, per-brand, per-bucket redistribution pool
# (see redistribute_unidentified_bottles()) that reassigns each unidentified
# bottle's size to a real SKU already detected on the same shelf/brand, or —
# if none exists — a Virtual SKU. See module docstring update below and
# redistribute_unidentified_bottles() for the full algorithm.
#
# UNIDENTIFIED_BUCKET_SOURCE: classid -> (packtype, min_of_range_ml). The
# min-of-range value is used ONLY to derive which bucket (large/small) the
# class belongs to — it is no longer used as an assigned size.
#   9020-9024, 9029 (various "(unresolved size)" PET/CAN/RGB)  -> small
#   9028 (Kinley Water, unresolved: 500/1000ml)                -> small (min=500)
#   9040 (Coca-Cola PET <1000ml: 250/500ml)                    -> small
#   9041 (Coca-Cola PET >=1000ml: 1500/2000/2250ml)             -> large (min=1500)
#   9042 (Sprite PET <=500ml: 250/500ml)                       -> small
#   9043 (Fanta PET <=500ml: 250/500ml)                        -> small
# NOTE: 9025 (Thums Up), 9026 (Limca), 9027 (Maaza), 9030 "Other Water
# Bottle Cap", and 9031 "Other Cap (unresolved brand)" are deliberately NOT
# in this table — unchanged from before, they stay excluded entirely (no
# bucket, no redistribution, same as any other unmapped/unresolved-brand SKU).
UNIDENTIFIED_BUCKET_SOURCE = {
    9020: ('PET', 250.0),   # Coca-Cola PET (unresolved size)
    9021: ('CAN', 250.0),   # Coca-Cola Can (unresolved size)
    9022: ('PET', 250.0),   # Coca-Cola Zero PET (unresolved size)
    9023: ('PET', 250.0),   # Sprite PET (unresolved size)
    9024: ('PET', 250.0),   # Fanta PET (unresolved size)
    9028: ('PET', 500.0),   # Kinley Water PET (unresolved size)
    9029: ('RGB', 250.0),   # Kinley Soda (unresolved size)
    9040: ('PET', 250.0),   # Coca-Cola PET <1000ml
    9041: ('PET', 1500.0),  # Coca-Cola PET >=1000ml
    9042: ('PET', 250.0),   # Sprite PET <=500ml
    9043: ('PET', 250.0),   # Fanta PET <=500ml
}

LARGE_MIN_ML = 1000.0   # >= this = Large bucket, else Small
LARGE_THRESHOLD = 3     # min confirmed bottles before a Large product is "established"
SMALL_THRESHOLD = 6     # min confirmed bottles before a Small product is "established"


def _bucket_for_size(size_ml: float) -> str:
    return 'large' if size_ml >= LARGE_MIN_ML else 'small'


# classid -> 'large' | 'small', derived from UNIDENTIFIED_BUCKET_SOURCE's
# min-of-range value.
UNIDENTIFIED_BUCKET_MAP = {
    classid: _bucket_for_size(size_ml)
    for classid, (_pkg, size_ml) in UNIDENTIFIED_BUCKET_SOURCE.items()
}

_IGNORE_NAME_MARKERS = (
    'other',            # Other CAN / Other PET / Other RGB / Other TPK /
                         # Other Water Bottle / Others / Other Cap / Other Water Bottle Cap
    'no coke products',
    'shelf-detection',
    'unresolved',
)



def _to_ml(size_str: str) -> float:
    """Convert '1000ml' / '1.5l' style strings (from extract_pack_size) to ml."""
    if not size_str:
        return 0.0
    try:
        if size_str.endswith('ml'):
            return float(size_str[:-2])
        if size_str.endswith('l'):
            return float(size_str[:-1]) * 1000.0
    except ValueError:
        return 0.0
    return 0.0


def _is_ignorable(name: str) -> bool:
    """True for 'Other'/unresolved/ambiguous rows that should never be size-checked."""
    if not name:
        return True
    nl = name.lower().strip()
    if nl in ('shelf', 'cooler', 'others'):
        return True
    if '/' in name:  # combined placeholders e.g. "Coca-Cola 1500/2000/2250"
        return True
    if 'small' in nl:  # "Coke Small" etc. — ambiguous, no fixed ml
        return True
    return any(marker in nl for marker in _IGNORE_NAME_MARKERS)


_COMBINED_SIZE_RE = re.compile(
    r'(\d[\d.]*(?:\s*/\s*\d[\d.]*)+)\s*(ml|l)?', re.IGNORECASE
)


def _parse_combined_sizes(name: str) -> list:
    """Parse a combined-size placeholder name like 'Coca-Cola 1500/2000/2250'
    (assumed ml if no unit is present) into [1500.0, 2000.0, 2250.0]."""
    m = _COMBINED_SIZE_RE.search(name)
    if not m:
        return []
    nums_part, unit = m.group(1), (m.group(2) or 'ml').lower()
    sizes = []
    for tok in nums_part.split('/'):
        tok = tok.strip()
        try:
            val = float(tok)
        except ValueError:
            continue
        if unit == 'l':
            val *= 1000.0
        sizes.append(val)
    return sizes


def load_product_maps(cur) -> dict:
    """
    Reads orgi.productmaster once and builds everything the shelf-sequence
    check + unidentified-bottle redistribution need together:

      size_map            : {classid: (packtype, size_ml)} — real,
                             exactly size-resolved, non-"Other" SKUs.
                             Unchanged meaning from the old
                             load_product_maps() (formerly
                             load_product_size_map()).
      brand_map            : {classid: brand} — every classid (resolved,
                             unidentified-bucket source, or dedicated
                             virtual-SKU placeholder) with a determinable
                             brand. Used to group detections for
                             redistribution.
      virtual_sku_map      : {(brand, bucket): classid} — dedicated
                             combined-size placeholder classids already
                             sitting in productmaster (e.g. "Coca-Cola
                             1500/2000/2250"), keyed by the brand+bucket
                             they represent.
      virtual_sku_sizes    : {(brand, bucket): size_ml} — representative
                             size (MIN of the combined range) for the same
                             keys as virtual_sku_map.
      brand_size_catalog   : {brand: {'large': [(classid,size_ml), ...],
                                       'small': [(classid,size_ml), ...]}}
                             — every real resolved SKU of that brand,
                             sorted ascending by size. Used as the
                             auto-derived fallback priority list when no
                             dedicated virtual SKU exists for a
                             brand+bucket.

    Falls back to STATIC_CLASS_SIZE_MAP for size_map only if
    orgi.productmaster can't be reached. In that degraded mode there are no
    product names to parse brand/virtual-SKU info from, so brand_map /
    virtual_sku_map / brand_size_catalog stay empty and unidentified-bottle
    redistribution has nothing to redistribute into (those bottles are
    simply dropped, same as the old pre-redistribution behaviour).
    """
    size_map = {}
    brand_map = {}
    virtual_sku_map = {}
    virtual_sku_sizes = {}
    brand_size_catalog = {}

    try:
        cur.execute("SELECT productclassid, productname FROM orgi.productmaster")
        rows = cur.fetchall()
        for classid, name in rows:
            if classid is None or not name:
                continue
            classid = int(classid)
            brand = extract_brand(name)

            # 1) Real, size-resolved SKUs (non-"Other", non-unidentified-bucket)
            if classid not in UNIDENTIFIED_BUCKET_MAP and not _is_ignorable(name):
                packtype = extract_package_type(name).upper()
                ml = _to_ml(extract_pack_size(name))
                if ml > 0:
                    size_map[classid] = (packtype, ml)
                    if brand:
                        brand_map[classid] = brand
                        bucket = _bucket_for_size(ml)
                        brand_size_catalog.setdefault(brand, {'large': [], 'small': []})
                        brand_size_catalog[brand][bucket].append((classid, ml))
                    continue

            # 2) Unidentified-bucket source classes (9020-9043, etc.) — no
            #    size assigned here; just record the brand so the
            #    redistribution pool can group by it.
            if classid in UNIDENTIFIED_BUCKET_MAP:
                if brand:
                    brand_map[classid] = brand
                continue

            # 3) Dedicated combined-size Virtual SKU placeholders already
            #    present in productmaster (name contains '/').
            if '/' in name:
                sizes = _parse_combined_sizes(name)
                if brand and sizes:
                    rep_size = min(sizes)
                    bucket = _bucket_for_size(rep_size)
                    key = (brand, bucket)
                    if key not in virtual_sku_map:
                        virtual_sku_map[key] = classid
                        virtual_sku_sizes[key] = rep_size
                        brand_map[classid] = brand
                    else:
                        logger.warning(
                            f"Multiple combined-size placeholders found for "
                            f"brand={brand} bucket={bucket}; keeping classid "
                            f"{virtual_sku_map[key]}, ignoring {classid} ('{name}')"
                        )
                continue

            # else: genuinely ignorable (Other / no brand / etc.) — dropped,
            # unchanged from before.
    except Exception as e:
        logger.warning(f"Could not build product maps from orgi.productmaster: {e}")

    if not size_map:
        logger.warning(
            "Falling back to STATIC_CLASS_SIZE_MAP for product sizes — "
            "brand-aware unidentified-bottle redistribution will be "
            "unavailable in this degraded mode"
        )
        size_map = dict(STATIC_CLASS_SIZE_MAP)

    for catalog in brand_size_catalog.values():
        catalog['large'].sort(key=lambda t: t[1])
        catalog['small'].sort(key=lambda t: t[1])

    logger.info(
        f"Loaded {len(size_map)} size-resolved product classes, "
        f"{len(virtual_sku_map)} dedicated virtual-SKU placeholder(s), "
        f"brand catalog covers {len(brand_size_catalog)} brand(s)"
    )
    return {
        'size_map': size_map,
        'brand_map': brand_map,
        'virtual_sku_map': virtual_sku_map,
        'virtual_sku_sizes': virtual_sku_sizes,
        'brand_size_catalog': brand_size_catalog,
    }


def reorder_catalog_by_frequency(product_maps: dict, classid_frequency: dict) -> None:
    """
    Re-ranks each brand's auto-derived fallback priority list (used only
    in Step 4 when NO dedicated Virtual SKU exists for a brand+bucket) by
    how often that classid is actually detected elsewhere in THIS batch —
    descending — instead of raw ascending size, using size only as a
    tiebreak. Mutates product_maps['brand_size_catalog'] in place.

    Why: sorting purely by size means whichever SKU happens to be a
    brand's smallest catalog entry (very often an uncommon "175ml Glass"
    variant) becomes the default target for EVERY unidentified bottle with
    no real product on the same shelf to anchor to — even if that SKU has
    zero real detections anywhere in the current run. Ranking by observed
    frequency first means an unseen/rare catalog SKU can only be picked if
    nothing more plausible is available.

    classid_frequency: {classid: count} — how many times each classid was
        actually detected across the fetched batch (see
        run_shelf_sequence_check).
    """
    for catalog in product_maps['brand_size_catalog'].values():
        for bucket_list in (catalog['large'], catalog['small']):
            bucket_list.sort(
                key=lambda t: (-classid_frequency.get(t[0], 0), t[1])
            )


def _redistribute_pool_into_products(pool_items: list, candidates: list, threshold: int) -> None:
    """
    Implements Steps 3 & 5 of the unidentified-bottle redistribution spec
    for ONE (brand, bucket) group on ONE image. Mutates each candidate's
    'items' list in place, consuming pool_items; every candidate's 'size'
    is stamped onto whichever pool items get assigned to it.

    candidates: list of {'classid', 'size', 'items'} — either real resolved
        products (possibly already holding detected items) or freshly
        created Virtual SKU placeholders (items=[] initially).
    """
    remaining = list(pool_items)

    # Pass A (Cases 1-3): top up products below threshold, descending by
    # current count — ties (e.g. several freshly-created Virtual SKUs, all
    # starting at 0) fall back to the given candidate order, i.e. priority
    # order for the auto-derived fallback list.
    deficient = sorted(
        (c for c in candidates if len(c['items']) < threshold),
        key=lambda c: -len(c['items']),
    )
    for cand in deficient:
        if not remaining:
            break
        needed = threshold - len(cand['items'])
        if len(remaining) >= needed:
            take, remaining = remaining[:needed], remaining[needed:]
            for it in take:
                it['size'] = cand['size']
            cand['items'].extend(take)
        # else (Case 3): not enough left to reach threshold — skip this
        # candidate entirely, pool untouched, try the next one.

    # Pass B (Step 5): dump ALL remaining pool bottles onto whichever
    # candidate now has the highest count, so nothing is ever lost.
    if remaining:
        target = max(candidates, key=lambda c: len(c['items']))
        for it in remaining:
            it['size'] = target['size']
        target['items'].extend(remaining)


def redistribute_unidentified_bottles(raw_items: list, product_maps: dict) -> list:
    """
    ONE image's worth of raw detections in, a final list of
    {'size','x1','y1','x2','y2'} dicts out — the same shape
    evaluate_store_shelf_sequence()/Condition 2 already expect.

    raw_items: list of {'classid', 'x1', 'y1', 'x2', 'y2'} — every
        detection on this image, not yet resolved to a size.

    Implements the full "Unidentified Bottle Redistribution Logic": real
    resolved SKUs pass through unchanged; unidentified-bucket detections
    (brand + Large/Small known, exact SKU not) are pooled per (image,
    brand, bucket) and redistributed into that brand's real SKUs on this
    same image (Steps 1-3), falling back to a Virtual SKU when no real SKU
    of that brand exists on this shelf (Step 4), with any final leftover
    assigned to the highest-count product so no bottle is lost (Step 5).
    """
    size_map = product_maps['size_map']
    brand_map = product_maps['brand_map']
    virtual_sku_map = product_maps['virtual_sku_map']
    virtual_sku_sizes = product_maps['virtual_sku_sizes']
    brand_size_catalog = product_maps['brand_size_catalog']

    resolved_by_brand = {}       # brand -> classid -> [item dict, ...]
    pool_by_brand_bucket = {}    # (brand, bucket) -> [item dict, ...] (no size yet)
    passthrough = []             # resolved items with no determinable brand

    for it in raw_items:
        classid = it.get('classid')
        if classid is None:
            continue
        classid = int(classid)

        if classid in size_map:
            _pkg, size_ml = size_map[classid]
            item = {'size': size_ml, 'x1': it.get('x1'), 'y1': it.get('y1'),
                     'x2': it.get('x2'), 'y2': it.get('y2')}
            brand = brand_map.get(classid)
            if brand:
                resolved_by_brand.setdefault(brand, {}).setdefault(classid, []).append(item)
            else:
                passthrough.append(item)
            continue

        if classid in UNIDENTIFIED_BUCKET_MAP:
            brand = brand_map.get(classid)
            if not brand:
                continue  # can't group without a brand -> dropped, as before
            bucket = UNIDENTIFIED_BUCKET_MAP[classid]
            item = {'x1': it.get('x1'), 'y1': it.get('y1'),
                     'x2': it.get('x2'), 'y2': it.get('y2')}
            pool_by_brand_bucket.setdefault((brand, bucket), []).append(item)
            continue

        # else: genuinely ignorable / unmapped -> dropped, unchanged

    final_items = list(passthrough)
    all_brands = set(resolved_by_brand) | {b for (b, _bucket) in pool_by_brand_bucket}

    for brand in all_brands:
        products_by_classid = resolved_by_brand.get(brand, {})
        pooled_buckets = set()

        for bucket in ('large', 'small'):
            pool_items = pool_by_brand_bucket.get((brand, bucket))
            if not pool_items:
                continue
            pooled_buckets.add(bucket)
            threshold = LARGE_THRESHOLD if bucket == 'large' else SMALL_THRESHOLD

            candidates = [
                {'classid': classid, 'size': items[0]['size'], 'items': items}
                for classid, items in products_by_classid.items()
                if _bucket_for_size(items[0]['size']) == bucket
            ]

            if not candidates:
                # Step 4 — no identified product of this brand+bucket on
                # this shelf at all.
                key = (brand, bucket)
                if key in virtual_sku_map:
                    candidates = [{
                        'classid': virtual_sku_map[key],
                        'size': virtual_sku_sizes[key],
                        'items': [],
                    }]
                else:
                    fallback_sizes = brand_size_catalog.get(brand, {}).get(bucket, [])
                    if fallback_sizes:
                        candidates = [
                            {'classid': cid, 'size': sz, 'items': []}
                            for cid, sz in fallback_sizes
                        ]
                    else:
                        logger.warning(
                            f"Unidentified {bucket} bottles for brand={brand}: "
                            f"no identified product, no dedicated virtual SKU, "
                            f"and no fallback sizes in productmaster — "
                            f"{len(pool_items)} bottle(s) dropped from this shelf's count"
                        )
                        continue

            _redistribute_pool_into_products(pool_items, candidates, threshold)

            for cand in candidates:
                final_items.extend(cand['items'])

        # Resolved products of this brand whose bucket never had any pool
        # activity this round haven't been emitted yet — flush them now.
        for classid, items in products_by_classid.items():
            bucket = _bucket_for_size(items[0]['size'])
            if bucket not in pooled_buckets:
                final_items.extend(items)

    return final_items


# ─────────────────────────────────────────────────────────────────────────────
# Shelf ordering rule — evaluated per store/cooler group, ranked by iterationtranid
# ─────────────────────────────────────────────────────────────────────────────

def _condition2_left_to_right(items):
    """
    Condition 2 for one shelf: sort its resolved SKUs by bounding-box left
    edge (x1) ascending and check sizes are non-decreasing left-to-right.

    items: list of dicts, each with keys 'size', 'x1', 'y1', 'x2', 'y2'
        (bbox values may be None). Only resolved (non-"Other") SKUs should
        already be in this list.

    Returns (passed: bool, remark: str)
    """
    if not items:
        return False, "condition2: shelf has no valid SKU detections at all"

    usable = [
        it for it in items
        if it['x1'] is not None and it['y1'] is not None
        and it['x2'] is not None and it['y2'] is not None
    ]
    if not usable:
        return False, "condition2: none of the shelf's SKUs have bounding-box coordinates"

    usable.sort(key=lambda it: it['x1'])
    sizes_lr = [it['size'] for it in usable]
    for i in range(1, len(sizes_lr)):
        if sizes_lr[i] < sizes_lr[i - 1]:
            return False, f"condition2: sizes not non-decreasing left-to-right, got {sizes_lr}"
    return True, ''


def evaluate_store_shelf_sequence(shelf_items_by_rank: list) -> list:
    """
    shelf_items_by_rank: list of lists, index 0 = shelf 1 (lowest
        iterationtranid in the group), index 1 = shelf 2, etc. Each inner
        list holds dicts {'size': packsize_ml, 'x1','y1','x2','y2'} for the
        resolved (non-"Other") SKUs detected on that shelf's image.

    Returns a list of (compliance_flag, remarks) tuples, same length/order
    as shelf_items_by_rank — i.e. result[i] is the verdict for shelf i+1.

    Shelf 1 (top-most) is always ('Y', ''). For shelf N>1: Condition 1
    (top-to-bottom size growth, vs. the immediately previous shelf only)
    is checked first; if it
    passes the shelf is 'Y'. Otherwise Condition 2 (left-to-right on this
    shelf alone) is checked; if that passes the shelf is 'Y'. Only if both
    fail is the shelf 'N'.
    """
    n = len(shelf_items_by_rank)
    if n == 0:
        return []

    results = [None] * n
    results[0] = ('Y', '')  # shelf 1 — always compliant, no check

    for i in range(1, n):
        prev_items = shelf_items_by_rank[i - 1]
        cur_items = shelf_items_by_rank[i]

        # ── Condition 1: top-to-bottom size growth, vs. previous shelf only ──
        # Every SKU on the current shelf must be at least as large as the largest
        # SKU on the previous shelf. Checking min(current) >= max(previous)
        # is mathematically identical to checking ALL current SKUs >= that
        # max (the smallest one clearing the bar means every larger one
        # does too) — but the remark below spells out every size and every
        # failing one explicitly, so it's never ambiguous which SKUs were
        # actually checked.
        cond1_pass = False
        if not prev_items:
            cond1_remark = "condition1: previous shelf has no valid SKU detections"
        elif not cur_items:
            cond1_remark = "condition1: current shelf has no valid SKU detections"
        else:
            prev_max = max(it['size'] for it in prev_items)
            cur_sizes = sorted(it['size'] for it in cur_items)
            if cur_sizes[0] >= prev_max:
                cond1_pass = True
                cond1_remark = ''
            else:
                failing = [s for s in cur_sizes if not (s >= prev_max)]
                cond1_remark = (
                    f"condition1: not all SKUs on this shelf are >= previous shelf "
                    f"max {prev_max}ml — this shelf's sizes {cur_sizes}, "
                    f"failing ones {failing}"
                )

        if cond1_pass:
            results[i] = ('Y', '')
            continue

        # ── Condition 2: left-to-right on this shelf alone ────────────────
        cond2_pass, cond2_remark = _condition2_left_to_right(cur_items)

        if cond2_pass:
            results[i] = ('Y', '')
        else:
            results[i] = ('N', f"{cond1_remark}; {cond2_remark}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DB access
# ─────────────────────────────────────────────────────────────────────────────

@retry_on_network_error(max_retries=3, delay=5)
def ensure_table_exists(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orgi.shelfsequencecompliance (
            iterationid       int4        NOT NULL,
            iterationtranid   int4        NOT NULL,
            storeid           int8        NOT NULL,
            imagefilename     varchar     NULL,
            shelfposition     int4        NULL,
            maxshelfnumber    int4        NULL,
            skusconsidered    int4        NULL,
            compliance_flag   bpchar(1)   NULL,
            remarks           varchar     NULL,
            checked_at        timestamp   NOT NULL DEFAULT now(),
            CONSTRAINT shelfsequencecompliance_pkey PRIMARY KEY (iterationid, iterationtranid)
        )
    """)
    # For installs created before `shelfposition` existed.
    cur.execute("""
        ALTER TABLE orgi.shelfsequencecompliance
        ADD COLUMN IF NOT EXISTS shelfposition int4
    """)
    # For installs created before `purity_status` existed. Populated from
    # temp.purity_status_temp (written by visicooler.py's GroundingDINO
    # Case 1/2/3 purity check) via fetch_purity_status_map() below.
    cur.execute("""
        ALTER TABLE orgi.shelfsequencecompliance
        ADD COLUMN IF NOT EXISTS purity_status varchar(20)
    """)
    # For installs created before `groundingdino_annotated_image` existed —
    # S3 key of the GroundingDINO debug image (boxes + PURE/IMPURE banner)
    # for Case-3 images. NULL when GroundingDINO was skipped (Case 1/2) or
    # the annotated image couldn't be built/saved/uploaded.
    cur.execute("""
        ALTER TABLE orgi.shelfsequencecompliance
        ADD COLUMN IF NOT EXISTS groundingdino_annotated_image varchar(500)
    """)
    logger.info("Verified orgi.shelfsequencecompliance exists")


def ensure_dino_detections_table(cur) -> None:
    """orgi.dino_nonbeverage_detections — one row per GroundingDINO
    detection that did NOT overlap a beverage box (both rescued-by-
    known-cap and genuine impurities), copied over from
    temp.dino_nonbeverage_detections_temp once iterationtranid is known.
    Requested columns: iterationid, iterationtranid, image name, product
    name, annotated image (storeid included as a useful extra)."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orgi.dino_nonbeverage_detections (
            id                SERIAL PRIMARY KEY,
            iterationid       int4        NOT NULL,
            iterationtranid   int4        NOT NULL,
            storeid           int8        NULL,
            imagefilename     varchar     NULL,
            productname       varchar     NULL,
            annotatedimage    varchar(500) NULL,
            detected_at       timestamp   NOT NULL DEFAULT now()
        )
    """)
    logger.info("Verified orgi.dino_nonbeverage_detections exists")


@retry_on_network_error(max_retries=3, delay=5)
def fetch_purity_status_map(cur, conn=None, iterationid=None) -> dict:
    """
    Returns {(iterationid, imagefilename): (purity_status, groundingdino_annotated_image)}
    sourced from temp.purity_status_temp, which visicooler.py populates
    per-image during the Case 1/2/3 GroundingDINO purity check.
    groundingdino_annotated_image is None whenever DINO was skipped or the
    annotated image couldn't be built/saved.

    Best-effort: if the temp table doesn't exist yet (e.g. visicooler
    hasn't run in this environment), logs a warning, rolls back the
    aborted transaction (so it doesn't poison later queries), and
    returns {} rather than failing the whole compliance check.
    """
    try:
        query = (
            "SELECT iteration_id, image_file_name, purity_status, "
            "groundingdino_annotated_image FROM temp.purity_status_temp"
        )
        params = ()
        if iterationid is not None:
            query += " WHERE iteration_id = %s"
            params = (iterationid,)
        cur.execute(query, params)
        rows = cur.fetchall()
        return {(iid, fname): (status, dino_img) for iid, fname, status, dino_img in rows}
    except Exception as e:
        logger.warning(f"Could not fetch purity_status map (temp table missing?): {e}")
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        return {}


@retry_on_network_error(max_retries=3, delay=5)
def fetch_dino_detections_temp(cur, conn=None, iterationid=None) -> dict:
    """
    Returns {(iterationid, imagefilename): [(store_id, product_name, annotated_image), ...]}
    sourced from temp.dino_nonbeverage_detections_temp — an image can have
    several rows (one per detected object).

    Best-effort: if the temp table doesn't exist yet, logs a warning,
    rolls back the aborted transaction, and returns {} rather than
    failing the whole compliance check.
    """
    try:
        query = (
            "SELECT iteration_id, image_file_name, store_id, product_name, "
            "annotated_image FROM temp.dino_nonbeverage_detections_temp"
        )
        params = ()
        if iterationid is not None:
            query += " WHERE iteration_id = %s"
            params = (iterationid,)
        cur.execute(query, params)
        rows = cur.fetchall()
        detections_map = {}
        for iid, fname, store_id, product_name, annotated_image in rows:
            detections_map.setdefault((iid, fname), []).append(
                (store_id, product_name, annotated_image)
            )
        return detections_map
    except Exception as e:
        logger.warning(f"Could not fetch dino_nonbeverage_detections_temp (missing?): {e}")
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        return {}


@retry_on_network_error(max_retries=3, delay=5)
def insert_dino_detections_final(cur, records: list) -> int:
    """
    records: list of (iterationid, iterationtranid, storeid, imagefilename,
                       productname, annotatedimage)
    Straight insert (not upsert) — this table is an append-only audit log,
    re-running the same iteration would otherwise need de-dup logic that
    isn't meaningful for a detection-count log.
    """
    if not records:
        return 0
    insert_sql = """
        INSERT INTO orgi.dino_nonbeverage_detections
            (iterationid, iterationtranid, storeid, imagefilename,
             productname, annotatedimage, detected_at)
        VALUES (%s, %s, %s, %s, %s, %s, now())
    """
    cur.executemany(insert_sql, records)
    logger.info(f"Inserted {len(records)} row(s) into orgi.dino_nonbeverage_detections")
    return len(records)


@retry_on_network_error(max_retries=3, delay=5)
def fetch_shelf_detections(cur, iterationid=None,
                            master_table="orgi.coolermetricsmaster",
                            transaction_table="orgi.coolermetricstransaction"):
    """
    Returns rows of (iterationid, iterationtranid, storeid, caserid,
    imagefilename, productclassid, x1, y1, x2, y2) — one row per detection,
    LEFT JOINed from `master_table` so that images with zero detections
    still show up (with productclassid/imagefilename/bbox NULL) and count
    toward the cooler's shelf total.

    `master_table`/`transaction_table` default to the regular 605-flow
    tables; pass the *_planogram equivalents to evaluate the subcategory-603
    planogram detections instead (see run_shelf_sequence_check_planogram).
    """
    query = f"""
        SELECT m.iterationid, m.iterationtranid, m.storeid, m.caserid,
               t.imagefilename, t.productclassid,
               t.x1, t.y1, t.x2, t.y2
        FROM {master_table} m
        LEFT JOIN {transaction_table} t
          ON t.iterationid = m.iterationid AND t.iterationtranid = m.iterationtranid
    """
    params = ()
    if iterationid is not None:
        query += " WHERE m.iterationid = %s"
        params = (iterationid,)
    query += " ORDER BY m.iterationid, m.storeid, m.caserid, m.iterationtranid"

    cur.execute(query, params)
    return cur.fetchall()


@retry_on_network_error(max_retries=3, delay=5)
def upsert_compliance_results(cur, records):
    """records: list of tuples matching the INSERT column order below."""
    if not records:
        return
    if not verify_connection(cur):
        raise Exception("Database connection lost before compliance upsert")

    query = """
        INSERT INTO orgi.shelfsequencecompliance
            (iterationid, iterationtranid, storeid, imagefilename,
             shelfposition, maxshelfnumber, skusconsidered,
             compliance_flag, remarks, purity_status,
             groundingdino_annotated_image, checked_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (iterationid, iterationtranid) DO UPDATE SET
            storeid                        = EXCLUDED.storeid,
            imagefilename                  = EXCLUDED.imagefilename,
            shelfposition                  = EXCLUDED.shelfposition,
            maxshelfnumber                 = EXCLUDED.maxshelfnumber,
            skusconsidered                 = EXCLUDED.skusconsidered,
            compliance_flag                = EXCLUDED.compliance_flag,
            remarks                        = EXCLUDED.remarks,
            purity_status                  = EXCLUDED.purity_status,
            groundingdino_annotated_image  = EXCLUDED.groundingdino_annotated_image,
            checked_at                     = now()
    """
    cur.executemany(query, records)
    logger.info(f"Upserted {len(records)} shelf-sequence compliance rows")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _most_common(values):
    values = [v for v in values if v]
    if not values:
        return None
    return max(set(values), key=values.count)


def run_shelf_sequence_check(db_config, iterationid=None,
                              master_table="orgi.coolermetricsmaster",
                              transaction_table="orgi.coolermetricstransaction"):
    """
    Main entry point. Connects to the DB, evaluates every store cooler's
    shelf ordering (grouped by iterationid+storeid+caserid, ranked by
    iterationtranid), and upserts the results into
    orgi.shelfsequencecompliance.

    `master_table`/`transaction_table` default to the regular 605-flow
    tables. Pass the *_planogram equivalents (see
    run_shelf_sequence_check_planogram below) to evaluate subcategory-603
    planogram data instead — both flows share iterationid but their
    iterationtranid ranges are kept non-overlapping upstream (see
    cap_pipeline_runner._build_planogram_queries), so both can safely
    upsert into the same orgi.shelfsequencecompliance table.

    Returns the list of (iterationid, iterationtranid, storeid,
    compliance_flag) evaluated, for logging/testing.
    """
    conn, cur = initialize_db_connection(db_config)
    try:
        ensure_table_exists(cur)
        ensure_dino_detections_table(cur)
        conn.commit()

        product_maps = load_product_maps(cur)
        rows = fetch_shelf_detections(
            cur, iterationid=iterationid,
            master_table=master_table, transaction_table=transaction_table,
        )
        logger.info(f"Fetched {len(rows)} rows to evaluate ({master_table})")

        # purity_status per (iterationid, imagefilename), written by
        # visicooler.py's GroundingDINO Case 1/2/3 purity check.
        purity_map = fetch_purity_status_map(cur, conn=conn, iterationid=iterationid)
        logger.info(f"Fetched {len(purity_map)} purity_status entries")

        # Non-beverage GroundingDINO detections per (iterationid,
        # imagefilename) — written by visicooler.py, copied into
        # orgi.dino_nonbeverage_detections below once iterationtranid is
        # resolved for each image.
        dino_detections_map = fetch_dino_detections_temp(cur, conn=conn, iterationid=iterationid)
        logger.info(f"Fetched DINO detections for {len(dino_detections_map)} image(s)")

        # Real-detection frequency across this batch, used to re-rank each
        # brand's Step-4 fallback priority list so an unseen/rare catalog
        # SKU (e.g. an uncommon "175ml Glass" variant) can't become the
        # default target just for being the smallest size on paper — see
        # reorder_catalog_by_frequency().
        classid_frequency = {}
        for row in rows:
            productclassid = row[5]
            if productclassid is not None:
                cid = int(productclassid)
                classid_frequency[cid] = classid_frequency.get(cid, 0) + 1
        reorder_catalog_by_frequency(product_maps, classid_frequency)

        # Step 1: group detections by (iterationid, iterationtranid) so we
        # have one bucket of RAW (unresolved) items + imagefilename per
        # image. Resolution to a final size happens per-image below, via
        # redistribute_unidentified_bottles(), so that unidentified-bucket
        # detections can be redistributed against their own image's real
        # SKUs rather than resolved in isolation row-by-row.
        image_data = {}
        for iid, itranid, storeid, caserid, imagefilename, productclassid, x1, y1, x2, y2 in rows:
            key = (iid, itranid)
            g = image_data.setdefault(key, {
                'storeid': storeid,
                'caserid': caserid,
                'imagefilenames': [],
                'raw_items': [],
            })
            if imagefilename:
                g['imagefilenames'].append(imagefilename)
            if productclassid is not None:
                g['raw_items'].append({
                    'classid': int(productclassid),
                    'x1': float(x1) if x1 is not None else None,
                    'y1': float(y1) if y1 is not None else None,
                    'x2': float(x2) if x2 is not None else None,
                    'y2': float(y2) if y2 is not None else None,
                })

        for g in image_data.values():
            g['items'] = redistribute_unidentified_bottles(g['raw_items'], product_maps)

        # Step 2: group those images by (iterationid, storeid, caserid) —
        # one physical cooler — and rank by iterationtranid ascending.
        cooler_groups = {}
        for (iid, itranid), g in image_data.items():
            cooler_key = (iid, g['storeid'], g['caserid'])
            cooler_groups.setdefault(cooler_key, []).append((itranid, g))

        results = []
        upsert_records = []
        dino_final_records = []
        for cooler_key, images in cooler_groups.items():
            iid, storeid, caserid = cooler_key
            images.sort(key=lambda x: x[0])  # rank by iterationtranid ascending

            items_by_rank = [g['items'] for _itranid, g in images]
            verdicts = evaluate_store_shelf_sequence(items_by_rank)
            total_shelves = len(images)

            for rank, ((itranid, g), (flag, remarks)) in enumerate(zip(images, verdicts), start=1):
                imagefilename = _most_common(g['imagefilenames'])
                skus_considered = len(g['items'])
                purity_status, dino_annotated_image = purity_map.get(
                    (iid, imagefilename), (None, None)
                )

                results.append((iid, itranid, storeid, flag))
                upsert_records.append((
                    iid, itranid, storeid, imagefilename,
                    rank, total_shelves, skus_considered,
                    flag, remarks[:500] if remarks else None,
                    purity_status, dino_annotated_image,
                ))

                # Now that iterationtranid is known for this image, resolve
                # any non-beverage DINO detections logged against it and
                # queue them for orgi.dino_nonbeverage_detections.
                for det_storeid, product_name, annotated_image in dino_detections_map.get(
                    (iid, imagefilename), []
                ):
                    dino_final_records.append((
                        iid, itranid, det_storeid or storeid, imagefilename,
                        product_name, annotated_image,
                    ))

        upsert_compliance_results(cur, upsert_records)
        if dino_final_records:
            insert_dino_detections_final(cur, dino_final_records)
        conn.commit()
        logger.info(f"Shelf-sequence check complete: {len(results)} images evaluated")
        return results

    except Exception as e:
        logger.error(f"Shelf-sequence check failed: {e}")
        conn.rollback()
        raise
    finally:
        close_db_connection(conn, cur)


def run_shelf_sequence_check_planogram(db_config, iterationid=None):
    """
    Thin wrapper: evaluates the subcategory-603 planogram shelves instead
    of the regular 605 ones, reading from orgi.coolermetricsmaster_planogram
    / orgi.coolermetricstransaction_planogram and upserting into the same
    orgi.shelfsequencecompliance table used by run_shelf_sequence_check.

    Call this AFTER run_shelf_sequence_check() for the same iterationid so
    the 605-flow's shelves are already in orgi.shelfsequencecompliance —
    not that it matters for correctness (iterationtranid ranges never
    overlap between the two flows), but it keeps the two checks' log output
    in a sensible order.
    """
    return run_shelf_sequence_check(
        db_config,
        iterationid=iterationid,
        master_table="orgi.coolermetricsmaster_planogram",
        transaction_table="orgi.coolermetricstransaction_planogram",
    )


if __name__ == "__main__":
    # Lightweight standalone run: reads db_config straight out of config.json
    # (no prompt-file dependency, unlike app.config_loader.load_config).
    config_path = os.environ.get("CONFIG_PATH", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    run_shelf_sequence_check(cfg["db_config"])