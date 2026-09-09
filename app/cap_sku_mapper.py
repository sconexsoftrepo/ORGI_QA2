from __future__ import annotations
"""
cap_sku_mapper.py  –  v4
========================
Changes vs v3
─────────────
1. Brand mapping  – unified with visicooler.py; handles sub-brands
   (Coca-Cola Zero, Kinley Soda, Kinley Water, Mountain Dew …), colour-cap
   synonyms, and "Other *" passthrough.

2. 1-cap-per-SKU enforcement  – P1/P2/P3 now HARD-BLOCK an already-claimed
   SKU slot instead of silently reusing it.  If no free slot is available the
   cap falls through to the next phase.  Reuse is only allowed at a dedicated
   "reuse" sub-phase (P1r / P2r / P3r) so it is visible in stats and logs but
   the algorithm first exhausts all cheaper options.

3. Pack-size mapping  –  new helper `extract_pack_size()` returns the ml/L
   volume string from the class name.  When brand + pkg match but multiple
   SKUs qualify, the one whose pack size matches the cap's inferred size is
   preferred.  Pack-size inference on a cap is intentionally loose (the cap
   class name rarely carries ml) so it is used only as a tiebreaker, never as
   a hard filter.

4. Duplicate-insertion guard  –  `insert_mapped_cap_records` now builds an
   in-memory seen-set keyed on the DB unique-index columns so the same
   (store_id, image_file_name, iteration_id, cap_class_id, x1, y1) row is
   never emitted twice even if map_caps_to_skus is called more than once per
   run.

Architecture reminder
─────────────────────
• One cap  →  one SKU slot  (1-cap-1-bottle).
• If a bottle has N caps detected above it the SKU slot is claimed by the
  nearest cap first; subsequent caps for that image fall through to the next
  phase (P2 → P3 → P4) rather than being silently pinned to the same bottle.
• P4 is the safety net: brand+pkg synthetic name so Step 3 SQL can still group
  them sensibly.
"""
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

# ── Tuneable thresholds ───────────────────────────────────────────────────────
MAX_MATCH_DIST_P2: float   = 500.0   # pixels – P2 brand+pkg nearest (no x-overlap)
MAX_MATCH_DIST_P3: float   = 350.0   # pixels – P3 brand-only fallback
MAX_MATCH_DIST_P2R: float  = 700.0   # pixels – P2-reuse (already-claimed slot)
MAX_MATCH_DIST_P3R: float  = 500.0   # pixels – P3-reuse
IOU_DEDUP_THRESHOLD: float = 0.5    # IoU above which two boxes are duplicates


# ── Brand normalisation ───────────────────────────────────────────────────────
# Ordered: longer / more-specific patterns must come before shorter ones.
_BRAND_RULES: list[tuple[list[str], str]] = [
    # sub-brands first
    (["coca-cola zero", "coca cola zero", "coke zero"],   "coca-cola zero"),
    (["coca-cola", "coca cola", "coke", "coca", "cola"],  "coca-cola"),
    (["mountain dew", "mountian dew"],                    "mountain dew"),
    (["kinley soda"],                                     "kinley soda"),
    (["kinley water"],                                    "kinley water"),
    (["kinley"],                                          "kinley"),
    (["thums up", "thumsup", "thumbs up"],                "thums up"),
    (["sprite"],   "sprite"),
    (["fanta"],    "fanta"),
    (["limca"],    "limca"),
    (["maaza"],    "maaza"),
    (["pepsi"],    "pepsi"),
    (["mirinda"],  "mirinda"),
    (["7up", "7-up", "7 up"], "7up"),
    (["slice"],    "slice"),
    (["sting"],    "sting"),
    (["aquafina"], "aquafina"),
]

# Colour-cap → brand mapping (caps detected without explicit brand text)
_CAP_COLOUR_BRAND: dict[str, str] = {
    "coke red cap_1labove": "coca-cola",   # >=1000ml large-PET – must be before "coke red cap"
    "coke red cap":    "coca-cola",
    "coke black cap":  "coca-cola zero",
    "sprite green cap":"sprite",
    "sprite blue cap": "sprite",
    "fanta orange cap":"fanta",
    "fanta blue cap":  "fanta",
    "kinley bottle cap":"kinley water",
    "kinley soda cap": "kinley soda",
    "kinley glass cap":"kinley",
    "thumsup black cap":"thums up",
    "coke glass cap":  "coca-cola",
    "coke glass capss":"coca-cola",
    "sprite glass cap":"sprite",
    "fanta glass cap": "fanta",
    "limca cap":       "limca",
    "maaza cap":       "maaza",
}

# Colour-cap → package type override
_CAP_COLOUR_PKG: dict[str, str] = {
    "coke red cap_1labove": "pet",         # >=1000ml large-PET – must be before "coke red cap"
    "coke red cap":     "pet",
    "coke black cap":   "pet",
    "sprite green cap": "pet",
    "sprite blue cap":  "pet",
    "fanta orange cap": "pet",
    "fanta blue cap":   "pet",
    "kinley bottle cap":"pet",
    "kinley soda cap":  "soda",
    "kinley glass cap": "glass",
    "thumsup black cap":"pet",
    "coke glass cap":   "glass",
    "coke glass capss": "glass",
    "sprite glass cap": "glass",
    "fanta glass cap":  "glass",
}


def extract_brand(name: str) -> str:
    """Return canonical brand string, or '' for unrecognised / 'Other' names."""
    nl = name.lower().strip()

    # Colour-cap shortcut
    for cap_key, brand in _CAP_COLOUR_BRAND.items():
        if cap_key in nl:
            return brand

    # Generic rule scan
    for patterns, canonical in _BRAND_RULES:
        for p in patterns:
            if p in nl:
                return canonical

    return ""


# ── Package-type normalisation ────────────────────────────────────────────────
def extract_package_type(name: str) -> str:
    """Return: 'glass' | 'pet' | 'can' | 'soda' | 'water' | ''"""
    nl = name.lower()

    # Colour-cap shortcut
    for cap_key, pkg in _CAP_COLOUR_PKG.items():
        if cap_key in nl:
            return pkg

    if "glass" in nl:
        return "glass"
    if "pet" in nl or "bottle" in nl:
        return "pet"
    if "can" in nl:
        return "can"
    if "soda" in nl:
        return "soda"
    if "water" in nl:
        return "water"
    # Red/green/black cap → PET (most common)
    for colour in ("red", "green", "orange", "black", "blue"):
        if colour in nl and "cap" in nl:
            return "pet"
    return ""


# ── Pack-size extraction ──────────────────────────────────────────────────────
import re as _re

_SIZE_RE = _re.compile(r'(\d[\d.]*)\s*(ml|l)\b', _re.IGNORECASE)


def extract_pack_size(name: str) -> str:
    """
    Return normalised size string like '250ml', '1.5l', etc., or ''.
    Normalises 'L' → 'l', strips spaces between number and unit.
    """
    m = _SIZE_RE.search(name)
    if not m:
        return ""
    num, unit = m.group(1), m.group(2).lower()
    return f"{num}{unit}"


# ── Geometry helpers ──────────────────────────────────────────────────────────
def _centroid(r: dict) -> tuple[float, float]:
    return ((r["x1"] + r["x2"]) / 2.0, (r["y1"] + r["y2"]) / 2.0)


def _dist(a: dict, b: dict) -> float:
    ax, ay = _centroid(a)
    bx, by = _centroid(b)
    return math.hypot(ax - bx, ay - by)


def _area(r: dict) -> float:
    return max(0.0, r["x2"] - r["x1"]) * max(0.0, r["y2"] - r["y1"])


def _iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _x_overlap(cap: dict, sku: dict) -> bool:
    """True if the cap's horizontal centre falls within the SKU's x-span."""
    cx = (cap["x1"] + cap["x2"]) / 2.0
    return sku["x1"] <= cx <= sku["x2"]


def _pkg_compat(cap_pkg: str, sku_pkg: str) -> bool:
    """Compatible when either side is unknown or they are equal."""
    return cap_pkg == "" or sku_pkg == "" or cap_pkg == sku_pkg


# ── IoU-based deduplication ───────────────────────────────────────────────────
def _dedup_by_iou(records: list, class_key: str) -> list:
    """
    Remove duplicate detections (IoU >= IOU_DEDUP_THRESHOLD) within the same
    (store_id, image_file_name, iteration_id, class_id) group.
    Keeps highest-confidence detection; falls back to largest area.
    """
    if not records:
        return records

    groups: dict = defaultdict(list)
    for r in records:
        key = (r["store_id"], r["image_file_name"], r["iteration_id"], r[class_key])
        groups[key].append(r)

    kept = []
    total_removed = 0

    for _key, group in groups.items():
        group_sorted = sorted(
            group,
            key=lambda r: (-(r.get("confidence") or 0.0), -_area(r)),
        )
        accepted: list = []
        for candidate in group_sorted:
            if any(_iou(candidate, keeper) >= IOU_DEDUP_THRESHOLD for keeper in accepted):
                total_removed += 1
            else:
                accepted.append(candidate)
        kept.extend(accepted)

    if total_removed:
        logger.info(
            "IoU dedup removed %d duplicate(s) (iou_threshold=%.2f)",
            total_removed, IOU_DEDUP_THRESHOLD,
        )
    return kept


# ── Nearest-SKU selector (free-only or reuse-only) ────────────────────────────
def _best_free(
    cap: dict,
    candidates: list,
    used: set,
    max_dist: float,
) -> dict | None:
    """
    Return the nearest *unclaimed* SKU within max_dist.
    Prefers pack-size match as tiebreaker.
    Returns None if nothing qualifies.
    """
    cap_size = cap.get("_size", "")
    within = [s for s in candidates if id(s) not in used and _dist(cap, s) <= max_dist]
    if not within:
        return None
    # Tiebreaker: same pack size → lower sort key
    def _sort_key(s):
        size_match = 0 if (cap_size and s.get("_size") == cap_size) else 1
        return (size_match, _dist(cap, s))
    return min(within, key=_sort_key)


def _best_reuse(
    cap: dict,
    candidates: list,
    used: set,
    max_dist: float,
) -> dict | None:
    """
    Return the nearest *already-claimed* SKU within max_dist.
    Used only when no free slot is available.
    """
    within = [s for s in candidates if id(s) in used and _dist(cap, s) <= max_dist]
    if not within:
        return None
    return min(within, key=lambda s: _dist(cap, s))


# ── Core mapping ──────────────────────────────────────────────────────────────
def map_caps_to_skus(
    cap_records: list,
    sku_records: list,
    cap_class_names: dict,
    sku_class_names: dict,
) -> list:
    """
    Map every cap detection to one SKU prod_class_id.

    Phases (strict 1-cap-per-SKU):
        P1   – x-overlap + brand + compatible pkg  (no distance cap, FREE slot)
        P1r  – same geometry rules but allows REUSE of already-claimed slot
        P2   – nearest brand+pkg within MAX_MATCH_DIST_P2 (FREE)
        P2r  – same, REUSE within MAX_MATCH_DIST_P2R
        P3   – nearest brand-only within MAX_MATCH_DIST_P3 (FREE)
        P3r  – same, REUSE within MAX_MATCH_DIST_P3R
        P4   – synthetic fallback (prod_class_id = -1)

    "1-cap-per-SKU" means: a free slot is always preferred.  Reuse only
    happens if there is genuinely no free slot in the relevant pool, i.e. the
    image has more caps than front-facing bottles of that brand/size.
    """

    # ── Step 0: IoU dedup ────────────────────────────────────────────────────
    before_cap, before_sku = len(cap_records), len(sku_records)
    cap_records = _dedup_by_iou(cap_records, "cap_class_id")
    sku_records  = _dedup_by_iou(sku_records,  "prod_class_id")
    logger.info(
        "After IoU dedup — caps: %d→%d, SKUs: %d→%d",
        before_cap, len(cap_records), before_sku, len(sku_records),
    )

    # ── Step 1: Annotate brand / pkg / size ──────────────────────────────────
    for c in cap_records:
        name = cap_class_names.get(c["cap_class_id"], "")
        c["_brand"]   = extract_brand(name)
        c["_pkg"]     = extract_package_type(name)
        c["_size"]    = extract_pack_size(name)   # usually '' for cap names
        c["_capname"] = name

    for s in sku_records:
        name = sku_class_names.get(s["prod_class_id"], "")
        s["_brand"] = extract_brand(name)
        s["_pkg"]   = extract_package_type(name)
        s["_size"]  = extract_pack_size(name)     # e.g. '250ml', '1.5l'
        s["_name"]  = name

    # ── Step 2: Index SKUs per image ─────────────────────────────────────────
    sku_index: dict = defaultdict(list)
    for s in sku_records:
        img_key = (s["store_id"], s["image_file_name"], s["iteration_id"])
        sku_index[img_key].append(s)

    # ── Step 3: Per-image occupied set (id(sku) already claimed) ─────────────
    occupied: dict = defaultdict(set)

    stats = {
        "p1": 0, "p1r": 0,
        "p2": 0, "p2r": 0,
        "p3": 0, "p3r": 0,
        "p4": 0,
        "p2_od": 0, "p3_od": 0,
    }

    # ── Step 4: Map each cap ─────────────────────────────────────────────────
    for cap in cap_records:
        img_key  = (cap["store_id"], cap["image_file_name"], cap["iteration_id"])
        all_skus = sku_index.get(img_key, [])
        used     = occupied[img_key]
        brand    = cap["_brand"]
        pkg      = cap["_pkg"]

        # Unknown / Other brand → straight to P4 (no random assignment)
        if not brand or brand == "other":
            _assign_p4(cap, stats)
            continue

        # ── P1: x-overlap + brand + compatible pkg (FREE slot) ───────────────
        p1_pool = [
            s for s in all_skus
            if s["_brand"] == brand
            and _pkg_compat(pkg, s["_pkg"])
            and _x_overlap(cap, s)
        ]
        best = _best_free(cap, p1_pool, used, float("inf"))
        if best is not None:
            _assign(cap, best, used, stats, "p1")
            logger.debug("[P1] %s → %s dist=%.0fpx", cap["_capname"], best["_name"], _dist(cap, best))
            continue

        # ── P1r: x-overlap + brand + compatible pkg (REUSE allowed) ──────────
        best = _best_reuse(cap, p1_pool, used, float("inf"))
        if best is not None:
            _assign(cap, best, used, stats, "p1r")
            logger.debug("[P1r-reuse] %s → %s dist=%.0fpx", cap["_capname"], best["_name"], _dist(cap, best))
            continue

        # ── P2: nearest brand+pkg within MAX_MATCH_DIST_P2 (FREE) ────────────
        p2_pool = [
            s for s in all_skus
            if s["_brand"] == brand and _pkg_compat(pkg, s["_pkg"])
        ]
        best = _best_free(cap, p2_pool, used, MAX_MATCH_DIST_P2)
        if best is not None:
            _assign(cap, best, used, stats, "p2")
            logger.debug("[P2] %s → %s dist=%.0fpx", cap["_capname"], best["_name"], _dist(cap, best))
            continue

        # ── P2r: same pool but REUSE within MAX_MATCH_DIST_P2R ───────────────
        best = _best_reuse(cap, p2_pool, used, MAX_MATCH_DIST_P2R)
        if best is not None:
            _assign(cap, best, used, stats, "p2r")
            logger.debug("[P2r-reuse] %s → %s dist=%.0fpx", cap["_capname"], best["_name"], _dist(cap, best))
            continue

        # Log over-distance warning for P2
        if p2_pool:
            stats["p2_od"] += 1
            nd = min(_dist(cap, s) for s in p2_pool)
            logger.warning(
                "[P2-over-dist] %s brand=%s pkg=%s: nearest=%.0fpx > limit=%.0fpx → P3",
                cap["_capname"], brand, pkg, nd, MAX_MATCH_DIST_P2R,
            )

        # ── P3: nearest brand-only within MAX_MATCH_DIST_P3 (FREE) ───────────
        p3_pool = [s for s in all_skus if s["_brand"] == brand]
        best = _best_free(cap, p3_pool, used, MAX_MATCH_DIST_P3)
        if best is not None:
            _assign(cap, best, used, stats, "p3")
            logger.debug("[P3] %s → %s dist=%.0fpx", cap["_capname"], best["_name"], _dist(cap, best))
            continue

        # ── P3r: same pool but REUSE within MAX_MATCH_DIST_P3R ───────────────
        best = _best_reuse(cap, p3_pool, used, MAX_MATCH_DIST_P3R)
        if best is not None:
            _assign(cap, best, used, stats, "p3r")
            logger.warning("[P3r-reuse] %s → %s (pkg mismatch OK; dist=%.0fpx)",
                           cap["_capname"], best["_name"], _dist(cap, best))
            continue

        # Log over-distance warning for P3
        if p3_pool:
            stats["p3_od"] += 1
            nd = min(_dist(cap, s) for s in p3_pool)
            logger.warning(
                "[P3-over-dist] %s brand=%s: nearest=%.0fpx > limit=%.0fpx → P4",
                cap["_capname"], brand, nd, MAX_MATCH_DIST_P3R,
            )

        # ── P4: synthetic fallback ────────────────────────────────────────────
        _assign_p4(cap, stats)

    logger.info(
        "cap_sku_mapper v4 | caps=%d | "
        "P1=%d P1r=%d P2=%d P2r=%d(od=%d) P3=%d P3r=%d(od=%d) P4=%d",
        len(cap_records),
        stats["p1"], stats["p1r"],
        stats["p2"], stats["p2r"], stats["p2_od"],
        stats["p3"], stats["p3r"], stats["p3_od"],
        stats["p4"],
    )

    # ── Remap new-model class IDs → old-model class IDs ──────────────────────
    # After cap→SKU mapping, prod_class_id is still in the new model's ID space
    # (0–41).  We translate those back to the old model IDs using the Excel
    # class_mapping table.  IDs ≥ 9000 are special unresolved-fallback classes
    # and are always left untouched.  Any ID not found in the table is also
    # left unchanged.
    cap_records = _remap_to_old_class_ids(cap_records)

    return cap_records


# ── New-model → old-model class ID remapping ─────────────────────────────────
# Built from class_mapping_Sheet1_.xlsx  (new_classid → old_classid).
# Keys are new-model class IDs (0-41); values are the corresponding old IDs.
_NEW_TO_OLD_CLASS_MAP: dict[int, int] = {
    0:  25,  # Alcohol Bottle        → new productclassid you just inserted
    1:  0,     # Coca-Cola 1000ml PET
    2:  1,     # Coca-Cola 1500ml PET
    3:  4,  # Coca-Cola 175ml Glass → new productclassid you just inserted
    4:  2,     # Coca-Cola 2000ml PET
    5:  3,     # Coca-Cola 2250ml PET
    6:  4,     # Coca-Cola 250ml Glass
    7:  5,     # Coca-Cola 250ml PET
    8:  6,     # Coca-Cola 500ml PET
    9:  36,    # Coke Zero 250ml PET
    10: 37,    # Coke Zero 500ml PET
    11: 15,  # Fanta 175ml Glass     → new productclassid you just inserted
    12: 7,     # Fanta Lemon 1000ml
    13: 8,     # Fanta Lemon 2000ml PET
    14: 9,     # Fanta Lemon 2250ml PET
    15: 10,    # Fanta Lemon 250ml
    16: 11,    # Fanta Orange 1000ml PET
    17: 12,    # Fanta Orange 1500ml PET
    18: 13,    # Fanta Orange 2000ml PET
    19: 14,    # Fanta Orange 2250ml PET
    20: 15,    # Fanta Orange 250ml Glass
    21: 16,    # Fanta Orange 250ml PET
    22: 17,    # Fanta Orange 500ml PET
    23: 18,    # Kinley Soda 250ml Glass
    24: 19,    # Kinley Soda 250ml PET
    25: 20,    # Kinley Soda 500ml PET
    26: 21,    # Kinley Water 1000ml PET
    27: 22,    # Kinley Water 500ml PET
    28: 25,  # Other Brand Glass     → new productclassid you just inserted
    29: 23,    # Other CAN
    30: 24,    # Other PET
    31: 26,    # Other TPK
    32: 27,    # Other Water Bottle
    33: 28,    # Shelf-detection
    34: 29,    # Sprite 1000ml PET
    35: 30,    # Sprite 1500ml PET
    36: 34,  # Sprite 175ml Glass    → new productclassid you just inserted
    37: 31,    # Sprite 2000ml PET
    38: 32,    # Sprite 2250ml PET
    39: 33,    # Sprite 250ml Glass
    40: 34,    # Sprite 250ml PET
    41: 35,    # Sprite 500ml PET
}


def _remap_to_old_class_ids(cap_records: list) -> list:
    """
    Translate prod_class_id values from the new model's ID space to the old
    model's ID space using _NEW_TO_OLD_CLASS_MAP.

    Rules:
    - prod_class_id >= 9000  →  keep as-is (unresolved-fallback classes)
    - prod_class_id in map   →  replace with old_classid from map
    - prod_class_id not in map and < 9000  →  keep as-is (unknown; no mapping)
    - prod_class_id is None or -1  →  keep as-is

    Logs a one-line summary of how many IDs were remapped.
    """
    remapped = 0
    kept_fallback = 0
    kept_unmapped = 0

    for cap in cap_records:
        pid = cap.get("prod_class_id")

        # Nothing to remap for NULL / generic-P4 (-1)
        if pid is None or pid == -1:
            continue

        # IDs >= 9000 are unresolved-fallback classes — leave untouched
        if pid >= 9000:
            kept_fallback += 1
            continue

        old_id = _NEW_TO_OLD_CLASS_MAP.get(pid)
        if old_id is not None:
            logger.debug(
                "[class-remap] prod_class_id %d → %d  (%s)",
                pid, old_id, cap.get("prod_class_name", ""),
            )
            cap["prod_class_id"] = old_id
            remapped += 1
        else:
            # Not in the mapping table and < 9000 — keep original ID
            kept_unmapped += 1
            logger.debug(
                "[class-remap] prod_class_id %d not in mapping table — kept as-is",
                pid,
            )

    logger.info(
        "class-remap | remapped=%d  kept_fallback(>=9000)=%d  kept_unmapped=%d",
        remapped, kept_fallback, kept_unmapped,
    )
    return cap_records


# ── Assignment helpers ────────────────────────────────────────────────────────
def _assign(cap: dict, best: dict, used: set, stats: dict, stat_key: str) -> None:
    cap["prod_class_id"]   = best["prod_class_id"]
    cap["prod_class_name"] = best["_name"]
    # used.add(id(best))
    stats[stat_key] += 1


def _assign_p4(cap: dict, stats: dict) -> None:
    brand    = cap.get("_brand", "")
    pkg      = cap.get("_pkg", "")
    size     = cap.get("_size", "")
    capname  = cap.get("_capname", "").lower().strip()

    # ── P4 special cases: named cap triggers → fixed class_id / class_name ──────
    # More-specific triggers (glass) come before generic colour caps so
    # "coke glass cap" doesn't accidentally match a shorter brand pattern first.
    _P4_SPECIAL: dict[str, tuple[int, str]] = {
        # Coke Red Cap – size-split (more-specific _1labove MUST come before generic)
        "coke red cap_1labove": (9041, "Coca-Cola PET >=1000ml"),
        "coke red cap":         (9040, "Coca-Cola PET <1000ml"),
        # Glass bottle caps (250ml RGB glass)
        "coke glass cap":   (9010, "Coca-Cola 250ml Glass"),
        "coke glass capss": (9010, "Coca-Cola 250ml Glass"),
        "sprite glass cap": (9011, "Sprite 250ml Glass"),
        "fanta glass cap":  (9012, "Fanta 250ml Glass"),
        "kinley glass cap": (9013, "Kinley 250ml Glass"),
        # Large-PET caps (>=1000ml bottles detected cap-only)
        "sprite green cap": (9001, "Sprite PET >=1000ml"),
        "fanta orange cap": (9002, "Fanta PET >=1000ml"),
        # Small-PET caps (<=500ml bottles detected cap-only)
        "sprite blue cap":  (9042, "Sprite PET <=500ml"),
        "fanta blue cap":   (9043, "Fanta PET <=500ml"),
        # No-brand caps → always get a class_id so DB never stores NULL
        "other water cap":  (9030, "Other Water Bottle Cap"),
        "other  cap":       (9031, "Other Cap (unresolved brand)"),
        "other cap":        (9031, "Other Cap (unresolved brand)"),
    }
    for trigger, (class_id, class_name) in _P4_SPECIAL.items():
        if trigger in capname:
            cap["prod_class_id"]   = class_id
            cap["prod_class_name"] = class_name
            stats["p4"] += 1
            logger.warning(
                "[P4-special] %s → class_id=%d '%s' (no SKU match; special fallback)",
                cap["_capname"], class_id, class_name,
            )
            return

    # ── Brand-level fallback: known brand but no SKU row found in image ────────
    # Caps on front-facing bottles whose SKU row lives in sku_prediction_temp
    # but wasn't loaded into sku_records get a real class_id instead of NULL.
    _BRAND_PKG_FALLBACK: dict[tuple[str, str], tuple[int, str]] = {
        ("coca-cola",      "pet"):   (9020, "Coca-Cola PET (unresolved size)"),
        ("coca-cola",      "glass"): (9010, "Coca-Cola 250ml Glass"),
        ("coca-cola",      "can"):   (9021, "Coca-Cola Can (unresolved size)"),
        ("coca-cola",      ""):      (9020, "Coca-Cola PET (unresolved size)"),
        ("coca-cola zero", "pet"):   (9022, "Coca-Cola Zero PET (unresolved size)"),
        ("coca-cola zero", ""):      (9022, "Coca-Cola Zero PET (unresolved size)"),
        ("sprite",         "pet"):   (9023, "Sprite PET (unresolved size)"),
        ("sprite",         "glass"): (9011, "Sprite 250ml Glass"),
        ("sprite",         ""):      (9023, "Sprite PET (unresolved size)"),
        ("fanta",          "pet"):   (9024, "Fanta PET (unresolved size)"),
        ("fanta",          "glass"): (9012, "Fanta 250ml Glass"),
        ("fanta",          ""):      (9024, "Fanta PET (unresolved size)"),
        ("thums up",       "pet"):   (9025, "Thums Up PET (unresolved size)"),
        ("thums up",       ""):      (9025, "Thums Up PET (unresolved size)"),
        ("limca",          "pet"):   (9026, "Limca PET (unresolved size)"),
        ("limca",          ""):      (9026, "Limca PET (unresolved size)"),
        ("maaza",          "pet"):   (9027, "Maaza PET (unresolved size)"),
        ("maaza",          ""):      (9027, "Maaza PET (unresolved size)"),
        ("kinley",         "pet"):   (9028, "Kinley Water PET (unresolved size)"),
        ("kinley",         "glass"): (9013, "Kinley 250ml Glass"),
        ("kinley water",   ""):      (9028, "Kinley Water PET (unresolved size)"),
        ("kinley soda",    ""):      (9029, "Kinley Soda (unresolved size)"),
    }
    if brand:
        fallback = _BRAND_PKG_FALLBACK.get((brand, pkg)) or _BRAND_PKG_FALLBACK.get((brand, ""))
        if fallback:
            class_id, class_name = fallback
            cap["prod_class_id"]   = class_id
            cap["prod_class_name"] = class_name
            stats["p4"] += 1
            logger.warning(
                "[P4-brand-fallback] %s → class_id=%d '%s' (brand='%s', pkg='%s'; no SKU row matched)",
                cap["_capname"], class_id, class_name, brand, pkg,
            )
            return

    # ── Generic P4 fallback (unknown brand) ──────────────────────────────────
    parts = [p for p in [brand, pkg, size] if p]
    synthetic = " ".join(parts) if parts else cap["_capname"]
    cap["prod_class_id"]   = -1
    cap["prod_class_name"] = synthetic
    stats["p4"] += 1
    logger.warning(
        "[P4-synthetic] %s → '%s' (no SKU match; brand='%s')",
        cap["_capname"], synthetic, brand,
    )


# ── DB insertion ──────────────────────────────────────────────────────────────
def insert_mapped_cap_records(
    cur,
    cap_records: list,
    conn=None,
    chunk_size: int = 50,
) -> int:
    """
    Insert mapped caps into temp.cap_prediction_temp.

    P4 records (prod_class_id == -1) are inserted with NULL prod_class_id.

    Duplicate guard: builds an in-memory seen-set on
    (store_id, image_file_name, iteration_id, cap_class_id, x1, y1)
    matching the DB unique index, so repeated calls within the same run
    never attempt to insert the same physical detection twice.
    """
    insert_sql = """
        INSERT INTO temp.cap_prediction_temp (
            store_id, image_file_name, s3path_annotated_file,
            iteration_id, cap_class_id, prod_class_id,
            x1, x2, y1, y2, shelfnumber, brand_name
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """

    seen: set = set()
    rows = []
    skipped_dup = 0

    for r in cap_records:
        dedup_key = (
            r["store_id"],
            r["image_file_name"],
            r["iteration_id"],
            r["cap_class_id"],
            r["x1"],
            r["y1"],
        )
        if dedup_key in seen:
            skipped_dup += 1
            continue
        seen.add(dedup_key)
        rows.append((
            r["store_id"],
            r["image_file_name"],
            r.get("s3path_annotated_file", ""),
            r["iteration_id"],
            r["cap_class_id"],
            # -1   → generic P4 unknown brand → NULL in DB
            # 9001+ → special P4 fallback classes (glass / large-PET / brand) → keep real ID
            r["prod_class_id"] if r["prod_class_id"] not in (-1, None) else None,
            r["x1"], r["x2"], r["y1"], r["y2"],
            r.get("shelfnumber", 0),
            r.get("_brand", ""),
        ))

    if skipped_dup:
        logger.warning("Duplicate-guard skipped %d already-seen cap rows", skipped_dup)

    inserted = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        try:
            cur.executemany(insert_sql, chunk)
            if conn:
                conn.commit()
            inserted += len(chunk)
        except Exception as exc:
            logger.error("Failed inserting cap chunk %d–%d: %s", i, i + len(chunk), exc)
            raise

    logger.info("Total cap rows inserted: %d (skipped %d duplicates)", inserted, skipped_dup)
    return inserted


# ── Tuple ↔ dict helpers ──────────────────────────────────────────────────────
CAP_TUPLE_FIELDS = [
    "store_id", "image_file_name", "iteration_id",
    "cap_class_id", "x1", "x2", "y1", "y2",
    "prod_class_id", "shelfnumber", "s3path_annotated_file",
]
SKU_TUPLE_FIELDS = [
    "store_id", "image_file_name", "iteration_id",
    "prod_class_id", "x1", "x2", "y1", "y2",
    "shelfnumber", "brand_name", "s3path_annotated_file",
]


def cap_tuples_to_dicts(tuples: list) -> list:
    return [dict(zip(CAP_TUPLE_FIELDS, t)) for t in tuples]


def sku_tuples_to_dicts(tuples: list) -> list:
    return [dict(zip(SKU_TUPLE_FIELDS, t)) for t in tuples]