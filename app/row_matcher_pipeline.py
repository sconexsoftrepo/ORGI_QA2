import math
import re
from collections import defaultdict


MAX_MATCH_DIST = 500


_BRAND_RULES = [
    (["coca-cola zero", "coca cola zero", "coke zero"], "coca-cola zero"),
    (["coca-cola", "coca cola", "coke", "coca", "cola"], "coca-cola"),
    (["sprite"], "sprite"),
    (["fanta"], "fanta"),
    (["kinley soda"], "kinley soda"),
    (["kinley water"], "kinley water"),
    (["kinley"], "kinley"),
    (["thums up", "thumsup", "thumbs up"], "thums up"),
    (["limca"], "limca"),
    (["maaza"], "maaza"),
    (["pepsi"], "pepsi"),
    (["mirinda"], "mirinda"),
    (["7up", "7-up"], "7up")
]


_CAP_BRAND = {
    "coke red cap": "coca-cola",
    "coke black cap": "coca-cola zero",
    "sprite green cap": "sprite",
    "sprite blue cap": "sprite",
    "sprite black cap": "sprite",
    "fanta orange cap": "fanta",
    "fanta blue cap": "fanta",
    "kinley bottle cap": "kinley water",
    "kinley soda cap": "kinley soda",
    "thumsup black cap": "thums up",
    "coke glass cap": "coca-cola",
    "sprite glass cap": "sprite",
    "fanta glass cap": "fanta"
}


_CAP_PACKAGE = {
    "coke red cap": "pet",
    "coke black cap": "pet",
    "sprite green cap": "pet",
    "sprite blue cap": "pet",
    "sprite black cap": "pet",
    "fanta orange cap": "pet",
    "fanta blue cap": "pet",
    "kinley bottle cap": "pet",
    "kinley soda cap": "soda",
    "coke glass cap": "glass",
    "sprite glass cap": "glass",
    "fanta glass cap": "glass"
}


def get_center(box):
    return (
        (box["x1"] + box["x2"]) / 2,
        (box["y1"] + box["y2"]) / 2
    )


def extract_brand(name):
    name_lower = name.lower()

    for key, brand in _CAP_BRAND.items():
        if key in name_lower:
            return brand

    for patterns, brand in _BRAND_RULES:
        for p in patterns:
            if p in name_lower:
                return brand

    return "other"


def extract_package(name):
    name_lower = name.lower()

    for key, pkg in _CAP_PACKAGE.items():
        if key in name_lower:
            return pkg

    if "glass" in name_lower:
        return "glass"

    if "pet" in name_lower or "bottle" in name_lower:
        return "pet"

    if "can" in name_lower:
        return "can"

    if "soda" in name_lower:
        return "soda"

    if "water" in name_lower:
        return "water"

    return ""


def distance(a, b):
    ax, ay = get_center(a)
    bx, by = get_center(b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def x_overlap(cap, sku):
    cx = (cap["x1"] + cap["x2"]) / 2
    return sku["x1"] <= cx <= sku["x2"]


def _calculate_overlap(boxA, boxB):
    """Return the pixel area of intersection between two bounding boxes."""
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    return inter_w * inter_h


CAP_COUNT_THRESHOLD = 10   # below this → trust raw detection; at/above → use structural estimate


def compute_counts(mapped_caps, sku_detections):
    """
    mapped_caps     : list of dicts, each already has "sku_name" set by
                      _build_mapped_caps (no geometry re-mapping needed here).
    sku_detections  : list of dicts built from sku_dicts, each with x1/y1/x2/y2
                      and "name" matching the sku_name values above.

    Counting strategy (per-SKU):
    ─────────────────────────────
    Case A  total_caps < CAP_COUNT_THRESHOLD (default 10)
        The cooler section is sparse enough that every cap is likely a distinct
        front-facing bottle.  Use total_caps directly as the final count.
        Depth signals are still computed and logged for transparency but do NOT
        affect the count written to the DB.

    Case B  total_caps >= CAP_COUNT_THRESHOLD
        The section is dense — there is meaningful depth to estimate.
        Five signals are computed; the highest wins (raw_depth):
          1. depth_caps        = total_caps / columns
          2. perspective_depth = 1 + (small_objects / columns)
                                 caps whose area < 75 % of avg area
          3. overlap_depth     = 1 + (overlap_pairs / columns)
                                 pairs of cap boxes with non-zero intersection
          4. cap_inside_depth  = max caps per column that are ≥ 25 % inside
                                 a real SKU bounding box
          5. raw_depth         = max of signals 1–4, with custom rounding
                                 (fractional >= 0.4 → ceil, else round)
        depth is clamped to [1, 8].
        final = columns * depth   (structural estimate, NOT max with total_caps)
    """
    sku_groups = defaultdict(list)
    for cap in mapped_caps:
        sku_groups[cap["sku_name"]].append(cap)

    final_counts = {}

    for sku in sku_groups:
        sku_boxes    = [s for s in sku_detections if s["name"] == sku]
        caps_for_sku = [c for c in mapped_caps    if c["sku_name"] == sku]

        matching_prod_id = None
        for s in sku_detections:
            if s["name"] == sku:
                matching_prod_id = s.get("prod_class_id")
                break

        total_caps = len(caps_for_sku)

        # ── Synthetic fallback — no SKU bounding box detected for this name ──
        # Always use raw cap count here regardless of threshold; there is no
        # geometry to compute columns/depth from.
        if len(sku_boxes) == 0:
            print("\n==============================")
            print(f"SKU = {sku}  [NO SKU BOX — synthetic fallback]")
            print(f"Detected Caps = {total_caps}")
            print(f"FINAL COUNT   = {total_caps}  (raw caps, no SKU geometry)")

            final_counts[sku] = {
                "count":             total_caps,
                "prod_class_id":     None,
                "detected_caps":     total_caps,
                "columns":           1,
                "depth":             1,
                "depth_caps":        float(total_caps),
                "perspective_depth": 1.0,
                "overlap_depth":     1.0,
                "cap_inside_depth":  0,
                "raw_depth":         float(total_caps),
                "count_mode":        "raw_caps_no_sku_box",
            }
            continue

        # ------------------------------------------------------------------
        # SKU geometry → column count (used in both Case A and B for logging)
        # ------------------------------------------------------------------
        widths  = [s["x2"] - s["x1"] for s in sku_boxes]
        heights = [s["y2"] - s["y1"] for s in sku_boxes]  # noqa: F841

        avg_w = sum(widths) / len(widths)

        min_x = min(s["x1"] for s in sku_boxes)
        max_x = max(s["x2"] for s in sku_boxes)
        shelf_width = max_x - min_x

        columns = len(sku_boxes)
        columns = max(1, columns)

        # ------------------------------------------------------------------
        # Signal 1 — caps-based depth
        # ------------------------------------------------------------------
        depth_caps = total_caps / columns if columns > 0 else total_caps

        # ------------------------------------------------------------------
        # Signal 2 — perspective depth
        # ------------------------------------------------------------------
        cap_widths  = [c["x2"] - c["x1"] for c in caps_for_sku]
        cap_heights = [c["y2"] - c["y1"] for c in caps_for_sku]

        avg_cap_area = (
            sum(w * h for w, h in zip(cap_widths, cap_heights)) / total_caps
            if total_caps > 0 else 1
        )

        small_objects = sum(
            1 for w, h in zip(cap_widths, cap_heights)
            if (w * h) / avg_cap_area < 0.75
        )

        perspective_depth = 1 + (small_objects / columns)

        # ------------------------------------------------------------------
        # Signal 3 — overlap depth
        # ------------------------------------------------------------------
        overlap_count = sum(
            1
            for i in range(len(caps_for_sku))
            for j in range(i + 1, len(caps_for_sku))
            if _calculate_overlap(caps_for_sku[i], caps_for_sku[j]) > 0
        )

        overlap_depth = 1 + (overlap_count / columns)

        # ------------------------------------------------------------------
        # Signal 4 — cap inside SKU depth
        # Count how many caps land in each column (must be ≥ 25 % inside
        # a real SKU box).  The maximum across all columns is the depth.
        # ------------------------------------------------------------------
        column_cap_counts = [0] * columns

        for cap in caps_for_sku:

            cap_area = (
                (cap["x2"] - cap["x1"]) *
                (cap["y2"] - cap["y1"])
            )

            best_overlap_ratio = 0

            for sku_box in sku_boxes:

                overlap = _calculate_overlap(cap, sku_box)

                if cap_area > 0:
                    ratio = overlap / cap_area
                else:
                    ratio = 0

                if ratio > best_overlap_ratio:
                    best_overlap_ratio = ratio

            # at least 25 % of the cap must sit inside a SKU box
            if best_overlap_ratio >= 0.25:

                cx = (cap["x1"] + cap["x2"]) / 2

                relative_x = cx - min_x

                column_idx = int(relative_x / avg_w)

                column_idx = max(0, min(column_idx, columns - 1))

                column_cap_counts[column_idx] += 1

        # maximum depth seen in any column
        cap_inside_depth = max(column_cap_counts)

        # ------------------------------------------------------------------
        # raw_depth — highest of all 5 signals
        # ------------------------------------------------------------------
        raw_depth = max(
            depth_caps,
            perspective_depth,
            overlap_depth,
            cap_inside_depth
        )

        # custom rounding: fractional >= 0.4 → ceil, else round
        fractional = raw_depth - int(raw_depth)

        if fractional >= 0.4:
            depth = math.ceil(raw_depth)
        else:
            depth = round(raw_depth)

        # safety clamp
        depth = max(1, min(depth, 8))

        structural = columns * depth

        # ------------------------------------------------------------------
        # CASE A: sparse section (< 10 caps) — trust raw detection count
        # CASE B: dense section  (>= 10 caps) — use structural estimate
        # ------------------------------------------------------------------
        if total_caps < CAP_COUNT_THRESHOLD:
            final      = total_caps
            count_mode = "raw_caps"          # Case A
        else:
            final      = structural          # Case B — columns * depth only
            count_mode = "structural"

        # ------------------------------------------------------------------
        # Debug prints
        # ------------------------------------------------------------------
        print("\n==============================")
        print(f"SKU = {sku}")
        print(f"Detected Caps          = {total_caps}")
        print(f"Threshold              = {CAP_COUNT_THRESHOLD}  →  mode = {count_mode}")
        print(f"Columns                = {columns}")
        print(f"Column Cap Counts      = {column_cap_counts}")
        print(f"Small Perspective Objs = {small_objects}")
        print(f"Overlap Pairs          = {overlap_count}")
        print(f"Depth From Caps        = {depth_caps:.2f}")
        print(f"Perspective Depth      = {perspective_depth:.2f}")
        print(f"Overlap Depth          = {overlap_depth:.2f}")
        print(f"Cap Inside Depth       = {cap_inside_depth}")
        print(f"RAW DEPTH              = {raw_depth:.2f}")
        print(f"FINAL DEPTH            = {depth}")
        print(f"Structural (col×depth) = {structural}")
        print(f"FINAL COUNT            = {final}  [{count_mode}]")

        final_counts[sku] = {
            "count":             final,
            "prod_class_id":     matching_prod_id,
            "detected_caps":     total_caps,
            "columns":           columns,
            "depth":             depth,
            "depth_caps":        round(depth_caps, 2),
            "perspective_depth": round(perspective_depth, 2),
            "overlap_depth":     round(overlap_depth, 2),
            "cap_inside_depth":  cap_inside_depth,
            "raw_depth":         round(raw_depth, 2),
            "count_mode":        count_mode,
        }

    return final_counts


def _resolve_sku_name(record, sku_model):
    """
    Resolve the human-readable SKU name for a cap or SKU record.

    Priority:
      1. prod_class_name  written by cap_sku_mapper BEFORE _remap_to_old_class_ids,
                          so always correct regardless of ID remapping.
      2. _name            written by cap_sku_mapper on SKU-side records.
      3. sku_model.names  direct lookup — safe for SKU records (not remapped),
                          but unreliable for cap records after remapping.
    """
    name = record.get("prod_class_name")
    if name and name.lower() not in ("unknown", ""):
        return name

    name = record.get("_name")
    if name and name.lower() not in ("unknown", ""):
        return name

    prod_class_id = record.get("prod_class_id")
    if prod_class_id is not None:
        return sku_model.names.get(prod_class_id, "unknown")

    return "unknown"


def _build_mapped_caps(mapped_caps, sku_model):
    """
    Convert cap_sku_mapper's output into the flat list that compute_counts needs,
    with "sku_name" already set — NO second geometry pass.

    Root cause this fixes
    ---------------------
    The previous version re-ran map_caps_to_sku() inside the pipeline.
    That second pass used MAX_MATCH_DIST=500px and fresh x_overlap checks.
    Caps that cap_sku_mapper resolved via its P4 fallback (e.g. a glass cap
    that was 613px from its nearest SKU box — which is WHY it fell to P4) then
    FAILED the 500px distance guard again and were dumped into a synthetic bucket
    ("coca-cola glass") instead of the real "Coca-Cola 250ml Glass" bucket.

    The fix is to skip the re-mapping entirely.  cap_sku_mapper already ran a
    6-phase match (P1 → P3r → P4) with distance limits up to 700px.  Its result
    is stored in prod_class_name, so we just read that and use it as sku_name.

    Concrete impact on the test image
    ----------------------------------
    Before fix:  glass bucket = 8 caps  →  8/6 = 1.33  → depth=1  → final=8
    After fix:   glass bucket = 9 caps  →  9/6 = 1.5   → depth=2  → structural=12 → final=12
    (The 9th cap was the P4-fallback glass cap that the old re-mapping lost.)
    """
    result = []

    for cap in mapped_caps:
        sku_name = _resolve_sku_name(cap, sku_model)

        if "shelf" in sku_name.lower():
            continue

        if sku_name not in ("unknown", ""):
            result.append({
                "x1":      cap["x1"],
                "y1":      cap["y1"],
                "x2":      cap["x2"],
                "y2":      cap["y2"],
                "sku_name": sku_name,
                "brand":    extract_brand(sku_name),
                "package":  extract_package(sku_name),
            })
        else:
            # Genuinely unresolved — keep it so the count is not silently lost
            cap_raw_name = cap.get("_capname", "")
            brand   = extract_brand(cap_raw_name)
            package = extract_package(cap_raw_name)
            synthetic = f"{brand} {package}".strip() or "unrecognized"
            result.append({
                "x1":      cap["x1"],
                "y1":      cap["y1"],
                "x2":      cap["x2"],
                "y2":      cap["y2"],
                "sku_name": synthetic,
                "brand":    brand,
                "package":  package,
            })

    return result


def _build_sku_detections(sku_dicts, sku_model):
    """
    Build the sku_detections list for compute_counts.

    SKU records are NOT remapped (cap_sku_mapper only remaps cap prod_class_ids),
    so sku_model.names.get(prod_class_id) is safe here as a final fallback.
    We still prefer _name when present (set by map_caps_to_skus) for robustness.
    """
    result = []

    for sku in sku_dicts:
        sku_name = _resolve_sku_name(sku, sku_model)

        if "shelf" in sku_name.lower():
            continue

        if sku_name in ("unknown", ""):
            continue

        result.append({
            "x1":          sku["x1"],
            "y1":          sku["y1"],
            "x2":          sku["x2"],
            "y2":          sku["y2"],
            "prod_class_id": sku["prod_class_id"],
            "brand":       extract_brand(sku_name),
            "package":     extract_package(sku_name),
            "name":        sku_name,
        })

    return result


def run_row_matcher_pipeline(
    mapped_caps,
    sku_dicts,
    sku_model
):
    """
    Entry point called from visicooler.py after cap_sku_mapper has run.

    Parameters
    ----------
    mapped_caps : list[dict]
        Output of map_caps_to_skus() filtered to a single image_file_name.
        Each cap already has:
          - prod_class_name  (correct SKU label, set before ID remapping)
          - prod_class_id    (remapped to old-model ID space — DO NOT use for
                              sku_model.names lookup on caps)
          - x1, y1, x2, y2  (pixel coords)
    sku_dicts   : list[dict]
        Output of sku_tuples_to_dicts() filtered to the same image_file_name.
        prod_class_id is in new-model space.
    sku_model   : YOLO model
        Provides .names dict for new-model class ID → label lookups.

    Returns None if mapped_caps is empty (caller must handle this).
    """
    # Fix 4: guard against empty input — avoids IndexError on mapped_caps[0]
    if not mapped_caps:
        return None

    # Step 1 — SKU geometry table (label + bounding box for every detected SKU)
    sku_detections = _build_sku_detections(sku_dicts, sku_model)

    # Step 2 — Cap list with sku_name already resolved from cap_sku_mapper.
    #           No second geometry pass — avoids losing P4-fallback caps.
    pipeline_caps = _build_mapped_caps(mapped_caps, sku_model)

    # Step 3 — Structural count (columns * depth) per SKU
    results = compute_counts(pipeline_caps, sku_detections)

    return {
        "store_id":        mapped_caps[0]["store_id"],
        "image_file_name": mapped_caps[0]["image_file_name"],
        "results":         results
    }