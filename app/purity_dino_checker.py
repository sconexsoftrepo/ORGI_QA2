# """
# purity_dino_checker.py — GroundingDINO-based Visi Cooler purity check.

# Replaces the old Ollama (qwen2.5vl) impurity_checker.py entirely. Purity is
# now decided per shelf image (subcategory 605) using a 3-case decision tree
# driven by detections the caller (visicooler.py) already has from the
# Shelf-SKU YOLO model (beverage boxes) and the Caps YOLO model (known-brand
# vs "other" cap boxes) — no models are reloaded here.

# Decision logic (see visicooler.py's per-image loop for where this is
# invoked):

#     Case 1: No beverages and no caps detected at all
#             -> IMPURE, GroundingDINO is skipped.
#     Case 2: Only unrecognized ("other") caps detected, no known beverage
#             SKU or known-brand cap
#             -> IMPURE, GroundingDINO is skipped.
#     Case 3: At least one known beverage SKU or known-brand cap detected
#             -> run the full GroundingDINO pipeline (this module's
#                check_purity_case3()).

# Case 3 flow (ported from the latest sample_dino.py, tuned thresholds):
#     * GroundingDINO "specific" pass — now split into four short thematic
#       prompts (tableware, food, personal, produce) instead of one long
#       40-phrase caption, each independently thresholdable via
#       config.json. Shorter captions reduce cross-attention drift between
#       unrelated phrases, so they can run at a lower box_threshold than
#       the old mega-prompt needed.
#     * GroundingDINO "generic" pass  — catch-all "object . item . thing .
#       stuff" prompt, for anything not covered by the specific themes.
#     * Signage-strip and degenerate full-frame boxes are dropped from all
#       passes first (printed branding strips misread as "lunch box"/"box",
#       and near-whole-image boxes from a weakly-matched phrase).
#     * The passes are merged with generic boxes dropped if they overlap a
#       specific-theme box (same physical object, less useful label).
#     * Any merged box overlapping a beverage bounding box (IoU or
#       containment, tuned thresholds) is dropped — it's part of a
#       beverage, not an impurity.
#     * Any remaining box that has a KNOWN-BRAND cap sitting inside it is
#       "rescued" (treated as a beverage false-positive) and dropped.
#       "Other"/unrecognized/alcohol caps do NOT rescue.
#     * Vague labels ("container"/"box") on a low-contrast, near-uniform
#       image patch are dropped (empty shelf corner, compressor housing,
#       shadow — not a real object).
#     * HARD OVERRIDE: if the caller passes any flagged cap or SKU hits
#       (class name starting with "other"/"others", or containing
#       "alcohol" — see is_other_cap_name()/is_excluded_sku_name()), the
#       image is forced IMPURE regardless of what GroundingDINO found.
#       This covers the mixed-shelf case that Case 1/2 doesn't catch: a
#       shelf with real Coca-Cola bottles (so it's Case 3, not Case 2)
#       that ALSO has an alcohol bottle or unrecognized-brand cap sitting
#       among them. That flagged item is a real bottle/cap shape, not a
#       cup/packet/utensil, so GroundingDINO's object prompts have no
#       reason to catch it on their own — the override closes that gap.
#     * Anything left over is a genuine foreign object -> IMPURE.
#       Nothing left over -> PURE.

# The per-image verdict (Case 1/2/3, PURE or IMPURE) is written to
# temp.purity_status_temp, keyed by (iteration_id, image_file_name).
# shelf_sequence_checker.py later joins this into
# orgi.shelfsequencecompliance.purity_status once iterationtranid is known.

# Every GroundingDINO detection that did NOT overlap a beverage box (both
# rescued-by-known-cap and genuine impurities) is also logged to
# temp.dino_nonbeverage_detections_temp, one row per detected object, later
# copied into orgi.dino_nonbeverage_detections for audit purposes.

# No per-object impurity rows are written to temp.cap_prediction_temp
# anymore — purity_status on the compliance table is the single source of
# truth for the verdict now.
# """

# import logging
# import threading

# logger = logging.getLogger(__name__)

# # ── device selection (ported from sample_dino.py) ──────────────────────────
# # groundingdino's predict() defaults to device="cuda" if not told otherwise,
# # which crashes with "Torch not compiled with CUDA enabled" on CPU-only
# # boxes. Detect once and pass explicitly, same as sample_dino.py did.
# def _detect_device() -> str:
#     try:
#         import torch
#         return "cuda" if torch.cuda.is_available() else "cpu"
#     except Exception:
#         return "cpu"


# DEVICE = _detect_device()

# # ── class id for the "nothing detected at all" sentinel (unrelated to the
# #    qwen removal — this bookkeeping stays as-is) ───────────────────────────
# NO_DETECTION_CLASS_ID = 10028

# # ── GroundingDINO prompts ────────────────────────────────────────────────
# # Previously this was one 40-phrase caption. Two problems with that:
# #   1. Bare nouns like "box" / "container" have almost no visual signature
# #      -> they fire on shelf edges, compressor housing, printed signage,
# #      shadows. Most of the signage-strip/full-frame/low-contrast filters
# #      below exist purely to clean up noise from those two words.
# #   2. One long concatenated caption increases cross-attention drift
# #      between unrelated phrases, forcing box_threshold higher than it
# #      should need to be to stay usable.
# #
# # Fix: split into short, thematically-grouped captions (shorter caption ->
# # less cross-talk -> can run at a lower, saner box_threshold), and drop
# # the bare "box"/"container" terms in favor of concrete objects. Produce
# # items get their own theme+threshold below because a fruit seen through
# # a translucent plastic bag is a much weaker visual match than a bare
# # fruit — it needs a lower bar than the other themes without reopening
# # false positives on them.
# PROMPT_TABLEWARE = (
#     "drinking glass . bowl . plate . spoon . fork . knife"
# )

# # Split out from PROMPT_TABLEWARE/PROMPT_FOOD and given its own, stricter
# # threshold below. Any "cup"-shaped term — cup, mug, curd cup, yogurt
# # cup — has been observed firing on repetitive grid/circular patterns: a
# # wire fridge shelf rack seen through reflection, a patterned ceiling
# # panel. curd cup/yogurt cup were originally left in PROMPT_FOOD and
# # still hit this exact failure at FOOD's low default threshold — moving
# # every cup-shaped term into one theme means they all get the same
# # stricter bar, instead of re-discovering this one term at a time.
# PROMPT_CUP_FAMILY = "cup . mug . curd cup . yogurt cup"

# # Split out from PROMPT_TABLEWARE and given its own, stricter threshold
# # below. "steel plate"/"steel bowl"/"steel glass" are bare shape nouns
# # with a weak visual signature — the same category of problem as the
# # original "box"/"container" terms — and have been observed firing on a
# # wire fridge shelf rack, whose repeating circular gaps (viewed at an
# # angle) are close enough to a row of stacked steel dishware to match.
# # Keeping them in their own theme lets them run at a higher bar without
# # raising the threshold for the rest of tableware (cup/mug/bowl/plate
# # etc. have much stronger, less ambiguous shapes).
# PROMPT_STEEL_TABLEWARE = "steel plate . steel bowl . steel glass"

# PROMPT_FOOD = (
#     "food packet . snack packet . chips packet . bread loaf . cake . "
#     "egg . milk packet . "
#     "paneer packet . ice cream tub . plastic food container . "
#     "steel tiffin box . lunch box . biscuit packet . chocolate bar . "
#     "detergent packet . soap bar"
# )

# # Split out from PROMPT_FOOD and given a stricter threshold below. Same
# # reasoning as "medicine bottle"/"cosmetic bottle" above — a real
# # beverage bottle can look shape-identical to a shampoo bottle, and a
# # real jar isn't reliably distinguishable from a bottle's silhouette
# # either, especially at a distance or through glass reflections. Unlike
# # medicine/cosmetic bottle (fully disabled, see below), these get to
# # stay active behind a much higher threshold rather than being disabled
# # outright — worth revisiting if they keep misfiring even at this bar.
# PROMPT_BOTTLE_LIKE_CONTAINERS = "shampoo bottle . glass jar"

# PROMPT_PERSONAL = (
#     "mobile phone . wallet . keys . "
#     "cloth . tissue paper . toy . umbrella . slipper . "
#     "shoe . lighter . matchbox . battery"
# )

# # "medicine bottle"/"cosmetic bottle" are, shape-wise, nearly
# # indistinguishable from a plain glass beverage bottle — DINO has no way
# # to read the Coca-Cola label. Unlike the other false-positive terms
# # fixed above (chocolate bar, biscuit packet, cup/mug, steel plate/bowl)
# # — which were misplaced/loosely-cropped BOXES, fixable with geometry
# # and overlap checks — this is genuine shape ambiguity: a real Coke
# # bottle and a real medicine bottle can produce equally high DINO
# # confidence, so raising the threshold filters out neither. The looser
# # bottle-shaped overlap threshold (below) only helps when a beverage box
# # exists to compare against; when the SKU model misses that particular
# # bottle (odd angle, partial view, low confidence), there's nothing to
# # exclude against and the false positive gets through regardless of how
# # loose the overlap threshold is. In practice this has kept firing on
# # real Coke bottles often enough that the term isn't worth keeping
# # active by default — disabled here; set
# # groundingdino_config.enable_bottle_shaped_personal_theme = true to
# # turn it back on if you want the recall on real medicine/cosmetic
# # bottles back and can tolerate the false-positive rate.
# PROMPT_BOTTLE_SHAPED_PERSONAL = "medicine bottle . cosmetic bottle"
# DEFAULT_ENABLE_BOTTLE_SHAPED_PERSONAL_THEME = False

# PROMPT_PRODUCE = "fruit . vegetable . plastic bag . fruit bag"

# # theme_name -> prompt string. Order matters only for debug-image naming.
# # "bottle_shaped_personal" is added conditionally at call time in
# # check_purity_case3() based on enable_bottle_shaped_personal_theme.
# SPECIFIC_PROMPTS = {
#     "tableware": PROMPT_TABLEWARE,
#     "cup_family": PROMPT_CUP_FAMILY,
#     "steel_tableware": PROMPT_STEEL_TABLEWARE,
#     "food": PROMPT_FOOD,
#     "bottle_like_containers": PROMPT_BOTTLE_LIKE_CONTAINERS,
#     "personal": PROMPT_PERSONAL,
#     "produce": PROMPT_PRODUCE,
# }

# # Per-theme threshold overrides, keyed by theme name, each a
# # (box_threshold, text_threshold) tuple. A theme not listed here falls
# # back to specific_box_threshold/specific_text_threshold from dino_cfg.
# # Overridable via config.json's groundingdino_config.theme_threshold_overrides
# # (dict of theme_name -> [box_threshold, text_threshold]).
# DEFAULT_THEME_THRESHOLD_OVERRIDES = {
#     # Text threshold nudged up slightly from the specific default — a
#     # fruit through translucent plastic still needs a low box_threshold
#     # to be caught at all, but a stricter text match cuts down on the
#     # term matching ambiguous dark/shadowed edge regions with no real
#     # object present.
#     "produce": (0.28, 0.32),
#     # Raised well above the default specific threshold — "cup"/"mug" and
#     # bare "steel X" shape nouns need a much higher bar to avoid
#     # matching wire-rack grating, patterned ceiling panels, and other
#     # repetitive circular/grid patterns.
#     "cup_family": (0.55, 0.5),
#     "steel_tableware": (0.55, 0.5),
#     # Small bump above the specific default — cuts down on weak
#     # "biscuit packet"-style matches on shadow/gap regions without
#     # requiring the much higher bar the shape-ambiguous themes need.
#     "food": (0.34, 0.30),
#     # Same treatment as cup_family/steel_tableware — real bottle-vs-
#     # shampoo-bottle/glass-jar ambiguity means these need high confidence
#     # to be worth acting on at all.
#     "bottle_like_containers": (0.55, 0.5),
# }

# # Terms whose beverage-overlap exclusion check should use a looser
# # threshold (see BOTTLE_SHAPED_BEVERAGE_IOU_THRESH /
# # BOTTLE_SHAPED_BEVERAGE_CONTAINMENT_THRESH below) because they're
# # shape-identical to a plain beverage bottle. DINO's box for these often
# # crops differently from the SKU model's box for the SAME physical
# # bottle (DINO includes more of the neck/cap, SKU sometimes stops at
# # the shoulder), so the normal overlap thresholds can fail to recognize
# # them as the same object even though they demonstrably are.
# DEFAULT_BOTTLE_SHAPED_TERMS = ("medicine bottle", "cosmetic bottle", "shampoo bottle", "glass jar")

# GROUNDING_PROMPT_GENERIC = "object . item . thing . stuff"

# # Defaults, all overridable via config.json's groundingdino_config block.
# # Tuned values (per latest sample_dino.py iteration) — the signage-strip,
# # degenerate-full-frame, and low-contrast filters below now catch most of
# # the noise that used to require a higher bar, so thresholds were lowered
# # to recover marginal true positives without a false-positive regression.
# DEFAULT_SPECIFIC_BOX_THRESHOLD = 0.28
# DEFAULT_SPECIFIC_TEXT_THRESHOLD = 0.25
# DEFAULT_GENERIC_BOX_THRESHOLD = 0.38
# DEFAULT_GENERIC_TEXT_THRESHOLD = 0.28
# DEFAULT_DEDUPE_IOU_THRESH = 0.5
# DEFAULT_DEDUPE_CONTAINMENT_THRESH = 0.6

# # Cap-rescue containment threshold — lowered from 0.5 -> 0.35. DINO's
# # "glass jar"/"medicine bottle" boxes are often drawn much taller than the
# # actual bottle, so a tight cap box sitting only in the top slice of a
# # tall DINO box was never clearing 0.5 containment despite clearly being
# # the same physical bottle.
# DEFAULT_CAP_CONTAINMENT_THRESH = 0.35
# DEFAULT_OTHER_CAP_PREFIX = "other"

# # Rescue eligibility guards — containment alone isn't a safe rescue
# # criterion. An oversized/loose DINO box (a bad "vegetable" box that
# # happens to sweep across a shelf of real bottles, a "steel tiffin box"
# # box spanning half the frame) can contain a real cap's full area by
# # sheer geometric accident, without that cap having anything to do with
# # the flagged object. Two independent sanity checks close this:
# #   1. RESCUE_MAX_BOX_AREA_FRAC — a candidate box this large relative to
# #      the frame is not "one bottle with a cap on it"; reject rescue
# #      outright regardless of containment.
# #   2. RESCUE_MAX_CANDIDATE_TO_CAP_RATIO — even for a smaller box, the
# #      candidate must be reasonably cap-shaped: a real bottle+cap box is
# #      a small multiple of the cap's own area, not tens of times larger.
# DEFAULT_RESCUE_MAX_BOX_AREA_FRAC = 0.12
# DEFAULT_RESCUE_MAX_CANDIDATE_TO_CAP_RATIO = 15.0

# # Beverage-box exclusion thresholds — same reasoning as the cap-rescue
# # threshold above: DINO's box on a bottle is often looser/larger than
# # YOLO's tight beverage box, so containment needed loosening too.
# DEFAULT_BEVERAGE_IOU_THRESH = 0.5
# DEFAULT_BEVERAGE_CONTAINMENT_THRESH = 0.4

# # Looser beverage-overlap thresholds used ONLY for bottle-shaped terms
# # (see DEFAULT_BOTTLE_SHAPED_TERMS above). A "medicine bottle" box and
# # the SKU model's beverage box, even when they're the exact same
# # physical bottle, are often cropped differently enough (DINO includes
# # more neck/cap, SKU sometimes stops at the shoulder/label) that the
# # normal thresholds fail to recognize them as the same object. Since
# # these terms are shape-identical to a plain bottle to begin with, a
# # much looser bar here is safe — the risk of dropping a genuine
# # medicine/cosmetic bottle sitting directly against a real beverage is
# # far smaller than the observed risk of flagging a real Coke bottle as
# # "medicine bottle" on itself.
# DEFAULT_BOTTLE_SHAPED_BEVERAGE_IOU_THRESH = 0.2
# DEFAULT_BOTTLE_SHAPED_BEVERAGE_CONTAINMENT_THRESH = 0.2

# # Universal column-overlap fallback (see overlaps_any_by_column above),
# # applied to EVERY specific-theme detection, not just bottle-shaped
# # terms — a loosely-cropped DINO box sitting in a real bottle's column
# # gets excluded regardless of which prompt term produced it.
# DEFAULT_COLUMN_OVERLAP_MIN_VERTICAL_FRAC = 0.5

# # Signage/branding-strip filter — the printed Coca-Cola/Sprite/Fanta strip
# # at the top or bottom of a shelf gets consistently misread by the
# # specific prompt as "lunch box"/"tiffin box"/"box". It's flat printed
# # signage, not a real object; drop anything matching that shape.
# DEFAULT_SIGNAGE_MIN_WIDTH_FRAC = 0.55    # unoccluded case: spans most of the frame width
# DEFAULT_SIGNAGE_MAX_HEIGHT_FRAC = 0.18   # and is short vertically (a "strip")
# DEFAULT_SIGNAGE_MIN_ASPECT_RATIO = 4.0   # occluded case: width / height of the visible sliver
# DEFAULT_SIGNAGE_MIN_ABS_WIDTH_FRAC = 0.12  # still require some minimum real width

# # Degenerate full-frame box filter — DINO occasionally returns a box
# # covering nearly the whole image for a weakly-matched phrase; that's not
# # a real localized object.
# DEFAULT_MAX_BOX_AREA_FRAC = 0.65

# # Low-contrast/low-variance region filter — applied to EVERY impurity
# # candidate (see check_purity_case3 below), not scoped to specific
# # terms. Originally added only for "container"/"box", then "plastic
# # bag"/"fruit bag" — the same failure (a loose box landing on a dark
# # shelf shadow or blank gap with no real object) kept resurfacing on
# # other terms, so the check now runs universally instead of requiring a
# # new term to be added to a list each time it recurs.
# DEFAULT_MIN_REGION_STD = 12.0

# GENERIC_IMPURE_LABEL = "Unidentified Item"
# IMPURE_LABEL = "Impure Object"

# # ── module-level singleton so GroundingDINO is loaded once per process,
# #    not once per image (mirrors how shelf_model/sku_model are loaded once
# #    per batch in visicooler.py) ─────────────────────────────────────────
# _grounding_model = None
# _grounding_model_key = None
# _grounding_lock = threading.Lock()


# def load_grounding_model(dino_cfg: dict):
#     """Load (or return the already-loaded) GroundingDINO model. Safe to
#     call once per batch — subsequent calls with the same config/weights
#     path are no-ops and return the cached model."""
#     global _grounding_model, _grounding_model_key

#     config_path = dino_cfg.get(
#         "config_path",
#         "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
#     )
#     weights_path = dino_cfg.get(
#         "weights_path", "GroundingDINO/weights/groundingdino_swint_ogc.pth"
#     )
#     key = (config_path, weights_path)

#     with _grounding_lock:
#         if _grounding_model is not None and _grounding_model_key == key:
#             return _grounding_model

#         from groundingdino.util.inference import load_model

#         logger.info(
#             f"Loading GroundingDINO model ({config_path}, {weights_path}) "
#             f"on device={DEVICE}"
#         )
#         _grounding_model = load_model(config_path, weights_path)
#         _grounding_model_key = key
#         logger.info("GroundingDINO model loaded successfully")
#         return _grounding_model


# # ── geometry helpers (ported from sample_dino.py) ──────────────────────────
# def iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter = max(0, xB - xA) * max(0, yB - yA)
#     if inter == 0:
#         return 0

#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     return inter / (areaA + areaB - inter)


# def containment_ratio(boxA, boxB):
#     """Fraction of the SMALLER box that lies inside the other box."""
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter = max(0, xB - xA) * max(0, yB - yA)
#     if inter == 0:
#         return 0

#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     smaller_area = min(areaA, areaB)
#     if smaller_area <= 0:
#         return 0

#     return inter / smaller_area


# def overlaps_any(candidate_box, other_boxes, iou_thresh=0.5, containment_thresh=0.6):
#     for bbox in other_boxes:
#         if iou(candidate_box, bbox) > iou_thresh:
#             return True
#         if containment_ratio(candidate_box, bbox) > containment_thresh:
#             return True
#     return False


# def overlaps_any_by_column(candidate_box, other_boxes, min_vertical_overlap_frac=0.5):
#     """
#     True if candidate_box's horizontal center falls within some box in
#     other_boxes AND the two share at least min_vertical_overlap_frac of
#     candidate_box's own height. A position-based alternative to
#     area-ratio overlap (IoU/containment): a tall, loosely-cropped DINO
#     box that only partially overlaps a bottle's tight YOLO box by area
#     can still be unambiguously "on that bottle" by column position —
#     this catches it regardless of which prompt term produced the box.
#     Originally added only for bottle-shaped terms (medicine/cosmetic
#     bottle); generalized here because the same misalignment has since
#     been observed on other terms (e.g. "chocolate bar") that have no
#     reason to be shape-confused with a bottle — the real cause is DINO's
#     box being loosely cropped relative to a real, adjacent bottle, not
#     anything specific to the term.
#     """
#     ax1, ay1, ax2, ay2 = candidate_box
#     a_center_x = (ax1 + ax2) / 2
#     a_h = max(1, ay2 - ay1)

#     for bx1, by1, bx2, by2 in other_boxes:
#         if not (bx1 <= a_center_x <= bx2):
#             continue
#         inter_y = max(0, min(ay2, by2) - max(ay1, by1))
#         if (inter_y / a_h) >= min_vertical_overlap_frac:
#             return True
#     return False


# def contains_known_cap(candidate_box, known_cap_boxes, containment_thresh=0.5,
#                         frame_w=None, frame_h=None,
#                         max_box_area_frac=DEFAULT_RESCUE_MAX_BOX_AREA_FRAC,
#                         max_candidate_to_cap_ratio=DEFAULT_RESCUE_MAX_CANDIDATE_TO_CAP_RATIO):
#     """
#     True if a KNOWN-BRAND cap is mostly contained inside candidate_box
#     AND candidate_box is plausibly "one bottle with that cap on it" —
#     not a loose/oversized DINO box that happens to sweep over an
#     unrelated cap elsewhere on the shelf.

#     Two independent size guards, checked before containment is even
#     consulted for a given cap:
#       - If frame_w/frame_h are given and candidate_box covers more than
#         max_box_area_frac of the frame, it's rejected outright (a real
#         bottle+cap never spans that much of a shelf photo).
#       - candidate_box's area must not exceed max_candidate_to_cap_ratio
#         times the cap's own area — a real bottle box is a small, fairly
#         consistent multiple of its cap's size, not tens of times larger.
#     """
#     x1, y1, x2, y2 = candidate_box
#     candidate_area = max(0, x2 - x1) * max(0, y2 - y1)

#     if frame_w and frame_h:
#         frame_area = frame_w * frame_h
#         if frame_area > 0 and (candidate_area / frame_area) > max_box_area_frac:
#             return False

#     for cap_box in known_cap_boxes:
#         cx1, cy1, cx2, cy2 = cap_box
#         cap_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
#         if cap_area <= 0:
#             continue
#         if (candidate_area / cap_area) > max_candidate_to_cap_ratio:
#             continue
#         if containment_ratio(cap_box, candidate_box) > containment_thresh:
#             return True
#     return False


# def looks_like_signage_strip(box, W, H,
#                               min_width_frac=DEFAULT_SIGNAGE_MIN_WIDTH_FRAC,
#                               max_height_frac=DEFAULT_SIGNAGE_MAX_HEIGHT_FRAC,
#                               min_aspect_ratio=DEFAULT_SIGNAGE_MIN_ASPECT_RATIO,
#                               min_abs_width_frac=DEFAULT_SIGNAGE_MIN_ABS_WIDTH_FRAC):
#     """True if `box` looks like a printed branding strip (Coca-Cola/Sprite/
#     Fanta signage at the top or bottom of a shelf) rather than a real
#     object. Covers both the unoccluded case (wide, short, at the edge) and
#     the partially-occluded case (only a sliver of the strip visible, but
#     still a flat wide-relative-to-itself band flush against the edge)."""
#     x1, y1, x2, y2 = box
#     box_w = x2 - x1
#     box_h = y2 - y1
#     if box_w <= 0 or box_h <= 0:
#         return False

#     near_top = y1 <= 0.12 * H
#     near_bottom = y2 >= 0.88 * H
#     near_edge = near_top or near_bottom

#     # Case 1: full/near-full width strip (unoccluded).
#     wide_enough = box_w >= min_width_frac * W
#     short_enough = box_h <= max_height_frac * H
#     if wide_enough and short_enough and near_edge:
#         return True

#     # Case 2: partially occluded strip — only a sliver is visible, but
#     # that sliver is still a flat, wide-relative-to-itself band right at
#     # the frame edge.
#     aspect_ratio = box_w / box_h
#     has_min_width = box_w >= min_abs_width_frac * W
#     if aspect_ratio >= min_aspect_ratio and has_min_width and near_edge:
#         return True

#     return False


# def is_degenerate_fullframe_box(box, W, H, max_box_area_frac=DEFAULT_MAX_BOX_AREA_FRAC):
#     """True if `box` covers more than max_box_area_frac of the whole
#     frame — a degenerate detection, not a real localized object."""
#     x1, y1, x2, y2 = box
#     box_w = max(0, x2 - x1)
#     box_h = max(0, y2 - y1)
#     box_area = box_w * box_h
#     frame_area = W * H
#     if frame_area <= 0:
#         return False
#     return (box_area / frame_area) > max_box_area_frac


# def is_low_contrast_region(image, box, min_region_std=DEFAULT_MIN_REGION_STD):
#     """True if the image region under `box` is a near-uniform, low-
#     contrast patch (empty shelf corner, compressor housing, shadow) with
#     no visible object structure. Used only for vague labels like
#     'container'/'box' — real low-contrast objects (a dark bottle) are
#     already excluded upstream via the beverage-box check."""
#     import cv2

#     x1, y1, x2, y2 = box
#     h, w = image.shape[:2]
#     x1 = max(0, min(w - 1, x1))
#     x2 = max(0, min(w, x2))
#     y1 = max(0, min(h - 1, y1))
#     y2 = max(0, min(h, y2))
#     if x2 <= x1 or y2 <= y1:
#         return True

#     crop = image[y1:y2, x1:x2]
#     if crop.size == 0:
#         return True

#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     return float(gray.std()) < min_region_std


# def is_other_cap_name(cls_name: str, other_prefix: str = DEFAULT_OTHER_CAP_PREFIX) -> bool:
#     """True if a cap OR SKU class name is NOT a confirmed Coca-Cola brand
#     item — i.e. it's the unrecognized 'other' catch-all (e.g. 'other cap',
#     'other water bottle') or an alcohol product ('alcohol bottle',
#     'alcohol cap'). These never count as a confirmed beverage: they don't
#     trigger Case 3 on their own, and an alcohol/other-brand box must NOT
#     rescue a GroundingDINO detection from being flagged IMPURE — it's
#     treated exactly like an unrecognized cap."""
#     name = cls_name.strip().lower()
#     return name.startswith(other_prefix) or "alcohol" in name


# # Same check, different name — used on the SKU/beverage side in
# # visicooler.py so the intent reads clearly at each call site. Both names
# # delegate to the identical logic in is_other_cap_name().
# def is_excluded_sku_name(cls_name: str, other_prefix: str = DEFAULT_OTHER_CAP_PREFIX) -> bool:
#     return is_other_cap_name(cls_name, other_prefix)


# # ── drawing helpers (ported from sample_dino.py) — used to build the
# #    Case-3 annotated debug image saved to groundingdino_annotated_image ──
# def _draw_label_box(img, box, text, color, font_scale=0.7, thickness=2):
#     import cv2

#     x1, y1, x2, y2 = box
#     cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

#     (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#     text_y = max(th + 8, y1)
#     cv2.rectangle(img, (x1, text_y - th - baseline - 4), (x1 + tw + 8, text_y + baseline), color, -1)
#     cv2.putText(img, text, (x1 + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# def _draw_banner(img, impure: bool):
#     import cv2

#     if impure:
#         banner, banner_color = "RESULT : IMPURE", (0, 0, 255)
#     else:
#         banner, banner_color = "RESULT : PURE", (0, 180, 0)

#     cv2.rectangle(img, (0, 0), (420, 45), banner_color, -1)
#     cv2.putText(img, banner, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)


# # ── GroundingDINO inference helper ──────────────────────────────────────────
# def _run_grounding(model, image_tensor, prompt, box_thresh, text_thresh, W, H):
#     from groundingdino.util.inference import predict

#     boxes, logits, phrases = predict(
#         model=model,
#         image=image_tensor,
#         caption=prompt,
#         box_threshold=box_thresh,
#         text_threshold=text_thresh,
#         device=DEVICE,
#     )

#     abs_boxes = []
#     for b, phrase in zip(boxes, phrases):
#         cx, cy, w, h = b.cpu().numpy()
#         x1 = int((cx - w / 2) * W)
#         y1 = int((cy - h / 2) * H)
#         x2 = int((cx + w / 2) * W)
#         y2 = int((cy + h / 2) * H)
#         abs_boxes.append(([x1, y1, x2, y2], phrase))

#     return abs_boxes


# def check_purity_case3(local_path: str, filename: str, beverage_boxes: list,
#                         known_cap_boxes: list, grounding_model, dino_cfg: dict,
#                         output_folder_path=None, s3_handler=None, cyclecountid=None,
#                         flagged_cap_hits: list = None, flagged_sku_hits: list = None):
#     """
#     Runs the full GroundingDINO purity pipeline for a Case-3 image (at
#     least one known beverage SKU or known-brand cap already confirmed by
#     the YOLO models).

#     flagged_cap_hits / flagged_sku_hits: optional lists of
#     ([x1, y1, x2, y2], class_name) for caps/SKUs the caller already
#     identified as "other"/"others"/alcohol via is_other_cap_name() /
#     is_excluded_sku_name() — i.e. detected on-shelf, but NOT trusted
#     enough to count as a confirmed beverage/cap. If either list is
#     non-empty, the image is forced IMPURE regardless of what
#     GroundingDINO finds (see module docstring, "HARD OVERRIDE"). Pass
#     None or [] if the caller isn't tracking these yet — the check
#     degrades gracefully to the pre-override behavior.

#     Returns a (verdict, annotated_s3_path, nonbeverage_detections) tuple:
#         verdict                "PURE" / "IMPURE" / None (None = check
#                                 failed — caller must not treat None as a
#                                 verdict)
#         annotated_s3_path      S3 key of the saved GroundingDINO debug
#                                 image, or None if DINO wasn't run, the
#                                 check failed, or the image couldn't be
#                                 built/saved/uploaded. Independent of the
#                                 verdict.
#         nonbeverage_detections list of product_name strings — one per
#                                 GroundingDINO box that did NOT overlap a
#                                 beverage box (i.e. every detection in the
#                                 'unknown' set, both the ones later
#                                 rescued by a known cap and the genuine
#                                 impurities), PLUS one entry per flagged
#                                 cap/SKU hit that triggered the hard
#                                 override. Empty list if DINO wasn't run
#                                 and no flagged hits were passed. This is
#                                 what the caller logs into
#                                 temp.dino_nonbeverage_detections_temp.

#     beverage_boxes / known_cap_boxes are reused from the YOLO detections
#     visicooler.py already ran for this image — no models are reloaded.

#     On any read/inference failure, logs a warning and returns
#     (None, None, []) — the caller must NOT treat None as a verdict and
#     should skip writing a purity_status row for this image rather than
#     guessing. A failed inspection is not the same as a dirty cooler.
#     """
#     flagged_cap_hits = flagged_cap_hits or []
#     flagged_sku_hits = flagged_sku_hits or []

#     try:
#         from groundingdino.util.inference import load_image
#         import cv2

#         image = cv2.imread(local_path)
#         if image is None:
#             logger.warning(f"Purity check: could not read image {local_path}")
#             return None, None, []

#         H, W = image.shape[:2]
#         _, image_tensor = load_image(local_path)

#         specific_thresh = dino_cfg.get("specific_box_threshold", DEFAULT_SPECIFIC_BOX_THRESHOLD)
#         specific_text_thresh = dino_cfg.get("specific_text_threshold", DEFAULT_SPECIFIC_TEXT_THRESHOLD)
#         generic_thresh = dino_cfg.get("generic_box_threshold", DEFAULT_GENERIC_BOX_THRESHOLD)
#         generic_text_thresh = dino_cfg.get("generic_text_threshold", DEFAULT_GENERIC_TEXT_THRESHOLD)
#         dedupe_iou = dino_cfg.get("dedupe_iou_thresh", DEFAULT_DEDUPE_IOU_THRESH)
#         dedupe_containment = dino_cfg.get("dedupe_containment_thresh", DEFAULT_DEDUPE_CONTAINMENT_THRESH)
#         cap_containment_thresh = dino_cfg.get("cap_containment_thresh", DEFAULT_CAP_CONTAINMENT_THRESH)
#         rescue_max_box_area_frac = dino_cfg.get("rescue_max_box_area_frac", DEFAULT_RESCUE_MAX_BOX_AREA_FRAC)
#         rescue_max_candidate_to_cap_ratio = dino_cfg.get(
#             "rescue_max_candidate_to_cap_ratio", DEFAULT_RESCUE_MAX_CANDIDATE_TO_CAP_RATIO
#         )
#         beverage_iou_thresh = dino_cfg.get("beverage_iou_thresh", DEFAULT_BEVERAGE_IOU_THRESH)
#         beverage_containment_thresh = dino_cfg.get("beverage_containment_thresh", DEFAULT_BEVERAGE_CONTAINMENT_THRESH)
#         bottle_shaped_terms = tuple(dino_cfg.get("bottle_shaped_terms", DEFAULT_BOTTLE_SHAPED_TERMS))
#         bottle_shaped_beverage_iou_thresh = dino_cfg.get(
#             "bottle_shaped_beverage_iou_thresh", DEFAULT_BOTTLE_SHAPED_BEVERAGE_IOU_THRESH
#         )
#         bottle_shaped_beverage_containment_thresh = dino_cfg.get(
#             "bottle_shaped_beverage_containment_thresh", DEFAULT_BOTTLE_SHAPED_BEVERAGE_CONTAINMENT_THRESH
#         )
#         column_overlap_min_vertical_frac = dino_cfg.get(
#             "column_overlap_min_vertical_frac", DEFAULT_COLUMN_OVERLAP_MIN_VERTICAL_FRAC
#         )
#         max_box_area_frac = dino_cfg.get("max_box_area_frac", DEFAULT_MAX_BOX_AREA_FRAC)
#         min_region_std = dino_cfg.get("min_region_std", DEFAULT_MIN_REGION_STD)

#         theme_overrides_cfg = dino_cfg.get("theme_threshold_overrides", {})

#         enable_bottle_shaped_personal_theme = dino_cfg.get(
#             "enable_bottle_shaped_personal_theme", DEFAULT_ENABLE_BOTTLE_SHAPED_PERSONAL_THEME
#         )
#         active_specific_prompts = dict(SPECIFIC_PROMPTS)
#         if enable_bottle_shaped_personal_theme:
#             active_specific_prompts["bottle_shaped_personal"] = PROMPT_BOTTLE_SHAPED_PERSONAL

#         specific_boxes = []
#         for theme_name, theme_prompt in active_specific_prompts.items():
#             box_t, text_t = theme_overrides_cfg.get(
#                 theme_name,
#                 DEFAULT_THEME_THRESHOLD_OVERRIDES.get(
#                     theme_name, (specific_thresh, specific_text_thresh)
#                 ),
#             )
#             specific_boxes.extend(
#                 _run_grounding(
#                     grounding_model, image_tensor, theme_prompt,
#                     box_t, text_t, W, H,
#                 )
#             )

#         generic_boxes = _run_grounding(
#             grounding_model, image_tensor, GROUNDING_PROMPT_GENERIC,
#             generic_thresh, generic_text_thresh, W, H,
#         )

#         # Drop signage-strip and degenerate full-frame false positives
#         # from BOTH prompts before anything else touches them.
#         def _strip_degenerate(boxes_with_phrase):
#             kept = []
#             for gbox, phrase in boxes_with_phrase:
#                 if looks_like_signage_strip(gbox, W, H):
#                     continue
#                 if is_degenerate_fullframe_box(gbox, W, H, max_box_area_frac):
#                     continue
#                 kept.append((gbox, phrase))
#             return kept

#         specific_boxes = _strip_degenerate(specific_boxes)
#         generic_boxes = _strip_degenerate(generic_boxes)

#         # Merge + de-duplicate: keep every specific box; add generic boxes
#         # only if they don't already overlap a specific box.
#         specific_only_boxes = [b for b, _ in specific_boxes]
#         merged_boxes = list(specific_boxes)
#         for gbox, gphrase in generic_boxes:
#             if not overlaps_any(gbox, specific_only_boxes, dedupe_iou, dedupe_containment):
#                 merged_boxes.append((gbox, gphrase if gphrase else GENERIC_IMPURE_LABEL))

#         # Drop anything overlapping a beverage box (looser thresholds —
#         # DINO's box on a bottle is often larger/looser than YOLO's).
#         # Bottle-shaped terms (medicine/cosmetic bottle) get an even
#         # looser threshold — see DEFAULT_BOTTLE_SHAPED_TERMS above.
#         def _is_bottle_shaped(phrase):
#             p = (phrase or "").strip().lower()
#             return any(term in p for term in bottle_shaped_terms)

#         unknown = []
#         for gbox, phrase in merged_boxes:
#             if _is_bottle_shaped(phrase):
#                 iou_t, containment_t = bottle_shaped_beverage_iou_thresh, bottle_shaped_beverage_containment_thresh
#             else:
#                 iou_t, containment_t = beverage_iou_thresh, beverage_containment_thresh
#             if overlaps_any(gbox, beverage_boxes, iou_t, containment_t):
#                 continue
#             # Universal fallback: even when area-based overlap misses
#             # (a tall/loose box that only partially covers a bottle by
#             # area), a box sitting squarely in a real bottle's column is
#             # almost certainly that same bottle, not a distinct object —
#             # regardless of which term produced the box.
#             if overlaps_any_by_column(gbox, beverage_boxes, column_overlap_min_vertical_frac):
#                 continue
#             unknown.append((gbox, phrase))

#         verdict = "PURE"
#         impure_boxes = []          # genuine foreign objects, for the debug image
#         rescued_boxes = []         # dropped via known-cap rescue, for the debug image
#         nonbeverage_detections = []  # product names of every non-beverage box, for the audit table

#         for box, phrase in unknown:
#             x1, y1, x2, y2 = box
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(W, x2), min(H, y2)
#             if x2 <= x1 or y2 <= y1:
#                 continue

#             clipped_box = (x1, y1, x2, y2)
#             item_name = phrase.strip() if phrase and phrase.strip() else IMPURE_LABEL

#             # Known-brand cap inside this box -> rescue (false positive).
#             # "Other" caps never rescue.
#             if contains_known_cap(
#                 clipped_box, known_cap_boxes, cap_containment_thresh,
#                 frame_w=W, frame_h=H,
#                 max_box_area_frac=rescue_max_box_area_frac,
#                 max_candidate_to_cap_ratio=rescue_max_candidate_to_cap_ratio,
#             ):
#                 rescued_boxes.append((clipped_box, item_name))
#                 nonbeverage_detections.append(item_name)
#                 continue

#             # Low-contrast/near-uniform patch check — was previously
#             # scoped to just "container"/"box"/"plastic bag" terms, but
#             # the same failure (a loose box landing on a dark shelf
#             # shadow or blank gap with no real object) has since shown
#             # up on other terms too (e.g. "biscuit packet", "medicine
#             # bottle" boxes clipping into shadow at a shelf edge).
#             # Applying it to every candidate closes that gap instead of
#             # requiring a new term to be added to a list each time it
#             # recurs. A real object — even a dark one — still has visible
#             # edges/text/shape giving it more than min_region_std of
#             # pixel variance; only truly flat, featureless patches score
#             # below it.
#             if is_low_contrast_region(image, clipped_box, min_region_std):
#                 continue

#             impure_boxes.append((clipped_box, item_name))
#             nonbeverage_detections.append(item_name)
#             verdict = "IMPURE"

#         # ── HARD OVERRIDE: flagged cap / SKU classes ───────────────────────
#         # A flagged cap ("other"/"others"/alcohol) or flagged SKU
#         # (unrecognized brand or alcohol product) is a real bottle/cap
#         # shape sitting on the shelf, not a cup/packet/utensil — so none
#         # of GroundingDINO's object prompts have any reason to catch it
#         # on their own. If the caller passed any, force IMPURE here
#         # regardless of what the DINO pass above found, and add them to
#         # both the debug image and the audit trail so they're visible in
#         # exactly the same places a genuine DINO-caught impurity would be.
#         flagged_hits = list(flagged_cap_hits) + list(flagged_sku_hits)
#         for fbox, fname in flagged_hits:
#             fx1, fy1, fx2, fy2 = fbox
#             fx1, fy1 = max(0, fx1), max(0, fy1)
#             fx2, fy2 = min(W, fx2), min(H, fy2)
#             if fx2 <= fx1 or fy2 <= fy1:
#                 continue
#             impure_boxes.append(((fx1, fy1, fx2, fy2), f"Flagged: {fname}"))
#             nonbeverage_detections.append(fname)

#         if flagged_hits:
#             verdict = "IMPURE"
#             logger.info(
#                 f"Purity check: {filename} forced IMPURE by hard override "
#                 f"({len(flagged_cap_hits)} flagged cap(s), "
#                 f"{len(flagged_sku_hits)} flagged SKU(s))"
#             )

#         # ── Build + save + upload the annotated debug image (best-effort;
#         #    failure here does NOT change `verdict`) ─────────────────────
#         annotated_s3_path = None
#         if output_folder_path is not None and s3_handler is not None:
#             try:
#                 vis = image.copy()
#                 for bx in beverage_boxes:
#                     _draw_label_box(vis, bx, "Beverage", (0, 255, 0))
#                 for bx in known_cap_boxes:
#                     _draw_label_box(vis, bx, "Known Cap", (0, 255, 255))
#                 for bx, name in rescued_boxes:
#                     _draw_label_box(vis, bx, f"Rescued: {name}", (255, 165, 0))
#                 for bx, name in impure_boxes:
#                     _draw_label_box(vis, bx, name, (0, 0, 255))
#                 _draw_banner(vis, verdict == "IMPURE")

#                 import os
#                 os.makedirs(output_folder_path, exist_ok=True)
#                 out_filename = f"groundingdino_{filename}"
#                 out_path = os.path.join(output_folder_path, out_filename)
#                 cv2.imwrite(out_path, vis)

#                 s3_key = f"ModelResults/Visicooler_{cyclecountid}/{out_filename}"
#                 s3_handler.upload_file_to_s3(out_path, s3_key)
#                 annotated_s3_path = s3_key
#             except Exception as e:
#                 logger.warning(
#                     f"Purity check: could not build/save/upload GroundingDINO "
#                     f"annotated image for {filename}: {e}"
#                 )
#                 annotated_s3_path = None

#         return verdict, annotated_s3_path, nonbeverage_detections

#     except Exception as e:
#         logger.warning(f"GroundingDINO purity check failed for {local_path}: {e}")
#         return None, None, []


# # ── temp.purity_status_temp bookkeeping ─────────────────────────────────────
# def ensure_purity_temp_table(cur) -> None:
#     """Create the per-image purity verdict table if it doesn't exist yet.
#     Keyed by (iteration_id, image_file_name) — shelf_sequence_checker.py
#     joins on this to populate orgi.shelfsequencecompliance.purity_status
#     and .groundingdino_annotated_image once iterationtranid is resolved
#     downstream."""
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS temp.purity_status_temp (
#             iteration_id                 INTEGER NOT NULL,
#             store_id                     TEXT,
#             image_file_name              TEXT NOT NULL,
#             purity_status                VARCHAR(20) NOT NULL,
#             groundingdino_annotated_image VARCHAR(500),
#             checked_at                   TIMESTAMP DEFAULT now(),
#             PRIMARY KEY (iteration_id, image_file_name)
#         )
#     """)
#     # For installs created before this column existed.
#     cur.execute("""
#         ALTER TABLE temp.purity_status_temp
#         ADD COLUMN IF NOT EXISTS groundingdino_annotated_image VARCHAR(500)
#     """)


# def insert_purity_status(cur, records: list, conn=None, chunk_size: int = 100) -> int:
#     """
#     Upsert per-image purity verdicts.

#     records: list of (iteration_id, store_id, image_file_name,
#                        purity_status, groundingdino_annotated_image)
#     groundingdino_annotated_image is None whenever GroundingDINO was
#     skipped (Case 1/2) or the annotated image couldn't be built/saved.
#     """
#     if not records:
#         return 0

#     upsert_sql = """
#         INSERT INTO temp.purity_status_temp
#             (iteration_id, store_id, image_file_name, purity_status,
#              groundingdino_annotated_image, checked_at)
#         VALUES (%s, %s, %s, %s, %s, now())
#         ON CONFLICT (iteration_id, image_file_name) DO UPDATE SET
#             store_id                      = EXCLUDED.store_id,
#             purity_status                 = EXCLUDED.purity_status,
#             groundingdino_annotated_image = EXCLUDED.groundingdino_annotated_image,
#             checked_at                    = now()
#     """

#     inserted = 0
#     for i in range(0, len(records), chunk_size):
#         chunk = records[i:i + chunk_size]
#         try:
#             cur.executemany(upsert_sql, chunk)
#             if conn:
#                 conn.commit()
#             inserted += len(chunk)
#         except Exception as e:
#             logger.error(f"Failed upserting purity_status chunk {i}-{i + len(chunk)}: {e}")
#             raise

#     logger.info(f"Upserted {inserted} purity_status record(s) into temp.purity_status_temp")
#     return inserted


# # ── temp.dino_nonbeverage_detections_temp bookkeeping ───────────────────────
# # One row per GroundingDINO detection that did NOT overlap a beverage box
# # (both rescued-by-known-cap and genuine-impurity detections — everything
# # in check_purity_case3()'s `unknown` set). shelf_sequence_checker.py joins
# # this in by (iteration_id, image_file_name) once iterationtranid is known
# # and copies it into orgi.dino_nonbeverage_detections.
# def ensure_dino_detection_temp_table(cur) -> None:
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS temp.dino_nonbeverage_detections_temp (
#             id                SERIAL PRIMARY KEY,
#             iteration_id      INTEGER NOT NULL,
#             store_id          TEXT,
#             image_file_name   TEXT NOT NULL,
#             product_name      TEXT,
#             annotated_image   VARCHAR(500),
#             detected_at       TIMESTAMP DEFAULT now()
#         )
#     """)


# def insert_dino_detections(cur, records: list, conn=None, chunk_size: int = 100) -> int:
#     """
#     Insert non-beverage GroundingDINO detections (one row per detected
#     object, not per image — an image can produce several rows).

#     records: list of (iteration_id, store_id, image_file_name,
#                        product_name, annotated_image)
#     """
#     if not records:
#         return 0

#     insert_sql = """
#         INSERT INTO temp.dino_nonbeverage_detections_temp
#             (iteration_id, store_id, image_file_name, product_name,
#              annotated_image, detected_at)
#         VALUES (%s, %s, %s, %s, %s, now())
#     """

#     inserted = 0
#     for i in range(0, len(records), chunk_size):
#         chunk = records[i:i + chunk_size]
#         try:
#             cur.executemany(insert_sql, chunk)
#             if conn:
#                 conn.commit()
#             inserted += len(chunk)
#         except Exception as e:
#             logger.error(f"Failed inserting dino_nonbeverage_detections chunk {i}-{i + len(chunk)}: {e}")
#             raise

#     logger.info(f"Inserted {inserted} DINO non-beverage detection record(s)")
#     return inserted


# # ── "no detections at all" sentinel (10028) — unchanged from
# #    impurity_checker.py, kept here since impurity_checker.py is removed ───
# def insert_no_detection_sentinel(cur, store_id, filename: str, iteration_id: int,
#                                   shelf_index: int, s3path_annotated: str,
#                                   conn=None) -> None:
#     """Insert one 10028 'no detections at all' sentinel row for an image
#     where neither the cap model nor the sku model found anything.

#     x1/x2/y1/y2 are sent as -1 (not NULL) — temp.cap_prediction_temp has a
#     NOT NULL constraint on these bbox columns, and -1 is an unambiguous
#     "no bounding box" placeholder that satisfies it."""
#     insert_sql = """
#         INSERT INTO temp.cap_prediction_temp (
#             store_id, image_file_name, s3path_annotated_file,
#             iteration_id, cap_class_id, prod_class_id,
#             x1, x2, y1, y2, shelfnumber, brand_name
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         ON CONFLICT DO NOTHING
#     """
#     try:
#         cur.execute(insert_sql, (
#             store_id, filename, s3path_annotated, iteration_id,
#             NO_DETECTION_CLASS_ID, None, -1, -1, -1, -1,
#             shelf_index, None,
#         ))
#         if conn:
#             conn.commit()
#     except Exception as e:
#         logger.error(f"Failed inserting no-detection sentinel for {filename}: {e}")
#         raise


"""
purity_dino_checker.py — GroundingDINO-based Visi Cooler purity check.

Replaces the old Ollama (qwen2.5vl) impurity_checker.py entirely. Purity is
now decided per shelf image (subcategory 605) using a 3-case decision tree
driven by detections the caller (visicooler.py) already has from the
Shelf-SKU YOLO model (beverage boxes) and the Caps YOLO model (known-brand
vs "other" cap boxes) — no models are reloaded here.

Decision logic (see visicooler.py's per-image loop for where this is
invoked):

    Case 1: No beverages and no caps detected at all
            -> IMPURE, GroundingDINO is skipped.
    Case 2: Only unrecognized ("other") caps detected, no known beverage
            SKU or known-brand cap
            -> IMPURE, GroundingDINO is skipped.
    Case 3: At least one known beverage SKU or known-brand cap detected
            -> run the full GroundingDINO pipeline (this module's
               check_purity_case3()).

Case 3 flow (ported from the latest sample_dino.py, tuned thresholds):
    * GroundingDINO "specific" pass — now split into four short thematic
      prompts (tableware, food, personal, produce) instead of one long
      40-phrase caption, each independently thresholdable via
      config.json. Shorter captions reduce cross-attention drift between
      unrelated phrases, so they can run at a lower box_threshold than
      the old mega-prompt needed.
    * GroundingDINO "generic" pass  — catch-all "object . item . thing .
      stuff" prompt, for anything not covered by the specific themes.
    * Signage-strip and degenerate full-frame boxes are dropped from all
      passes first (printed branding strips misread as "lunch box"/"box",
      and near-whole-image boxes from a weakly-matched phrase).
    * Intra-pass de-duplication — independent themed prompts (or multiple
      phrases within one prompt's caption) can each fire on the SAME
      physical object with a DIFFERENT label (e.g. a plastic bag matching
      both "food" and "produce"). Boxes within the specific pass, and
      within the generic pass, are de-duped against each other (greedy
      NMS: higher-confidence, then larger-area box wins) before anything
      else touches them. See _dedup_boxes_by_overlap().
    * The passes are merged with generic boxes dropped if they overlap a
      specific-theme box (same physical object, less useful label).
    * Any merged box overlapping a beverage bounding box (IoU or
      containment, tuned thresholds) is dropped — it's part of a
      beverage, not an impurity.
    * Any remaining box that has a KNOWN-BRAND cap sitting inside it is
      "rescued" (treated as a beverage false-positive) and dropped.
      "Other"/unrecognized/alcohol caps do NOT rescue.
    * Vague labels ("container"/"box") on a low-contrast, near-uniform
      image patch are dropped (empty shelf corner, compressor housing,
      shadow — not a real object).
    * HARD OVERRIDE: if the caller passes any flagged cap or SKU hits
      (class name starting with "other"/"others", or containing
      "alcohol" — see is_other_cap_name()/is_excluded_sku_name()), the
      image is forced IMPURE regardless of what GroundingDINO found.
      This covers the mixed-shelf case that Case 1/2 doesn't catch: a
      shelf with real Coca-Cola bottles (so it's Case 3, not Case 2)
      that ALSO has an alcohol bottle or unrecognized-brand cap sitting
      among them. That flagged item is a real bottle/cap shape, not a
      cup/packet/utensil, so GroundingDINO's object prompts have no
      reason to catch it on their own — the override closes that gap.
    * Anything left over is a genuine foreign object -> IMPURE.
      Nothing left over -> PURE.

The per-image verdict (Case 1/2/3, PURE or IMPURE) is written to
temp.purity_status_temp, keyed by (iteration_id, image_file_name).
shelf_sequence_checker.py later joins this into
orgi.shelfsequencecompliance.purity_status once iterationtranid is known.

Every GroundingDINO detection that did NOT overlap a beverage box (both
rescued-by-known-cap and genuine impurities) is also logged to
temp.dino_nonbeverage_detections_temp, one row per detected object, later
copied into orgi.dino_nonbeverage_detections for audit purposes.

No per-object impurity rows are written to temp.cap_prediction_temp
anymore — purity_status on the compliance table is the single source of
truth for the verdict now.
"""

import logging
import threading

logger = logging.getLogger(__name__)

# ── device selection (ported from sample_dino.py) ──────────────────────────
# groundingdino's predict() defaults to device="cuda" if not told otherwise,
# which crashes with "Torch not compiled with CUDA enabled" on CPU-only
# boxes. Detect once and pass explicitly, same as sample_dino.py did.
def _detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = _detect_device()

# ── class id for the "nothing detected at all" sentinel (unrelated to the
#    qwen removal — this bookkeeping stays as-is) ───────────────────────────
NO_DETECTION_CLASS_ID = 10028

# ── GroundingDINO prompts ────────────────────────────────────────────────
# Previously this was one 40-phrase caption. Two problems with that:
#   1. Bare nouns like "box" / "container" have almost no visual signature
#      -> they fire on shelf edges, compressor housing, printed signage,
#      shadows. Most of the signage-strip/full-frame/low-contrast filters
#      below exist purely to clean up noise from those two words.
#   2. One long concatenated caption increases cross-attention drift
#      between unrelated phrases, forcing box_threshold higher than it
#      should need to be to stay usable.
#
# Fix: split into short, thematically-grouped captions (shorter caption ->
# less cross-talk -> can run at a lower, saner box_threshold), and drop
# the bare "box"/"container" terms in favor of concrete objects. Produce
# items get their own theme+threshold below because a fruit seen through
# a translucent plastic bag is a much weaker visual match than a bare
# fruit — it needs a lower bar than the other themes without reopening
# false positives on them.
PROMPT_TABLEWARE = (
    "drinking glass . bowl . plate . spoon . fork . knife"
)

# Split out from PROMPT_TABLEWARE/PROMPT_FOOD and given its own, stricter
# threshold below. Any "cup"-shaped term — cup, mug, curd cup, yogurt
# cup — has been observed firing on repetitive grid/circular patterns: a
# wire fridge shelf rack seen through reflection, a patterned ceiling
# panel. curd cup/yogurt cup were originally left in PROMPT_FOOD and
# still hit this exact failure at FOOD's low default threshold — moving
# every cup-shaped term into one theme means they all get the same
# stricter bar, instead of re-discovering this one term at a time.
PROMPT_CUP_FAMILY = "cup . mug . curd cup . yogurt cup"

# Split out from PROMPT_TABLEWARE and given its own, stricter threshold
# below. "steel plate"/"steel bowl"/"steel glass" are bare shape nouns
# with a weak visual signature — the same category of problem as the
# original "box"/"container" terms — and have been observed firing on a
# wire fridge shelf rack, whose repeating circular gaps (viewed at an
# angle) are close enough to a row of stacked steel dishware to match.
# Keeping them in their own theme lets them run at a higher bar without
# raising the threshold for the rest of tableware (cup/mug/bowl/plate
# etc. have much stronger, less ambiguous shapes).
PROMPT_STEEL_TABLEWARE = "steel plate . steel bowl . steel glass"

PROMPT_FOOD = (
    "food packet . snack packet . chips packet . bread loaf . cake . "
    "egg . milk packet . "
    "paneer packet . ice cream tub . plastic food container . "
    "steel tiffin box . lunch box . biscuit packet . chocolate bar . "
    "detergent packet . soap bar"
)

# Split out from PROMPT_FOOD and given a stricter threshold below. Same
# reasoning as "medicine bottle"/"cosmetic bottle" above — a real
# beverage bottle can look shape-identical to a shampoo bottle, and a
# real jar isn't reliably distinguishable from a bottle's silhouette
# either, especially at a distance or through glass reflections. Unlike
# medicine/cosmetic bottle (fully disabled, see below), these get to
# stay active behind a much higher threshold rather than being disabled
# outright — worth revisiting if they keep misfiring even at this bar.
PROMPT_BOTTLE_LIKE_CONTAINERS = "shampoo bottle . glass jar"

PROMPT_PERSONAL = (
    "mobile phone . wallet . keys . "
    "cloth . tissue paper . toy . umbrella . slipper . "
    "shoe . lighter . matchbox . battery"
)

# "medicine bottle"/"cosmetic bottle" are, shape-wise, nearly
# indistinguishable from a plain glass beverage bottle — DINO has no way
# to read the Coca-Cola label. Unlike the other false-positive terms
# fixed above (chocolate bar, biscuit packet, cup/mug, steel plate/bowl)
# — which were misplaced/loosely-cropped BOXES, fixable with geometry
# and overlap checks — this is genuine shape ambiguity: a real Coke
# bottle and a real medicine bottle can produce equally high DINO
# confidence, so raising the threshold filters out neither. The looser
# bottle-shaped overlap threshold (below) only helps when a beverage box
# exists to compare against; when the SKU model misses that particular
# bottle (odd angle, partial view, low confidence), there's nothing to
# exclude against and the false positive gets through regardless of how
# loose the overlap threshold is. In practice this has kept firing on
# real Coke bottles often enough that the term isn't worth keeping
# active by default — disabled here; set
# groundingdino_config.enable_bottle_shaped_personal_theme = true to
# turn it back on if you want the recall on real medicine/cosmetic
# bottles back and can tolerate the false-positive rate.
PROMPT_BOTTLE_SHAPED_PERSONAL = "medicine bottle . cosmetic bottle"
DEFAULT_ENABLE_BOTTLE_SHAPED_PERSONAL_THEME = False

PROMPT_PRODUCE = "fruit . vegetable . plastic bag . fruit bag"

# theme_name -> prompt string. Order matters only for debug-image naming.
# "bottle_shaped_personal" is added conditionally at call time in
# check_purity_case3() based on enable_bottle_shaped_personal_theme.
SPECIFIC_PROMPTS = {
    "tableware": PROMPT_TABLEWARE,
    "cup_family": PROMPT_CUP_FAMILY,
    "steel_tableware": PROMPT_STEEL_TABLEWARE,
    "food": PROMPT_FOOD,
    "bottle_like_containers": PROMPT_BOTTLE_LIKE_CONTAINERS,
    "personal": PROMPT_PERSONAL,
    "produce": PROMPT_PRODUCE,
}

# Per-theme threshold overrides, keyed by theme name, each a
# (box_threshold, text_threshold) tuple. A theme not listed here falls
# back to specific_box_threshold/specific_text_threshold from dino_cfg.
# Overridable via config.json's groundingdino_config.theme_threshold_overrides
# (dict of theme_name -> [box_threshold, text_threshold]).
DEFAULT_THEME_THRESHOLD_OVERRIDES = {
    # Text threshold nudged up slightly from the specific default — a
    # fruit through translucent plastic still needs a low box_threshold
    # to be caught at all, but a stricter text match cuts down on the
    # term matching ambiguous dark/shadowed edge regions with no real
    # object present.
    "produce": (0.28, 0.32),
    # Raised well above the default specific threshold — "cup"/"mug" and
    # bare "steel X" shape nouns need a much higher bar to avoid
    # matching wire-rack grating, patterned ceiling panels, and other
    # repetitive circular/grid patterns.
    "cup_family": (0.55, 0.5),
    "steel_tableware": (0.55, 0.5),
    # Small bump above the specific default — cuts down on weak
    # "biscuit packet"-style matches on shadow/gap regions without
    # requiring the much higher bar the shape-ambiguous themes need.
    "food": (0.34, 0.30),
    # Same treatment as cup_family/steel_tableware — real bottle-vs-
    # shampoo-bottle/glass-jar ambiguity means these need high confidence
    # to be worth acting on at all.
    "bottle_like_containers": (0.55, 0.5),
}

# Terms whose beverage-overlap exclusion check should use a looser
# threshold (see BOTTLE_SHAPED_BEVERAGE_IOU_THRESH /
# BOTTLE_SHAPED_BEVERAGE_CONTAINMENT_THRESH below) because they're
# shape-identical to a plain beverage bottle. DINO's box for these often
# crops differently from the SKU model's box for the SAME physical
# bottle (DINO includes more of the neck/cap, SKU sometimes stops at
# the shoulder), so the normal overlap thresholds can fail to recognize
# them as the same object even though they demonstrably are.
DEFAULT_BOTTLE_SHAPED_TERMS = ("medicine bottle", "cosmetic bottle", "shampoo bottle", "glass jar")

GROUNDING_PROMPT_GENERIC = "object . item . thing . stuff"

# Defaults, all overridable via config.json's groundingdino_config block.
# Tuned values (per latest sample_dino.py iteration) — the signage-strip,
# degenerate-full-frame, and low-contrast filters below now catch most of
# the noise that used to require a higher bar, so thresholds were lowered
# to recover marginal true positives without a false-positive regression.
DEFAULT_SPECIFIC_BOX_THRESHOLD = 0.28
DEFAULT_SPECIFIC_TEXT_THRESHOLD = 0.25
DEFAULT_GENERIC_BOX_THRESHOLD = 0.38
DEFAULT_GENERIC_TEXT_THRESHOLD = 0.28
DEFAULT_DEDUPE_IOU_THRESH = 0.5
DEFAULT_DEDUPE_CONTAINMENT_THRESH = 0.6

# Cap-rescue containment threshold — lowered from 0.5 -> 0.35. DINO's
# "glass jar"/"medicine bottle" boxes are often drawn much taller than the
# actual bottle, so a tight cap box sitting only in the top slice of a
# tall DINO box was never clearing 0.5 containment despite clearly being
# the same physical bottle.
DEFAULT_CAP_CONTAINMENT_THRESH = 0.35
DEFAULT_OTHER_CAP_PREFIX = "other"

# Rescue eligibility guards — containment alone isn't a safe rescue
# criterion. An oversized/loose DINO box (a bad "vegetable" box that
# happens to sweep across a shelf of real bottles, a "steel tiffin box"
# box spanning half the frame) can contain a real cap's full area by
# sheer geometric accident, without that cap having anything to do with
# the flagged object. Two independent sanity checks close this:
#   1. RESCUE_MAX_BOX_AREA_FRAC — a candidate box this large relative to
#      the frame is not "one bottle with a cap on it"; reject rescue
#      outright regardless of containment.
#   2. RESCUE_MAX_CANDIDATE_TO_CAP_RATIO — even for a smaller box, the
#      candidate must be reasonably cap-shaped: a real bottle+cap box is
#      a small multiple of the cap's own area, not tens of times larger.
DEFAULT_RESCUE_MAX_BOX_AREA_FRAC = 0.12
DEFAULT_RESCUE_MAX_CANDIDATE_TO_CAP_RATIO = 15.0

# Beverage-box exclusion thresholds — same reasoning as the cap-rescue
# threshold above: DINO's box on a bottle is often looser/larger than
# YOLO's tight beverage box, so containment needed loosening too.
DEFAULT_BEVERAGE_IOU_THRESH = 0.5
DEFAULT_BEVERAGE_CONTAINMENT_THRESH = 0.4

# Looser beverage-overlap thresholds used ONLY for bottle-shaped terms
# (see DEFAULT_BOTTLE_SHAPED_TERMS above). A "medicine bottle" box and
# the SKU model's beverage box, even when they're the exact same
# physical bottle, are often cropped differently enough (DINO includes
# more neck/cap, SKU sometimes stops at the shoulder/label) that the
# normal thresholds fail to recognize them as the same object. Since
# these terms are shape-identical to a plain bottle to begin with, a
# much looser bar here is safe — the risk of dropping a genuine
# medicine/cosmetic bottle sitting directly against a real beverage is
# far smaller than the observed risk of flagging a real Coke bottle as
# "medicine bottle" on itself.
DEFAULT_BOTTLE_SHAPED_BEVERAGE_IOU_THRESH = 0.2
DEFAULT_BOTTLE_SHAPED_BEVERAGE_CONTAINMENT_THRESH = 0.2

# Universal column-overlap fallback (see overlaps_any_by_column above),
# applied to EVERY specific-theme detection, not just bottle-shaped
# terms — a loosely-cropped DINO box sitting in a real bottle's column
# gets excluded regardless of which prompt term produced it.
DEFAULT_COLUMN_OVERLAP_MIN_VERTICAL_FRAC = 0.5

# Signage/branding-strip filter — the printed Coca-Cola/Sprite/Fanta strip
# at the top or bottom of a shelf gets consistently misread by the
# specific prompt as "lunch box"/"tiffin box"/"box". It's flat printed
# signage, not a real object; drop anything matching that shape.
DEFAULT_SIGNAGE_MIN_WIDTH_FRAC = 0.55    # unoccluded case: spans most of the frame width
DEFAULT_SIGNAGE_MAX_HEIGHT_FRAC = 0.18   # and is short vertically (a "strip")
DEFAULT_SIGNAGE_MIN_ASPECT_RATIO = 4.0   # occluded case: width / height of the visible sliver
DEFAULT_SIGNAGE_MIN_ABS_WIDTH_FRAC = 0.12  # still require some minimum real width

# Degenerate full-frame box filter — DINO occasionally returns a box
# covering nearly the whole image for a weakly-matched phrase; that's not
# a real localized object.
DEFAULT_MAX_BOX_AREA_FRAC = 0.65

# Low-contrast/low-variance region filter — applied to EVERY impurity
# candidate (see check_purity_case3 below), not scoped to specific
# terms. Originally added only for "container"/"box", then "plastic
# bag"/"fruit bag" — the same failure (a loose box landing on a dark
# shelf shadow or blank gap with no real object) kept resurfacing on
# other terms, so the check now runs universally instead of requiring a
# new term to be added to a list each time it recurs.
DEFAULT_MIN_REGION_STD = 12.0

GENERIC_IMPURE_LABEL = "Unidentified Item"
IMPURE_LABEL = "Impure Object"

# ── module-level singleton so GroundingDINO is loaded once per process,
#    not once per image (mirrors how shelf_model/sku_model are loaded once
#    per batch in visicooler.py) ─────────────────────────────────────────
_grounding_model = None
_grounding_model_key = None
_grounding_lock = threading.Lock()


def load_grounding_model(dino_cfg: dict):
    """Load (or return the already-loaded) GroundingDINO model. Safe to
    call once per batch — subsequent calls with the same config/weights
    path are no-ops and return the cached model."""
    global _grounding_model, _grounding_model_key

    config_path = dino_cfg.get(
        "config_path",
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    weights_path = dino_cfg.get(
        "weights_path", "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )
    key = (config_path, weights_path)

    with _grounding_lock:
        if _grounding_model is not None and _grounding_model_key == key:
            return _grounding_model

        from groundingdino.util.inference import load_model

        logger.info(
            f"Loading GroundingDINO model ({config_path}, {weights_path}) "
            f"on device={DEVICE}"
        )
        _grounding_model = load_model(config_path, weights_path)
        _grounding_model_key = key
        logger.info("GroundingDINO model loaded successfully")
        return _grounding_model


# ── geometry helpers (ported from sample_dino.py) ──────────────────────────
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter)


def containment_ratio(boxA, boxB):
    """Fraction of the SMALLER box that lies inside the other box."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    smaller_area = min(areaA, areaB)
    if smaller_area <= 0:
        return 0

    return inter / smaller_area


def overlaps_any(candidate_box, other_boxes, iou_thresh=0.5, containment_thresh=0.6):
    for bbox in other_boxes:
        if iou(candidate_box, bbox) > iou_thresh:
            return True
        if containment_ratio(candidate_box, bbox) > containment_thresh:
            return True
    return False


def overlaps_any_by_column(candidate_box, other_boxes, min_vertical_overlap_frac=0.5):
    """
    True if candidate_box's horizontal center falls within some box in
    other_boxes AND the two share at least min_vertical_overlap_frac of
    candidate_box's own height. A position-based alternative to
    area-ratio overlap (IoU/containment): a tall, loosely-cropped DINO
    box that only partially overlaps a bottle's tight YOLO box by area
    can still be unambiguously "on that bottle" by column position —
    this catches it regardless of which prompt term produced the box.
    Originally added only for bottle-shaped terms (medicine/cosmetic
    bottle); generalized here because the same misalignment has since
    been observed on other terms (e.g. "chocolate bar") that have no
    reason to be shape-confused with a bottle — the real cause is DINO's
    box being loosely cropped relative to a real, adjacent bottle, not
    anything specific to the term.
    """
    ax1, ay1, ax2, ay2 = candidate_box
    a_center_x = (ax1 + ax2) / 2
    a_h = max(1, ay2 - ay1)

    for bx1, by1, bx2, by2 in other_boxes:
        if not (bx1 <= a_center_x <= bx2):
            continue
        inter_y = max(0, min(ay2, by2) - max(ay1, by1))
        if (inter_y / a_h) >= min_vertical_overlap_frac:
            return True
    return False


def contains_known_cap(candidate_box, known_cap_boxes, containment_thresh=0.5,
                        frame_w=None, frame_h=None,
                        max_box_area_frac=DEFAULT_RESCUE_MAX_BOX_AREA_FRAC,
                        max_candidate_to_cap_ratio=DEFAULT_RESCUE_MAX_CANDIDATE_TO_CAP_RATIO):
    """
    True if a KNOWN-BRAND cap is mostly contained inside candidate_box
    AND candidate_box is plausibly "one bottle with that cap on it" —
    not a loose/oversized DINO box that happens to sweep over an
    unrelated cap elsewhere on the shelf.

    Two independent size guards, checked before containment is even
    consulted for a given cap:
      - If frame_w/frame_h are given and candidate_box covers more than
        max_box_area_frac of the frame, it's rejected outright (a real
        bottle+cap never spans that much of a shelf photo).
      - candidate_box's area must not exceed max_candidate_to_cap_ratio
        times the cap's own area — a real bottle box is a small, fairly
        consistent multiple of its cap's size, not tens of times larger.
    """
    x1, y1, x2, y2 = candidate_box
    candidate_area = max(0, x2 - x1) * max(0, y2 - y1)

    if frame_w and frame_h:
        frame_area = frame_w * frame_h
        if frame_area > 0 and (candidate_area / frame_area) > max_box_area_frac:
            return False

    for cap_box in known_cap_boxes:
        cx1, cy1, cx2, cy2 = cap_box
        cap_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
        if cap_area <= 0:
            continue
        if (candidate_area / cap_area) > max_candidate_to_cap_ratio:
            continue
        if containment_ratio(cap_box, candidate_box) > containment_thresh:
            return True
    return False


def looks_like_signage_strip(box, W, H,
                              min_width_frac=DEFAULT_SIGNAGE_MIN_WIDTH_FRAC,
                              max_height_frac=DEFAULT_SIGNAGE_MAX_HEIGHT_FRAC,
                              min_aspect_ratio=DEFAULT_SIGNAGE_MIN_ASPECT_RATIO,
                              min_abs_width_frac=DEFAULT_SIGNAGE_MIN_ABS_WIDTH_FRAC):
    """True if `box` looks like a printed branding strip (Coca-Cola/Sprite/
    Fanta signage at the top or bottom of a shelf) rather than a real
    object. Covers both the unoccluded case (wide, short, at the edge) and
    the partially-occluded case (only a sliver of the strip visible, but
    still a flat wide-relative-to-itself band flush against the edge)."""
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return False

    near_top = y1 <= 0.12 * H
    near_bottom = y2 >= 0.88 * H
    near_edge = near_top or near_bottom

    # Case 1: full/near-full width strip (unoccluded).
    wide_enough = box_w >= min_width_frac * W
    short_enough = box_h <= max_height_frac * H
    if wide_enough and short_enough and near_edge:
        return True

    # Case 2: partially occluded strip — only a sliver is visible, but
    # that sliver is still a flat, wide-relative-to-itself band right at
    # the frame edge.
    aspect_ratio = box_w / box_h
    has_min_width = box_w >= min_abs_width_frac * W
    if aspect_ratio >= min_aspect_ratio and has_min_width and near_edge:
        return True

    return False


def is_degenerate_fullframe_box(box, W, H, max_box_area_frac=DEFAULT_MAX_BOX_AREA_FRAC):
    """True if `box` covers more than max_box_area_frac of the whole
    frame — a degenerate detection, not a real localized object."""
    x1, y1, x2, y2 = box
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    box_area = box_w * box_h
    frame_area = W * H
    if frame_area <= 0:
        return False
    return (box_area / frame_area) > max_box_area_frac


def is_low_contrast_region(image, box, min_region_std=DEFAULT_MIN_REGION_STD):
    """True if the image region under `box` is a near-uniform, low-
    contrast patch (empty shelf corner, compressor housing, shadow) with
    no visible object structure. Used only for vague labels like
    'container'/'box' — real low-contrast objects (a dark bottle) are
    already excluded upstream via the beverage-box check."""
    import cv2

    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return True

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return True

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) < min_region_std


def is_other_cap_name(cls_name: str, other_prefix: str = DEFAULT_OTHER_CAP_PREFIX) -> bool:
    """True if a cap OR SKU class name is NOT a confirmed Coca-Cola brand
    item — i.e. it's the unrecognized 'other' catch-all (e.g. 'other cap',
    'other water bottle') or an alcohol product ('alcohol bottle',
    'alcohol cap'). These never count as a confirmed beverage: they don't
    trigger Case 3 on their own, and an alcohol/other-brand box must NOT
    rescue a GroundingDINO detection from being flagged IMPURE — it's
    treated exactly like an unrecognized cap."""
    name = cls_name.strip().lower()
    return name.startswith(other_prefix) or "alcohol" in name


# Same check, different name — used on the SKU/beverage side in
# visicooler.py so the intent reads clearly at each call site. Both names
# delegate to the identical logic in is_other_cap_name().
def is_excluded_sku_name(cls_name: str, other_prefix: str = DEFAULT_OTHER_CAP_PREFIX) -> bool:
    return is_other_cap_name(cls_name, other_prefix)


# ── drawing helpers (ported from sample_dino.py) — used to build the
#    Case-3 annotated debug image saved to groundingdino_annotated_image ──
def _draw_label_box(img, box, text, color, font_scale=0.7, thickness=2):
    import cv2

    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_y = max(th + 8, y1)
    cv2.rectangle(img, (x1, text_y - th - baseline - 4), (x1 + tw + 8, text_y + baseline), color, -1)
    cv2.putText(img, text, (x1 + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_banner(img, impure: bool):
    import cv2

    if impure:
        banner, banner_color = "RESULT : IMPURE", (0, 0, 255)
    else:
        banner, banner_color = "RESULT : PURE", (0, 180, 0)

    cv2.rectangle(img, (0, 0), (420, 45), banner_color, -1)
    cv2.putText(img, banner, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)


# ── GroundingDINO inference helper ──────────────────────────────────────────
def _run_grounding(model, image_tensor, prompt, box_thresh, text_thresh, W, H):
    from groundingdino.util.inference import predict

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        device=DEVICE,
    )

    abs_boxes = []
    for b, logit, phrase in zip(boxes, logits, phrases):
        cx, cy, w, h = b.cpu().numpy()
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        try:
            confidence = float(logit)
        except (TypeError, ValueError):
            confidence = 0.0
        abs_boxes.append(([x1, y1, x2, y2], phrase, confidence))

    return abs_boxes


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _dedup_boxes_by_overlap(boxes_with_conf, iou_thresh=DEFAULT_DEDUPE_IOU_THRESH,
                             containment_thresh=DEFAULT_DEDUPE_CONTAINMENT_THRESH):
    """
    Greedy NMS-style de-duplication WITHIN one detection pass (either all
    specific-theme boxes combined, or the generic pass alone).

    Root cause this fixes: `active_specific_prompts` runs 7-8 independent
    themed GroundingDINO calls (tableware, cup_family, food, produce, ...),
    and each prompt is itself a multi-phrase caption. Nothing previously
    checked whether two boxes FROM DIFFERENT THEMES (or two phrases within
    the same call) landed on the same physical object — only the
    generic-vs-specific merge below de-dupes, and only across passes, never
    within one. A single object (e.g. a plastic bag) scoring above
    threshold on two different themed prompts therefore produced two boxes
    at nearly identical coordinates with two different labels, and both
    survived into merged_boxes, the debug image, and the audit table.

    Tie-break mirrors cap_sku_mapper._dedup_by_iou(), which solves the
    identical duplicate-detection problem on the cap/SKU side: sort by
    (-confidence, -area) and keep a box only if it doesn't already overlap
    (by IoU or containment) a higher-priority box that was kept. The lower-
    confidence duplicate's label is dropped silently — logged at debug
    level only, since this is a model-confidence-debugging concern, not
    something the audit table's schema needs to carry.

    boxes_with_conf: list of (box, phrase, confidence).
    Returns a filtered list of the same shape.
    """
    if not boxes_with_conf:
        return boxes_with_conf

    ordered = sorted(
        boxes_with_conf,
        key=lambda item: (-(item[2] or 0.0), -_box_area(item[0])),
    )

    kept = []
    removed = 0
    for candidate in ordered:
        cbox, cphrase, _cconf = candidate
        is_dup = any(
            iou(cbox, kbox) > iou_thresh or containment_ratio(cbox, kbox) > containment_thresh
            for kbox, _kphrase, _kconf in kept
        )
        if is_dup:
            removed += 1
            logger.debug(f"Intra-pass dedup dropped duplicate box '{cphrase}' at {cbox}")
            continue
        kept.append(candidate)

    if removed:
        logger.debug(
            f"Intra-pass dedup removed {removed} duplicate box(es) out of "
            f"{len(boxes_with_conf)} (iou_thresh={iou_thresh}, "
            f"containment_thresh={containment_thresh})"
        )
    return kept


def check_purity_case3(local_path: str, filename: str, beverage_boxes: list,
                        known_cap_boxes: list, grounding_model, dino_cfg: dict,
                        output_folder_path=None, s3_handler=None, cyclecountid=None,
                        flagged_cap_hits: list = None, flagged_sku_hits: list = None):
    """
    Runs the full GroundingDINO purity pipeline for a Case-3 image (at
    least one known beverage SKU or known-brand cap already confirmed by
    the YOLO models).

    flagged_cap_hits / flagged_sku_hits: optional lists of
    ([x1, y1, x2, y2], class_name) for caps/SKUs the caller already
    identified as "other"/"others"/alcohol via is_other_cap_name() /
    is_excluded_sku_name() — i.e. detected on-shelf, but NOT trusted
    enough to count as a confirmed beverage/cap. If either list is
    non-empty, the image is forced IMPURE regardless of what
    GroundingDINO finds (see module docstring, "HARD OVERRIDE"). Pass
    None or [] if the caller isn't tracking these yet — the check
    degrades gracefully to the pre-override behavior.

    Returns a (verdict, annotated_s3_path, nonbeverage_detections) tuple:
        verdict                "PURE" / "IMPURE" / None (None = check
                                failed — caller must not treat None as a
                                verdict)
        annotated_s3_path      S3 key of the saved GroundingDINO debug
                                image, or None if DINO wasn't run, the
                                check failed, or the image couldn't be
                                built/saved/uploaded. Independent of the
                                verdict.
        nonbeverage_detections list of product_name strings — one per
                                GroundingDINO box that did NOT overlap a
                                beverage box (i.e. every detection in the
                                'unknown' set, both the ones later
                                rescued by a known cap and the genuine
                                impurities), PLUS one entry per flagged
                                cap/SKU hit that triggered the hard
                                override. Empty list if DINO wasn't run
                                and no flagged hits were passed. This is
                                what the caller logs into
                                temp.dino_nonbeverage_detections_temp.

    beverage_boxes / known_cap_boxes are reused from the YOLO detections
    visicooler.py already ran for this image — no models are reloaded.

    On any read/inference failure, logs a warning and returns
    (None, None, []) — the caller must NOT treat None as a verdict and
    should skip writing a purity_status row for this image rather than
    guessing. A failed inspection is not the same as a dirty cooler.
    """
    flagged_cap_hits = flagged_cap_hits or []
    flagged_sku_hits = flagged_sku_hits or []

    try:
        from groundingdino.util.inference import load_image
        import cv2

        image = cv2.imread(local_path)
        if image is None:
            logger.warning(f"Purity check: could not read image {local_path}")
            return None, None, []

        H, W = image.shape[:2]
        _, image_tensor = load_image(local_path)

        specific_thresh = dino_cfg.get("specific_box_threshold", DEFAULT_SPECIFIC_BOX_THRESHOLD)
        specific_text_thresh = dino_cfg.get("specific_text_threshold", DEFAULT_SPECIFIC_TEXT_THRESHOLD)
        generic_thresh = dino_cfg.get("generic_box_threshold", DEFAULT_GENERIC_BOX_THRESHOLD)
        generic_text_thresh = dino_cfg.get("generic_text_threshold", DEFAULT_GENERIC_TEXT_THRESHOLD)
        dedupe_iou = dino_cfg.get("dedupe_iou_thresh", DEFAULT_DEDUPE_IOU_THRESH)
        dedupe_containment = dino_cfg.get("dedupe_containment_thresh", DEFAULT_DEDUPE_CONTAINMENT_THRESH)
        cap_containment_thresh = dino_cfg.get("cap_containment_thresh", DEFAULT_CAP_CONTAINMENT_THRESH)
        rescue_max_box_area_frac = dino_cfg.get("rescue_max_box_area_frac", DEFAULT_RESCUE_MAX_BOX_AREA_FRAC)
        rescue_max_candidate_to_cap_ratio = dino_cfg.get(
            "rescue_max_candidate_to_cap_ratio", DEFAULT_RESCUE_MAX_CANDIDATE_TO_CAP_RATIO
        )
        beverage_iou_thresh = dino_cfg.get("beverage_iou_thresh", DEFAULT_BEVERAGE_IOU_THRESH)
        beverage_containment_thresh = dino_cfg.get("beverage_containment_thresh", DEFAULT_BEVERAGE_CONTAINMENT_THRESH)
        bottle_shaped_terms = tuple(dino_cfg.get("bottle_shaped_terms", DEFAULT_BOTTLE_SHAPED_TERMS))
        bottle_shaped_beverage_iou_thresh = dino_cfg.get(
            "bottle_shaped_beverage_iou_thresh", DEFAULT_BOTTLE_SHAPED_BEVERAGE_IOU_THRESH
        )
        bottle_shaped_beverage_containment_thresh = dino_cfg.get(
            "bottle_shaped_beverage_containment_thresh", DEFAULT_BOTTLE_SHAPED_BEVERAGE_CONTAINMENT_THRESH
        )
        column_overlap_min_vertical_frac = dino_cfg.get(
            "column_overlap_min_vertical_frac", DEFAULT_COLUMN_OVERLAP_MIN_VERTICAL_FRAC
        )
        max_box_area_frac = dino_cfg.get("max_box_area_frac", DEFAULT_MAX_BOX_AREA_FRAC)
        min_region_std = dino_cfg.get("min_region_std", DEFAULT_MIN_REGION_STD)

        theme_overrides_cfg = dino_cfg.get("theme_threshold_overrides", {})

        enable_bottle_shaped_personal_theme = dino_cfg.get(
            "enable_bottle_shaped_personal_theme", DEFAULT_ENABLE_BOTTLE_SHAPED_PERSONAL_THEME
        )
        active_specific_prompts = dict(SPECIFIC_PROMPTS)
        if enable_bottle_shaped_personal_theme:
            active_specific_prompts["bottle_shaped_personal"] = PROMPT_BOTTLE_SHAPED_PERSONAL

        specific_boxes = []
        for theme_name, theme_prompt in active_specific_prompts.items():
            box_t, text_t = theme_overrides_cfg.get(
                theme_name,
                DEFAULT_THEME_THRESHOLD_OVERRIDES.get(
                    theme_name, (specific_thresh, specific_text_thresh)
                ),
            )
            specific_boxes.extend(
                _run_grounding(
                    grounding_model, image_tensor, theme_prompt,
                    box_t, text_t, W, H,
                )
            )

        generic_boxes = _run_grounding(
            grounding_model, image_tensor, GROUNDING_PROMPT_GENERIC,
            generic_thresh, generic_text_thresh, W, H,
        )

        # Drop signage-strip and degenerate full-frame false positives
        # from BOTH prompts before anything else touches them. Items are
        # (box, phrase, confidence) 3-tuples at this point.
        def _strip_degenerate(boxes_with_conf):
            kept = []
            for item in boxes_with_conf:
                gbox = item[0]
                if looks_like_signage_strip(gbox, W, H):
                    continue
                if is_degenerate_fullframe_box(gbox, W, H, max_box_area_frac):
                    continue
                kept.append(item)
            return kept

        specific_boxes = _strip_degenerate(specific_boxes)
        generic_boxes = _strip_degenerate(generic_boxes)

        # Intra-pass de-duplication: independent themed prompts (tableware,
        # cup_family, food, produce, ...) — and even multiple phrases within
        # one prompt's caption — can each fire on the SAME physical object
        # with a DIFFERENT label, since nothing before this compared boxes
        # against each other within a pass. Collapse those same-object
        # duplicates now, keeping the higher-confidence (then larger-area)
        # box, before the specific-vs-generic merge below — which only ever
        # de-dupes ACROSS passes and would never have caught this. See
        # _dedup_boxes_by_overlap()'s docstring for the full rationale.
        specific_boxes = _dedup_boxes_by_overlap(specific_boxes, dedupe_iou, dedupe_containment)
        generic_boxes = _dedup_boxes_by_overlap(generic_boxes, dedupe_iou, dedupe_containment)

        # Merge + de-duplicate: keep every specific box; add generic boxes
        # only if they don't already overlap a specific box. Confidence is
        # no longer needed past this point, so collapse back to the
        # (box, phrase) 2-tuples every downstream consumer (beverage-overlap
        # check, cap-rescue, debug-image drawing) already expects.
        specific_only_boxes = [b for b, _p, _c in specific_boxes]
        merged_boxes = [(b, p) for b, p, _c in specific_boxes]
        for gbox, gphrase, _gconf in generic_boxes:
            if not overlaps_any(gbox, specific_only_boxes, dedupe_iou, dedupe_containment):
                merged_boxes.append((gbox, gphrase if gphrase else GENERIC_IMPURE_LABEL))

        # Drop anything overlapping a beverage box (looser thresholds —
        # DINO's box on a bottle is often larger/looser than YOLO's).
        # Bottle-shaped terms (medicine/cosmetic bottle) get an even
        # looser threshold — see DEFAULT_BOTTLE_SHAPED_TERMS above.
        def _is_bottle_shaped(phrase):
            p = (phrase or "").strip().lower()
            return any(term in p for term in bottle_shaped_terms)

        unknown = []
        for gbox, phrase in merged_boxes:
            if _is_bottle_shaped(phrase):
                iou_t, containment_t = bottle_shaped_beverage_iou_thresh, bottle_shaped_beverage_containment_thresh
            else:
                iou_t, containment_t = beverage_iou_thresh, beverage_containment_thresh
            if overlaps_any(gbox, beverage_boxes, iou_t, containment_t):
                continue
            # Universal fallback: even when area-based overlap misses
            # (a tall/loose box that only partially covers a bottle by
            # area), a box sitting squarely in a real bottle's column is
            # almost certainly that same bottle, not a distinct object —
            # regardless of which term produced the box.
            if overlaps_any_by_column(gbox, beverage_boxes, column_overlap_min_vertical_frac):
                continue
            unknown.append((gbox, phrase))

        verdict = "PURE"
        impure_boxes = []          # genuine foreign objects, for the debug image
        rescued_boxes = []         # dropped via known-cap rescue, for the debug image
        nonbeverage_detections = []  # product names of every non-beverage box, for the audit table

        for box, phrase in unknown:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            clipped_box = (x1, y1, x2, y2)
            item_name = phrase.strip() if phrase and phrase.strip() else IMPURE_LABEL

            # Known-brand cap inside this box -> rescue (false positive).
            # "Other" caps never rescue.
            if contains_known_cap(
                clipped_box, known_cap_boxes, cap_containment_thresh,
                frame_w=W, frame_h=H,
                max_box_area_frac=rescue_max_box_area_frac,
                max_candidate_to_cap_ratio=rescue_max_candidate_to_cap_ratio,
            ):
                rescued_boxes.append((clipped_box, item_name))
                nonbeverage_detections.append(item_name)
                continue

            # Low-contrast/near-uniform patch check — was previously
            # scoped to just "container"/"box"/"plastic bag" terms, but
            # the same failure (a loose box landing on a dark shelf
            # shadow or blank gap with no real object) has since shown
            # up on other terms too (e.g. "biscuit packet", "medicine
            # bottle" boxes clipping into shadow at a shelf edge).
            # Applying it to every candidate closes that gap instead of
            # requiring a new term to be added to a list each time it
            # recurs. A real object — even a dark one — still has visible
            # edges/text/shape giving it more than min_region_std of
            # pixel variance; only truly flat, featureless patches score
            # below it.
            if is_low_contrast_region(image, clipped_box, min_region_std):
                continue

            impure_boxes.append((clipped_box, item_name))
            nonbeverage_detections.append(item_name)
            verdict = "IMPURE"

        # ── HARD OVERRIDE: flagged cap / SKU classes ───────────────────────
        # A flagged cap ("other"/"others"/alcohol) or flagged SKU
        # (unrecognized brand or alcohol product) is a real bottle/cap
        # shape sitting on the shelf, not a cup/packet/utensil — so none
        # of GroundingDINO's object prompts have any reason to catch it
        # on their own. If the caller passed any, force IMPURE here
        # regardless of what the DINO pass above found, and add them to
        # both the debug image and the audit trail so they're visible in
        # exactly the same places a genuine DINO-caught impurity would be.
        flagged_hits = list(flagged_cap_hits) + list(flagged_sku_hits)
        for fbox, fname in flagged_hits:
            fx1, fy1, fx2, fy2 = fbox
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(W, fx2), min(H, fy2)
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            impure_boxes.append(((fx1, fy1, fx2, fy2), f"Flagged: {fname}"))
            nonbeverage_detections.append(fname)

        if flagged_hits:
            verdict = "IMPURE"
            logger.info(
                f"Purity check: {filename} forced IMPURE by hard override "
                f"({len(flagged_cap_hits)} flagged cap(s), "
                f"{len(flagged_sku_hits)} flagged SKU(s))"
            )

        # ── Build + save + upload the annotated debug image (best-effort;
        #    failure here does NOT change `verdict`) ─────────────────────
        annotated_s3_path = None
        if output_folder_path is not None and s3_handler is not None:
            try:
                vis = image.copy()
                for bx in beverage_boxes:
                    _draw_label_box(vis, bx, "Beverage", (0, 255, 0))
                for bx in known_cap_boxes:
                    _draw_label_box(vis, bx, "Known Cap", (0, 255, 255))
                for bx, name in rescued_boxes:
                    _draw_label_box(vis, bx, f"Rescued: {name}", (255, 165, 0))
                for bx, name in impure_boxes:
                    _draw_label_box(vis, bx, name, (0, 0, 255))
                _draw_banner(vis, verdict == "IMPURE")

                import os
                os.makedirs(output_folder_path, exist_ok=True)
                out_filename = f"groundingdino_{filename}"
                out_path = os.path.join(output_folder_path, out_filename)
                cv2.imwrite(out_path, vis)

                s3_key = f"ModelResults/Visicooler_{cyclecountid}/{out_filename}"
                s3_handler.upload_file_to_s3(out_path, s3_key)
                annotated_s3_path = s3_key
            except Exception as e:
                logger.warning(
                    f"Purity check: could not build/save/upload GroundingDINO "
                    f"annotated image for {filename}: {e}"
                )
                annotated_s3_path = None

        return verdict, annotated_s3_path, nonbeverage_detections

    except Exception as e:
        logger.warning(f"GroundingDINO purity check failed for {local_path}: {e}")
        return None, None, []


# ── temp.purity_status_temp bookkeeping ─────────────────────────────────────
def ensure_purity_temp_table(cur) -> None:
    """Create the per-image purity verdict table if it doesn't exist yet.
    Keyed by (iteration_id, image_file_name) — shelf_sequence_checker.py
    joins on this to populate orgi.shelfsequencecompliance.purity_status
    and .groundingdino_annotated_image once iterationtranid is resolved
    downstream."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS temp.purity_status_temp (
            iteration_id                 INTEGER NOT NULL,
            store_id                     TEXT,
            image_file_name              TEXT NOT NULL,
            purity_status                VARCHAR(20) NOT NULL,
            groundingdino_annotated_image VARCHAR(500),
            checked_at                   TIMESTAMP DEFAULT now(),
            PRIMARY KEY (iteration_id, image_file_name)
        )
    """)
    # For installs created before this column existed.
    cur.execute("""
        ALTER TABLE temp.purity_status_temp
        ADD COLUMN IF NOT EXISTS groundingdino_annotated_image VARCHAR(500)
    """)


def insert_purity_status(cur, records: list, conn=None, chunk_size: int = 100) -> int:
    """
    Upsert per-image purity verdicts.

    records: list of (iteration_id, store_id, image_file_name,
                       purity_status, groundingdino_annotated_image)
    groundingdino_annotated_image is None whenever GroundingDINO was
    skipped (Case 1/2) or the annotated image couldn't be built/saved.
    """
    if not records:
        return 0

    upsert_sql = """
        INSERT INTO temp.purity_status_temp
            (iteration_id, store_id, image_file_name, purity_status,
             groundingdino_annotated_image, checked_at)
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (iteration_id, image_file_name) DO UPDATE SET
            store_id                      = EXCLUDED.store_id,
            purity_status                 = EXCLUDED.purity_status,
            groundingdino_annotated_image = EXCLUDED.groundingdino_annotated_image,
            checked_at                    = now()
    """

    inserted = 0
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        try:
            cur.executemany(upsert_sql, chunk)
            if conn:
                conn.commit()
            inserted += len(chunk)
        except Exception as e:
            logger.error(f"Failed upserting purity_status chunk {i}-{i + len(chunk)}: {e}")
            raise

    logger.info(f"Upserted {inserted} purity_status record(s) into temp.purity_status_temp")
    return inserted


# ── temp.dino_nonbeverage_detections_temp bookkeeping ───────────────────────
# One row per GroundingDINO detection that did NOT overlap a beverage box
# (both rescued-by-known-cap and genuine-impurity detections — everything
# in check_purity_case3()'s `unknown` set). shelf_sequence_checker.py joins
# this in by (iteration_id, image_file_name) once iterationtranid is known
# and copies it into orgi.dino_nonbeverage_detections.
def ensure_dino_detection_temp_table(cur) -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS temp.dino_nonbeverage_detections_temp (
            id                SERIAL PRIMARY KEY,
            iteration_id      INTEGER NOT NULL,
            store_id          TEXT,
            image_file_name   TEXT NOT NULL,
            product_name      TEXT,
            annotated_image   VARCHAR(500),
            detected_at       TIMESTAMP DEFAULT now()
        )
    """)


def insert_dino_detections(cur, records: list, conn=None, chunk_size: int = 100) -> int:
    """
    Insert non-beverage GroundingDINO detections (one row per detected
    object, not per image — an image can produce several rows).

    records: list of (iteration_id, store_id, image_file_name,
                       product_name, annotated_image)
    """
    if not records:
        return 0

    insert_sql = """
        INSERT INTO temp.dino_nonbeverage_detections_temp
            (iteration_id, store_id, image_file_name, product_name,
             annotated_image, detected_at)
        VALUES (%s, %s, %s, %s, %s, now())
    """

    inserted = 0
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        try:
            cur.executemany(insert_sql, chunk)
            if conn:
                conn.commit()
            inserted += len(chunk)
        except Exception as e:
            logger.error(f"Failed inserting dino_nonbeverage_detections chunk {i}-{i + len(chunk)}: {e}")
            raise

    logger.info(f"Inserted {inserted} DINO non-beverage detection record(s)")
    return inserted


# ── "no detections at all" sentinel (10028) — unchanged from
#    impurity_checker.py, kept here since impurity_checker.py is removed ───
def insert_no_detection_sentinel(cur, store_id, filename: str, iteration_id: int,
                                  shelf_index: int, s3path_annotated: str,
                                  conn=None) -> None:
    """Insert one 10028 'no detections at all' sentinel row for an image
    where neither the cap model nor the sku model found anything.

    x1/x2/y1/y2 are sent as -1 (not NULL) — temp.cap_prediction_temp has a
    NOT NULL constraint on these bbox columns, and -1 is an unambiguous
    "no bounding box" placeholder that satisfies it."""
    insert_sql = """
        INSERT INTO temp.cap_prediction_temp (
            store_id, image_file_name, s3path_annotated_file,
            iteration_id, cap_class_id, prod_class_id,
            x1, x2, y1, y2, shelfnumber, brand_name
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """
    try:
        cur.execute(insert_sql, (
            store_id, filename, s3path_annotated, iteration_id,
            NO_DETECTION_CLASS_ID, None, -1, -1, -1, -1,
            shelf_index, None,
        ))
        if conn:
            conn.commit()
    except Exception as e:
        logger.error(f"Failed inserting no-detection sentinel for {filename}: {e}")
        raise