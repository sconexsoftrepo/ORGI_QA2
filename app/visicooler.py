import os
import cv2
from ultralytics import YOLO
import logging
from app.db_handler import close_db_connection, get_classtext
from app.config_loader import load_config
from app.s3_handler import S3Handler
from app.db_retry import retry_on_network_error, verify_connection
from app.cap_sku_mapper import (
    map_caps_to_skus,
    cap_tuples_to_dicts,
    sku_tuples_to_dicts,
    insert_mapped_cap_records,
)
from app.row_matcher_pipeline import (
    run_row_matcher_pipeline
)
from app.purity_dino_checker import (
    load_grounding_model,
    check_purity_case3,
    is_other_cap_name,
    is_excluded_sku_name,
    ensure_purity_temp_table,
    insert_purity_status,
    ensure_dino_detection_temp_table,
    insert_dino_detections,
    insert_no_detection_sentinel,
)
from datetime import datetime
import re
from collections import defaultdict
import pg8000.dbapi as pg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_visibilitydetails_schema(cur):
    # Check if visibilitydetails table exists and has correct schema (stub)
    return True


def upload_to_visibilitydetails(conn, cur, records, cyclecountid):
    # Upload visicooler records to orgi.visibilitydetails (stub - not currently used)
    pass


def should_ignore_class(cls_id: int, class_names: dict) -> bool:
    # Filter out unwanted detections
    name = class_names.get(cls_id, "").lower()
    false_negative_keywords = [" 700ml", "750ml", "visicooler", "cooler"]
    return any(keyword in name for keyword in false_negative_keywords)


def extract_brand_from_name(name: str) -> str:
    # Extract brand from product name
    name_lower = name.lower().strip()

    # Remove size and package type info
    name_lower = re.sub(r'\d+ml|\d+l|pet|glass|bottle|can|cap', '', name_lower)
    name_lower = name_lower.strip()

    # Check for specific brands
    if "coca-cola zero" in name_lower or "coca cola zero" in name_lower:
        return "coca-cola zero"
    if "coca-cola" in name_lower or "coca cola" in name_lower or "coke" in name_lower:
        return "coca-cola"
    if "mountain dew" in name_lower:
        return "mountain dew"
    if "kinley soda" in name_lower:
        return "kinley soda"
    if "kinley water" in name_lower:
        return "kinley water"
    if "sprite" in name_lower:
        return "sprite"
    if "fanta" in name_lower:
        return "fanta"
    if "thums up" in name_lower or "thumbs up" in name_lower:
        return "thums up"
    if "limca" in name_lower:
        return "limca"
    if "maaza" in name_lower:
        return "maaza"
    if "pepsi" in name_lower:
        return "pepsi"
    if "mirinda" in name_lower:
        return "mirinda"
    if "7up" in name_lower or "7 up" in name_lower:
        return "7up"
    if "slice" in name_lower:
        return "slice"
    if "sting" in name_lower:
        return "sting"
    if "aquafina" in name_lower:
        return "aquafina"
    if "other" in name_lower:
        return "other"

    # Default to first word if no match
    words = name_lower.split()
    return words[0] if words else ""


@retry_on_network_error(max_retries=3, delay=5)
def insert_cap_predictions(cur, cap_records):
    # Insert cap predictions with retry on network errors
    if not cap_records:
        return

    # Check if connection is still alive
    if not verify_connection(cur):
        raise Exception("Database connection lost before cap insert")

    insert_query = """
    INSERT INTO temp.cap_prediction_temp
    (store_id, image_file_name, iteration_id, cap_class_id, x1, x2, y1, y2, prod_class_id, shelfnumber, s3path_annotated_file)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        cur.executemany(insert_query, cap_records)
        logger.info(f"Inserted {len(cap_records)} cap predictions")
    except Exception as e:
        logger.error(f"Failed to insert cap predictions: {type(e).__name__}: {e}")
        raise


@retry_on_network_error(max_retries=3, delay=5)
def insert_sku_predictions(cur, sku_records):
    # Insert SKU predictions with retry on network errors
    if not sku_records:
        return

    # Check if connection is still alive
    if not verify_connection(cur):
        raise Exception("Database connection lost before SKU insert")

    insert_query = """
    INSERT INTO temp.sku_prediction_temp
    (store_id, image_file_name, iteration_id, prod_class_id, x1, x2, y1, y2, shelfnumber, brand_name, s3path_annotated_file)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        cur.executemany(insert_query, sku_records)
        logger.info(f"Inserted {len(sku_records)} SKU predictions")
    except Exception as e:
        logger.error(f"Failed to insert SKU predictions: {type(e).__name__}: {e}")
        raise


def reconnect_database(config):
    # Reconnect to database after connection loss
    try:
        db_config = config['db_config']
        conn = pg.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password'],
            timeout=30
        )
        cur = conn.cursor()
        logger.info("Database reconnected successfully")
        return conn, cur
    except Exception as e:
        logger.error(f"Failed to reconnect to database: {e}")
        raise


def run_visicooler_analysis(image_paths, config, s3_handler, conn, cur,
                            output_folder_path, cyclecountid, iterationid=None):

    try:
        # Load model paths from config
        shelf_model_path = config['visicooler_config']['caps_model_path']
        sku_model_path = config['visicooler_config']['sku_model_path']
        # annotation_model_path (model_path / capmodelnew.pt) no longer used —
        # sku_model (weights.pt) now handles both detection and S3 annotation.

        # Get confidence thresholds
        cap_conf_threshold = config['visicooler_config'].get('cap_conf_threshold', 0.1)
        sku_conf_threshold = config['visicooler_config'].get('sku_conf_threshold', 0.35)
        # annotation_conf_threshold removed — sku_conf_threshold is used for annotation too.

        # GroundingDINO purity-check settings, optional in config.json
        dino_cfg = config.get('groundingdino_config', {})
        other_cap_prefix = dino_cfg.get('other_cap_prefix', 'other')

        # Load YOLO models
        # Note: annotation_model (capmodelnew.pt) removed — sku_model (weights.pt)
        # is used for both detection AND annotated image generation, as it is the
        # upgraded model with more accurate results.
        logger.info("Loading YOLO models for visicooler analysis")
        shelf_model = YOLO(shelf_model_path)
        sku_model = YOLO(sku_model_path)
        logger.info("YOLO models loaded successfully")

        # Load GroundingDINO once per batch (Case 3 purity check) — not
        # reloaded per image.
        grounding_model = load_grounding_model(dino_cfg)

        # Get class names from models
        sku_class_names = sku_model.names
        shelf_class_names = shelf_model.names

        def _norm_storeid(sid):
            # Normalize store ID format
            if sid is None:
                return None
            if isinstance(sid, str):
                s = sid.strip()
                if s.isdigit():
                    return int(s)
                return s
            return sid

        def _get_canonical_storeid(filename, orig_storeid):
            # Get canonical store ID from database
            canonical = orig_storeid
            try:
                cur.execute("""
                    SELECT storeid
                    FROM orgi.batchtransactionvisibilityitems
                    WHERE imagefilename = %s
                    LIMIT 1
                """, (filename,))
                row = cur.fetchone()
                if row and row[0] is not None:
                    canonical = row[0]
            except Exception:
                pass
            return canonical

        def normalize_subcat(val):
            # Normalize subcategory ID
            if val is None:
                return None
            s = str(val).strip().replace(',', '')
            if s.endswith('.0'):
                s = s[:-2]
            m = re.search(r'(\d+)', s)
            return int(m.group(1)) if m else None

        # Group images by store
        store_images = {}
        for row in image_paths:
            fileseqid, storename, filename, local_path, s3_key, orig_storeid, subcategory_id = row
            canonical_storeid = _get_canonical_storeid(filename, orig_storeid)
            sid = _norm_storeid(canonical_storeid)
            subcat_norm = normalize_subcat(subcategory_id)
            store_images.setdefault(sid, []).append(
                (fileseqid, storename, filename, local_path, s3_key, canonical_storeid, subcat_norm)
            )

        logger.info("Processing subcategory 605 only")

        # Get or generate iteration ID
        if iterationid is None:
            try:
                cur.execute("SELECT COALESCE(MAX(iterationid), 0) FROM orgi.coolermetricsmaster")
                iterationid = cur.fetchone()[0] + 1
                logger.info(f"Generated new iterationid: {iterationid}")
            except Exception as e:
                logger.warning(f"Failed to get iterationid from database: {e}")
                iterationid = 1
                logger.info(f"Using default iterationid: {iterationid}")
        else:
            logger.info(f"Using provided iterationid: {iterationid}")

        # Close database before long processing
        try:
            if conn is not None and cur is not None:
                close_db_connection(conn, cur)
                logger.info("Database closed before inference (will reopen for insert)")
        except Exception as e:
            logger.warning(f"Connection already closed or error closing: {e}")

        conn = None
        cur = None

        # Initialize counters
        total_processed = 0
        total_cap_detections = 0
        total_sku_detections = 0

        # Storage for all detections
        all_cap_records = []
        all_sku_records = []

        # Purity-check bookkeeping (subcategory 605 only):
        #   all_purity_records   -> one (iterationid, storeid, filename,
        #                            verdict) tuple per image, verdict is
        #                            "PURE"/"IMPURE" from the Case 1/2/3
        #                            decision (see per-image loop below)
        #   no_detection_images  -> images where NEITHER model detected
        #                            anything at all (10028 sentinel,
        #                            independent bookkeeping — always a
        #                            Case 1 IMPURE too, but tracked
        #                            separately for cap_prediction_temp)
        all_purity_records = []
        all_dino_detection_records = []
        no_detection_images = []
        dino_failed_count = 0

        # Track every 605 image that was successfully processed (for no-detection sentinel)
        # Key: filename → (final_storeid, iterationid, shelf_index, s3path_annotated)
        processed_605_images = {}

        # Count total subcategory 605 images
        total_605_images = sum(1 for rows in store_images.values() for r in rows if r[6] == 605)
        logger.info(f"Starting visicooler analysis on {total_605_images} images (subcategory 605 only)")

        # Process each store
        for sid, rows in store_images.items():
            # Get only subcategory 605 images for shelf numbering
            shelf_605_images = [r for r in rows if r[6] == 605]

            # Process each image in the store
            for stored_row in rows:
                fileseqid, storename, filename, local_path, s3_key, final_storeid, subcat_norm = stored_row

                # Skip if not subcategory 605
                if subcat_norm != 605:
                    continue

                try:
                    # Calculate shelf number for this image
                    shelf_index = shelf_605_images.index(stored_row) + 1

                    # Per-image detection counters/boxes (used to decide the
                    # purity Case 1/2/3 branch and, for Case 3, reused as
                    # GroundingDINO's beverage/known-cap boxes)
                    image_cap_raw_count = 0   # every shelf_model box, any class
                    image_cap_count = 0       # shelf_model boxes classed as "cap"
                    image_sku_count = 0       # sku_model boxes
                    image_beverage_boxes = []   # sku_model boxes, reused by DINO
                    image_known_cap_boxes = []  # known-brand cap boxes, reused by DINO
                    image_other_cap_boxes = []  # unrecognized ("other") cap boxes
                    # Flagged (other/alcohol) cap and SKU hits, WITH class
                    # names — reused by DINO's hard override so a mixed
                    # shelf (real Coca-Cola bottles + one alcohol bottle
                    # or unrecognized cap) still gets flagged IMPURE even
                    # though it qualifies as Case 3.
                    image_flagged_cap_hits = []  # [(box, class_name), ...]
                    image_flagged_sku_hits = []  # [(box, class_name), ...]

                    # Read image
                    image = cv2.imread(local_path)
                    if image is None:
                        logger.warning(f"Failed to read image: {filename}")
                        continue

                    image_height, image_width = image.shape[:2]
                    os.makedirs(output_folder_path, exist_ok=True)

                    # Generate S3 path for annotated image
                    s3path_annotated = f"ModelResults/Visicooler_{cyclecountid}/segmented_{filename}"

                    # -------------------------------------------------------
                    # Run SKU detection model
                    # -------------------------------------------------------
                    # FIX: YOLO's .xyxy already returns absolute pixel coords
                    # when the input is a file path — no sw/sh scaling needed.
                    # The old code did `int(x1 * sw)` which double-scaled the
                    # coordinates (e.g. x1=800 * sw≈1 → still ~800, but when
                    # the model internally resizes to 640 px, sw = 1920/640 = 3,
                    # so x1=200 * 3 = 600 instead of the correct 600 raw px).
                    # Using box.xyxy directly gives the correct pixel values.
                    # -------------------------------------------------------
                    sku_results = sku_model(local_path, conf=sku_conf_threshold)

                    for result in sku_results:
                        if not result.orig_shape:
                            continue

                        for box in result.boxes:
                            cls_id = int(box.cls[0])

                            # Skip unwanted classes
                            if should_ignore_class(cls_id, sku_class_names):
                                continue

                            # FIX: Use .xyxy[0] directly — these are already in
                            # the original image's pixel coordinate space.
                            # Do NOT multiply by sw/sh; that scaling was only
                            # needed if you were working with normalised [0,1]
                            # coords from .xywhn or .xyxyn.
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1_px = int(x1)
                            y1_px = int(y1)
                            x2_px = int(x2)
                            y2_px = int(y2)

                            # Extract brand name from product name
                            product_name = sku_class_names[cls_id]
                            brand_name = extract_brand_from_name(product_name)

                            # Store detection — tuple field order matches SKU_TUPLE_FIELDS:
                            # store_id, image_file_name, iteration_id, prod_class_id,
                            # x1, x2, y1, y2, shelfnumber, brand_name, s3path_annotated_file
                            # NOTE: x2 comes BEFORE y1 in the tuple to match
                            # SKU_TUPLE_FIELDS in cap_sku_mapper.py.
                            all_sku_records.append((
                                final_storeid,
                                filename,
                                iterationid,
                                cls_id,
                                x1_px,
                                x2_px,
                                y1_px,
                                y2_px,
                                shelf_index,
                                brand_name,
                                s3path_annotated
                            ))

                            total_sku_detections += 1
                            image_sku_count += 1
                            # Only a genuine known-brand SKU counts as a
                            # "confirmed beverage" for the purity Case 1/2/3
                            # decision and DINO's rescue logic. Alcohol and
                            # unrecognized ("other") SKUs are excluded here —
                            # they still get stored in all_sku_records as
                            # normal, this only affects the purity check.
                            # They're also tracked separately (with class
                            # name) so DINO's hard override can force IMPURE
                            # even on a Case-3 shelf that otherwise has real
                            # Coca-Cola bottles alongside this flagged one.
                            if not is_excluded_sku_name(product_name, other_cap_prefix):
                                image_beverage_boxes.append([x1_px, y1_px, x2_px, y2_px])
                            else:
                                image_flagged_sku_hits.append(
                                    ([x1_px, y1_px, x2_px, y2_px], product_name)
                                )

                    # -------------------------------------------------------
                    # Run cap detection model
                    # -------------------------------------------------------
                    cap_results = shelf_model(local_path, conf=cap_conf_threshold)

                    for result in cap_results:
                        if not result.orig_shape:
                            continue

                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            name = shelf_class_names.get(cls_id, "").lower()

                            # Count every raw shelf_model box regardless of
                            # class, so we can tell "nothing on the shelf"
                            # apart from "something on the shelf, just not
                            # a recognized cap" further down.
                            image_cap_raw_count += 1

                            # Only process cap detections
                            if "cap" not in name:
                                continue

                            # FIX: Same as above — use .xyxy[0] directly.
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1_px = int(x1)
                            y1_px = int(y1)
                            x2_px = int(x2)
                            y2_px = int(y2)

                            # Store detection — tuple field order matches CAP_TUPLE_FIELDS:
                            # store_id, image_file_name, iteration_id, cap_class_id,
                            # x1, x2, y1, y2, prod_class_id, shelfnumber, s3path_annotated_file
                            # NOTE: x2 comes BEFORE y1, and prod_class_id (None at this
                            # stage) comes AFTER y2 — this matches cap_sku_mapper's
                            # CAP_TUPLE_FIELDS definition exactly.
                            all_cap_records.append((
                                final_storeid,
                                filename,
                                iterationid,
                                cls_id,
                                x1_px,
                                x2_px,
                                y1_px,
                                y2_px,
                                None,        # prod_class_id — filled by cap_sku_mapper
                                shelf_index,
                                s3path_annotated
                            ))

                            total_cap_detections += 1
                            image_cap_count += 1

                            if is_other_cap_name(name, other_cap_prefix):
                                image_other_cap_boxes.append([x1_px, y1_px, x2_px, y2_px])
                                image_flagged_cap_hits.append(
                                    ([x1_px, y1_px, x2_px, y2_px], name)
                                )
                            else:
                                image_known_cap_boxes.append([x1_px, y1_px, x2_px, y2_px])

                    # Generate annotated image using the upgraded SKU model (weights.pt).
                    # Previously used annotation_model (capmodelnew.pt), but sku_model
                    # produces more accurate detections and richer visualizations.
                    try:
                        sku_annotation_results = sku_model(local_path, conf=sku_conf_threshold)
                        rendered = sku_annotation_results[0].plot()
                        out = os.path.join(output_folder_path, f"segmented_{filename}")
                        cv2.imwrite(out, rendered)
                        s3_handler.upload_file_to_s3(out, s3path_annotated)
                        logger.info(f"Uploaded SKU-annotated image: segmented_{filename}")
                    except Exception as e:
                        logger.warning(f"Annotation failed: {e}")

                    # ── Purity check (Case 1/2/3, GroundingDINO for Case 3) ──
                    #   Case 1: no beverages and no caps at all detected
                    #           -> IMPURE, GroundingDINO skipped.
                    #   Case 2: only unrecognized ("other") caps and/or
                    #           alcohol/other-brand SKUs detected — no known
                    #           Coca-Cola SKU or known-brand cap
                    #           -> IMPURE, GroundingDINO skipped.
                    #   Case 3: at least one known beverage SKU or
                    #           known-brand cap detected
                    #           -> run the full GroundingDINO pipeline. If
                    #              this same shelf ALSO has a flagged
                    #              (other/alcohol) cap or SKU sitting among
                    #              the real bottles, check_purity_case3's
                    #              hard override forces IMPURE regardless
                    #              of what GroundingDINO itself finds.
                    is_case3 = len(image_beverage_boxes) > 0 or len(image_known_cap_boxes) > 0

                    if is_case3:
                        purity_verdict, dino_annotated_path, nonbeverage_detections = check_purity_case3(
                            local_path,
                            filename,
                            image_beverage_boxes,
                            image_known_cap_boxes,
                            grounding_model,
                            dino_cfg,
                            output_folder_path=output_folder_path,
                            s3_handler=s3_handler,
                            cyclecountid=cyclecountid,
                            flagged_cap_hits=image_flagged_cap_hits,
                            flagged_sku_hits=image_flagged_sku_hits,
                        )
                        if purity_verdict is None:
                            dino_failed_count += 1
                            logger.warning(
                                f"[PURITY] {filename}: Case 3 GroundingDINO check "
                                f"failed -> no purity_status written (left NULL)"
                            )
                        else:
                            logger.info(
                                f"[PURITY] {filename}: Case 3 (confirmed beverage) "
                                f"-> GroundingDINO verdict = {purity_verdict}, "
                                f"annotated_image = {dino_annotated_path}"
                            )
                            # Log every non-beverage detection (rescued or
                            # genuinely impure) for the audit table, one
                            # row per detected object.
                            for product_name in nonbeverage_detections:
                                all_dino_detection_records.append((
                                    iterationid, final_storeid, filename,
                                    product_name, dino_annotated_path,
                                ))
                    else:
                        purity_verdict = "IMPURE"
                        dino_annotated_path = None  # GroundingDINO skipped -> no image
                        if image_cap_raw_count == 0 and image_sku_count == 0:
                            logger.info(
                                f"[PURITY] {filename}: Case 1 (nothing detected) "
                                f"-> IMPURE (GroundingDINO skipped)"
                            )
                        else:
                            logger.info(
                                f"[PURITY] {filename}: Case 2 (only unrecognized/"
                                f"alcohol caps or SKUs, no known Coca-Cola beverage) "
                                f"-> IMPURE (GroundingDINO skipped)"
                            )

                    # Only record a purity_status row when we actually have a
                    # verdict — a GroundingDINO failure (purity_verdict is
                    # None) means "couldn't tell", not "IMPURE", so no row
                    # is written and the compliance table's purity_status
                    # stays NULL for this image rather than guessing.
                    # dino_annotated_path is independently None whenever
                    # GroundingDINO was skipped (Case 1/2) or the annotated
                    # image couldn't be built/saved/uploaded.
                    if purity_verdict is not None:
                        all_purity_records.append((
                            iterationid, final_storeid, filename,
                            purity_verdict, dino_annotated_path,
                        ))

                    # 10028 "no detections at all" sentinel bookkeeping —
                    # independent of the purity verdict above, tracks images
                    # where neither model found anything at all (still a
                    # Case 1 IMPURE, but kept as its own list because it
                    # also drives cap_prediction_temp sentinel rows).
                    if image_cap_count == 0 and image_sku_count == 0 and image_cap_raw_count == 0:
                        no_detection_images.append((
                            final_storeid,
                            filename,
                            iterationid,
                            shelf_index,
                            s3path_annotated,
                        ))

                    # Track this image so we can detect zero-cap images later
                    processed_605_images[filename] = (
                        final_storeid,
                        iterationid,
                        shelf_index,
                        s3path_annotated
                    )

                    total_processed += 1

                    # Show progress every 5 images or for the last image
                    if total_processed % 5 == 0 or total_processed == total_605_images:
                        remaining = total_605_images - total_processed
                        logger.info(f"Visicooler progress: {total_processed}/{total_605_images} images processed, {remaining} remaining")

                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")

        # ── Sentinel rows for images with ZERO cap detections ────────────────
        # Any image that was processed but produced no cap records would be
        # completely absent from the cooler metrics tables.  Insert one row
        # per such image using cap_class_id = 10010 so the store/image is
        # always trackable even when the cooler is empty or no caps are visible.
        images_with_caps = {rec[1] for rec in all_cap_records}   # field[1] = image_file_name
        # Images already routed to the 10028 "no detections at all" sentinel
        # (see impurity-check block above) are excluded here so they don't
        # also pick up a 10010 sentinel — 10028 is the more specific signal.
        no_detection_filenames = {rec[1] for rec in no_detection_images}
        no_cap_images = [
            fname for fname in processed_605_images
            if fname not in images_with_caps and fname not in no_detection_filenames
        ]

        if no_cap_images:
            logger.info(
                f"Found {len(no_cap_images)} image(s) with zero cap detections — "
                f"inserting sentinel rows with cap_class_id=10010"
            )
            for fname in no_cap_images:
                f_storeid, f_iterid, f_shelf, f_s3path = processed_605_images[fname]
                all_cap_records.append((
                    f_storeid,   # store_id
                    fname,       # image_file_name
                    f_iterid,    # iteration_id
                    10010,       # cap_class_id  — "no detection" sentinel
                    None,        # x1
                    None,        # x2  (NOTE: x2 before y1, matches CAP_TUPLE_FIELDS)
                    None,        # y1
                    None,        # y2
                    None,        # prod_class_id — no SKU mapped
                    f_shelf,     # shelfnumber
                    f_s3path,    # s3path_annotated_file
                ))
            logger.info(f"Sentinel rows added. Total cap records: {len(all_cap_records)}")
        else:
            logger.info("All processed images have at least one cap detection — no sentinel rows needed")

        # Inference complete, now reopen database for insertion
        logger.info("Visicooler analysis complete, reopening database for insertion")

        db_config = config['db_config']

        # Retry connection up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = pg.connect(
                    host=db_config['host'],
                    port=db_config['port'],
                    database=db_config['database'],
                    user=db_config['user'],
                    password=db_config['password'],
                    timeout=30
                )
                cur = conn.cursor()
                logger.info("Database reconnected successfully")
                break
            except Exception as e:
                logger.warning(f"Failed to reconnect (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5)
                else:
                    logger.error("Failed to reconnect to database after all retries")
                    raise

        # Upsert per-image purity verdicts (Case 1/2 = IMPURE without DINO,
        # Case 3 = GroundingDINO verdict) into temp.purity_status_temp.
        # shelf_sequence_checker.py joins this in by (iterationid,
        # imagefilename) once iterationtranid is resolved, and writes it to
        # orgi.shelfsequencecompliance.purity_status.
        ensure_purity_temp_table(cur)
        if all_purity_records:
            insert_purity_status(cur, all_purity_records, conn=conn)

        # Log every non-beverage GroundingDINO detection (rescued or
        # genuinely impure) into temp.dino_nonbeverage_detections_temp for
        # audit purposes. shelf_sequence_checker.py joins this in by
        # (iterationid, imagefilename) once iterationtranid is resolved
        # and copies it into orgi.dino_nonbeverage_detections.
        ensure_dino_detection_temp_table(cur)
        if all_dino_detection_records:
            insert_dino_detections(cur, all_dino_detection_records, conn=conn)

        # Insert 10028 "no detections at all" sentinels — images where
        # neither the cap model nor the sku model found anything, so the
        # LLM impurity check was skipped entirely.
        if no_detection_images:
            for f_storeid, fname, f_iterid, f_shelf, f_s3path in no_detection_images:
                insert_no_detection_sentinel(
                    cur, f_storeid, fname, f_iterid, f_shelf, f_s3path, conn=conn
                )
            logger.info(
                f"Inserted {len(no_detection_images)} no-detection (10028) "
                f"sentinel row(s)"
            )

        # Insert SKU detections into temp table
        if all_sku_records:
            insert_sku_predictions(cur, all_sku_records)

        # ── Python-side cap→SKU mapping (replaces SQL steps 3 & 4) ──────────
        # Convert tuples → dicts, run brand-aware mapper (IoU dedup + 4-phase
        # matching), then insert with prod_class_id already filled.
        # NOTE: Sentinel rows (cap_class_id=10010, no bbox) are excluded from
        # mapping — they exist only for tracking and are already inserted into
        # temp.cap_prediction_temp via insert_cap_predictions above.
        real_cap_records = [r for r in all_cap_records if r[3] != 10010]
        if real_cap_records:
            cap_dicts = cap_tuples_to_dicts(real_cap_records)
            sku_dicts = sku_tuples_to_dicts(all_sku_records) if all_sku_records else []
            mapped_caps = map_caps_to_skus(
                cap_dicts, sku_dicts,
                shelf_model.names,   # cap class names from YOLO model
                sku_model.names,     # SKU class names from YOLO model
            )
            if mapped_caps:
                insert_mapped_cap_records(
                    cur,
                    mapped_caps,
                    conn=conn
                )
                logger.info(
                    f"Inserted {len(mapped_caps)} mapped cap record(s) via "
                    "cap_sku_mapper (IoU-deduped, brand+pkg-aware)"
                )

                # -----------------------------------------------------------
                # FIX 1: Run row_matcher_pipeline PER IMAGE, not once for the
                # whole batch.  Mixing caps from different cooler images into
                # one call makes the column/depth geometry meaningless because
                # SKU box widths and positions are image-specific.
                #
                # FIX 2: Guard against empty per-image cap lists — if no caps
                # were mapped for a given image, skip it instead of letting
                # mapped_caps[0] raise an IndexError that kills the batch.
                # -----------------------------------------------------------

                # Create structural count temp table once before the loop.
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS temp.structural_count_temp (
                        iteration_id    INTEGER,
                        store_id        TEXT,
                        image_file_name TEXT,
                        prod_class_id   INTEGER,
                        sku_name        TEXT,
                        extra_count     INTEGER
                    )
                """)

                # Group caps and SKUs by image_file_name for per-image runs.
                from collections import defaultdict as _dd
                caps_by_image = _dd(list)
                for c in mapped_caps:
                    caps_by_image[c["image_file_name"]].append(c)

                skus_by_image = _dd(list)
                for s in sku_dicts:
                    skus_by_image[s["image_file_name"]].append(s)

                for img_filename, img_caps in caps_by_image.items():
                    # Guard: skip if no caps mapped for this image (Fix 2)
                    if not img_caps:
                        logger.warning(
                            f"No mapped caps for {img_filename} — skipping pipeline"
                        )
                        continue

                    img_skus = skus_by_image.get(img_filename, [])

                    pipeline_result = run_row_matcher_pipeline(
                        img_caps,
                        img_skus,
                        sku_model
                    )

                    # run_row_matcher_pipeline returns None when img_caps is
                    # empty (defensive guard inside the function itself).
                    if pipeline_result is None:
                        logger.warning(
                            f"Pipeline returned no result for {img_filename}"
                        )
                        continue

                    final_counts   = pipeline_result["results"]
                    result_store_id = pipeline_result["store_id"]
                    result_image   = pipeline_result["image_file_name"]

                    logger.info(
                        f"============== FINAL COUNTS [{img_filename}] =============="
                    )

                    for sku_name, data in final_counts.items():
                        final_count   = data["count"]
                        detected_caps = data["detected_caps"]
                        columns       = data["columns"]
                        depth         = data["depth"]
                        count_mode    = data.get("count_mode", "unknown")

                        logger.info(
                            f"[COUNT DEBUG] "
                            f"SKU={sku_name} | "
                            f"CAPS={detected_caps} | "
                            f"COLUMNS={columns} | "
                            f"DEPTH={depth} | "
                            f"FINAL={final_count} | "
                            f"MODE={count_mode}"
                        )

                        # Match caps by prod_class_name (not prod_class_id)
                        # because mapped_caps carry OLD-model IDs after remap
                        # while sku_detections still has NEW-model IDs.
                        caps_for_sku = [
                            c for c in img_caps
                            if c.get("prod_class_name", "").strip().lower()
                            == sku_name.strip().lower()
                        ]
                        detected_for_sku = len(caps_for_sku)
                        extra_needed = final_count - detected_for_sku

                        logger.info(
                            f"[EXTRA ROW DEBUG] "
                            f"SKU={sku_name} | "
                            f"Detected={detected_for_sku} | "
                            f"Final={final_count} | "
                            f"Extra Needed={extra_needed}"
                        )

                        if extra_needed > 0 and caps_for_sku:
                            # Prefer a non-fallback (< 9000) prod_class_id so
                            # extra rows share the same class as real detections.
                            resolved_pid = next(
                                (c["prod_class_id"] for c in caps_for_sku
                                 if c.get("prod_class_id") not in (None, -1)
                                 and c["prod_class_id"] < 9000),
                                caps_for_sku[0].get("prod_class_id")
                            )
                            if resolved_pid is not None:
                                logger.info(
                                    f"Queuing {extra_needed} extra rows for "
                                    f"{sku_name} (prod_class_id={resolved_pid}) "
                                    f"→ will insert in cap_pipeline_runner Step 5"
                                )
                                cur.execute("""
                                    INSERT INTO temp.structural_count_temp
                                    (iteration_id, store_id, image_file_name,
                                     prod_class_id, sku_name, extra_count)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                """, (
                                    iterationid,
                                    result_store_id,
                                    result_image,
                                    int(resolved_pid),
                                    sku_name,
                                    extra_needed,
                                ))

        # Commit all changes
        conn.commit()
        logger.info("Database commit successful")

        # Close database connection
        try:
            close_db_connection(conn, cur)
            logger.info("Database closed after insert")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

        # Log final statistics
        logger.info("Visicooler batch complete")
        logger.info(f"Iteration ID: {iterationid}")
        logger.info(f"Images processed: {total_processed}")
        logger.info(f"SKU detections: {total_sku_detections}")
        logger.info(f"Cap detections: {total_cap_detections}")
        impure_count = sum(1 for r in all_purity_records if r[3] == "IMPURE")
        pure_count = sum(1 for r in all_purity_records if r[3] == "PURE")
        logger.info(f"Purity verdicts (GroundingDINO): {pure_count} PURE, {impure_count} IMPURE")
        logger.info(f"Non-beverage DINO detections logged: {len(all_dino_detection_records)}")
        if dino_failed_count:
            logger.warning(
                f"GroundingDINO check failed on {dino_failed_count} Case-3 "
                f"image(s) — purity_status left NULL for those, no verdict guessed"
            )
        logger.info(f"No-detection (10028) images: {len(no_detection_images)}")

        return []

    except Exception as e:
        logger.error(f"Fatal error: {type(e).__name__}: {e}")

        # Try to close database connection if still open
        if conn is not None:
            try:
                close_db_connection(conn, cur)
            except Exception as close_error:
                logger.warning(f"Error closing connection during cleanup: {close_error}")

        raise