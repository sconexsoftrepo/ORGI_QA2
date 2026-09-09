"""
planogram_detector.py — shelf-segmented SKU-only planogram detector for
subcategory-603 images.

Each subcategory-603 image is a FULL visicooler photo covering several
physical shelves. The pipeline is:

    1. Run the shelf-segmentation model (shelfmodel.pt) on the full image
       to find shelf divider lines, merge overlapping ones, and derive
       top-to-bottom shelf regions (region 1 = topmost shelf, region 2 =
       next one down, etc.) — exactly the detect_shelf_regions()/
       merge_overlapping_boxes() logic from the reference shelf-test
       script.
    2. Crop the image to each shelf region.
    3. Run the SKU/availability model (availability_may_5th.pt) on ONLY
       that crop — never the full image — and save an annotated copy of
       the crop.
    4. Every detected box (offset back into full-image y-coordinates) is
       one row in temp.sku_planogram_temp, with shelfnumber = that
       region's shelf_id (1 = topmost). image_file_name is the per-SHELF
       name ("IMG-XXXX_shelf1.jpg", "IMG-XXXX_shelf2.jpg", ...), matching
       the annotated crop 1:1 — NOT the single original uploaded filename
       repeated across every shelf of that photo.

No cap model, no cap<->SKU matching, no GroundingDINO purity check —
the availability model's own detections are the final placement.

Each detected box's class id is translated from the SKU model's native
class-id space into the catalog's product-class-id space using the same
`_NEW_TO_OLD_CLASS_MAP` that `cap_sku_mapper` applies to cap detections,
since these rows are consumed directly (not just for other/alcohol
filtering) and must line up with `orgi.productmaster`.

Writes into temp.sku_planogram_temp (created if missing):

    CREATE TABLE temp.sku_planogram_temp (
        store_id int8 NOT NULL,
        image_file_name text NOT NULL,
        iteration_id int4 NOT NULL,
        prod_class_id int4 NOT NULL,
        x1 numeric(12, 4) NOT NULL,
        x2 numeric(12, 4) NOT NULL,
        y1 numeric(12, 4) NOT NULL,
        y2 numeric(12, 4) NOT NULL,
        shelfnumber int4 NULL,
        brand_name varchar(255) NULL,
        s3path_annotated_file text NULL,
        CONSTRAINT sku_planogram_temp_pkey PRIMARY KEY (
            store_id, image_file_name, iteration_id, prod_class_id, x1, x2, y1, y2
        )
    );

Usage (called once per batch from main.py's execute_models(), alongside
run_visicooler_analysis and run_visicooler_presence_analysis):

    from planogram_detector import run_planogram_detection

    inserted = run_planogram_detection(
        image_paths=image_paths,
        config=config,
        s3_handler=s3_handler,
        db_config=db_config,
        cyclecountid=cyclecountid,
        iterationid=iterationid,
    )
"""

import os
import logging

import cv2

from app.db_handler import initialize_db_connection, close_db_connection
from app.visicooler import should_ignore_class, extract_brand_from_name
from app.cap_sku_mapper import _NEW_TO_OLD_CLASS_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_PLANOGRAM_SUBCATEGORY_ID = 603

_PLANOGRAM_SHELF_YOLO = None
_PLANOGRAM_SKU_YOLO = None


def get_planogram_shelf_yolo(model_path):
    global _PLANOGRAM_SHELF_YOLO
    if _PLANOGRAM_SHELF_YOLO is None:
        from ultralytics import YOLO
        logger.info(f"Loading planogram shelf-segmentation YOLO: {model_path}")
        _PLANOGRAM_SHELF_YOLO = YOLO(model_path)
    return _PLANOGRAM_SHELF_YOLO


def get_planogram_sku_yolo(model_path):
    global _PLANOGRAM_SKU_YOLO
    if _PLANOGRAM_SKU_YOLO is None:
        from ultralytics import YOLO
        logger.info(f"Loading planogram SKU/availability YOLO: {model_path}")
        _PLANOGRAM_SKU_YOLO = YOLO(model_path)
    return _PLANOGRAM_SKU_YOLO


def ensure_sku_planogram_temp_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS temp.sku_planogram_temp (
            store_id int8 NOT NULL,
            image_file_name text NOT NULL,
            iteration_id int4 NOT NULL,
            prod_class_id int4 NOT NULL,
            x1 numeric(12, 4) NOT NULL,
            x2 numeric(12, 4) NOT NULL,
            y1 numeric(12, 4) NOT NULL,
            y2 numeric(12, 4) NOT NULL,
            shelfnumber int4 NULL,
            brand_name varchar(255) NULL,
            s3path_annotated_file text NULL,
            CONSTRAINT sku_planogram_temp_pkey PRIMARY KEY (
                store_id, image_file_name, iteration_id, prod_class_id, x1, x2, y1, y2
            )
        )
    """)
    logger.info("Verified temp.sku_planogram_temp exists")


def _remap_class_id(cls_id: int) -> int:
    """Same new-model->old-model translation cap_sku_mapper applies to
    caps, applied here directly to the SKU model's own class id since
    there's no cap-matching step to do it for us."""
    return _NEW_TO_OLD_CLASS_MAP.get(cls_id, cls_id)


# ── Shelf-region segmentation (shelfmodel.pt) ────────────────────────────────
def _merge_overlapping_boxes(boxes, threshold=10):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b["top_y"])
    merged = [boxes[0]]
    for box in boxes[1:]:
        prev = merged[-1]
        if box["top_y"] <= prev["bottom_y"] + threshold:
            prev["bottom_y"] = max(prev["bottom_y"], box["bottom_y"])
        else:
            merged.append(box)
    return merged


def detect_shelf_regions(image_path, shelf_model, shelf_class_id,
                          conf_threshold, merge_threshold=10):
    """
    Returns (shelf_regions, merged_shelves, image).

    shelf_regions: list of {"shelf_id": N, "top": y, "bottom": y}, ordered
    top-to-bottom, shelf_id starting at 1 (1 = topmost physical shelf).
    Falls back to a single region spanning the whole image if the shelf
    model finds nothing, so every image still yields at least one shelf.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    image_height, _image_width = image.shape[:2]
    shelf_results = shelf_model(image_path, conf=conf_threshold)

    shelves = []
    for result in shelf_results:
        scale_h = image_height / result.orig_shape[0]
        for box in result.boxes:
            cls_id = int(box.cls[0])
            try:
                name = shelf_model.names[cls_id]
            except Exception:
                name = ""
            if cls_id == shelf_class_id or "shelf" in name.lower():
                _x1, y1, _x2, y2 = box.xyxy[0]
                y1 = int(y1 * scale_h)
                y2 = int(y2 * scale_h)
                shelves.append({"top_y": y1, "bottom_y": y2})

    merged_shelves = _merge_overlapping_boxes(shelves, threshold=merge_threshold)

    if len(merged_shelves) == 0:
        logger.warning(f"No shelf lines detected in {os.path.basename(image_path)} — using whole image as 1 shelf")
        shelf_regions = [{"shelf_id": 1, "top": 0, "bottom": image_height}]
    else:
        shelf_regions = []
        shelf_id = 1
        shelf_regions.append({"shelf_id": shelf_id, "top": 0, "bottom": merged_shelves[0]["top_y"]})
        shelf_id += 1
        for i in range(len(merged_shelves) - 1):
            shelf_regions.append({
                "shelf_id": shelf_id,
                "top": merged_shelves[i]["bottom_y"],
                "bottom": merged_shelves[i + 1]["top_y"],
            })
            shelf_id += 1
        shelf_regions.append({
            "shelf_id": shelf_id,
            "top": merged_shelves[-1]["bottom_y"],
            "bottom": image_height,
        })

    return shelf_regions, merged_shelves, image


def _norm_storeid(sid):
    if sid is None:
        return None
    if isinstance(sid, str):
        s = sid.strip()
        if s.isdigit():
            return int(s)
        return s
    return sid


def run_planogram_detection(
    image_paths,
    config,
    s3_handler,
    db_config,
    cyclecountid,
    iterationid,
    subcategory_id=DEFAULT_PLANOGRAM_SUBCATEGORY_ID,
):
    """
    End-to-end: filter image_paths down to `subcategory_id` (default 603),
    segment each image into shelf regions, run the SKU/availability model
    on each region's crop, upload an annotated copy of each crop to S3,
    and insert every detected box (offset into full-image y-coordinates)
    into temp.sku_planogram_temp.

    Returns the number of rows inserted.
    """
    planogram_cfg = config.get('planogram_config', {})
    visicooler_cfg = config.get('visicooler_config', {})

    shelf_model_path = planogram_cfg.get('shelf_model_path', 'data/shelfmodel.pt')
    shelf_class_id = int(planogram_cfg.get('shelf_class_id', 80))
    shelf_conf_threshold = float(planogram_cfg.get('shelf_conf_threshold', 0.05))
    shelf_merge_threshold = int(planogram_cfg.get('shelf_merge_threshold', 10))

    sku_model_path = planogram_cfg.get('sku_model_path') or visicooler_cfg.get('sku_model_path')
    availability_conf = float(
        planogram_cfg.get('availability_conf', planogram_cfg.get('sku_conf_threshold', 0.25))
    )

    output_folder_path = planogram_cfg.get('output_folder_path', 'outputs/planogram_segmented_images/')
    output_s3_folder = planogram_cfg.get('output_s3_folder', 'ModelResults/Planogram')

    if not shelf_model_path:
        logger.error("No shelf-segmentation model path configured for planogram detection.")
        return 0
    if not sku_model_path:
        logger.error("No SKU/availability model path configured for planogram detection.")
        return 0

    target_images = [img for img in image_paths if img[6] == subcategory_id]
    logger.info(
        f"Planogram detector: {len(target_images)} of {len(image_paths)} images "
        f"are subcategory {subcategory_id}"
    )
    if not target_images:
        return 0

    shelf_model = get_planogram_shelf_yolo(shelf_model_path)
    sku_model = get_planogram_sku_yolo(sku_model_path)
    sku_class_names = sku_model.names

    os.makedirs(output_folder_path, exist_ok=True)

    all_records = []  # tuples matching temp.sku_planogram_temp column order
    for row in target_images:
        fileseqid, storename, filename, local_path, s3_key, storeid, subcat = row
        storeid = _norm_storeid(storeid)
        name, ext = os.path.splitext(filename)
        ext = ext or ".jpg"

        try:
            shelf_regions, merged_shelves, image = detect_shelf_regions(
                local_path, shelf_model, shelf_class_id,
                shelf_conf_threshold, shelf_merge_threshold,
            )
        except Exception as e:
            logger.error(f"Shelf segmentation failed for {filename}: {e}", exc_info=True)
            continue

        logger.info(f"{filename}: segmented into {len(shelf_regions)} shelf region(s)")

        for region in shelf_regions:
            shelf_id = region["shelf_id"]
            top = max(0, region["top"])
            bottom = min(image.shape[0], region["bottom"])
            if bottom <= top:
                continue

            crop = image[top:bottom, :]
            shelf_image_name = f"{name}_shelf{shelf_id}{ext}"
            s3path_annotated = (
                f"{output_s3_folder}_{cyclecountid}/segmented_{shelf_image_name}"
            )

            try:
                results = sku_model.predict(source=crop, conf=availability_conf, verbose=False)
            except Exception as e:
                logger.error(f"Availability model failed on {filename} shelf {shelf_id}: {e}", exc_info=True)
                continue

            shelf_records = []
            annotated_crop = crop
            for result in results:
                try:
                    annotated_crop = result.plot()
                except Exception:
                    pass

                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if should_ignore_class(cls_id, sku_class_names):
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1_px, x2_px = int(x1), int(x2)
                    # Offset y back into full-image coordinates — crop is
                    # full width, only cropped vertically.
                    y1_px = int(y1) + top
                    y2_px = int(y2) + top

                    product_name = sku_class_names[cls_id]
                    brand_name = extract_brand_from_name(product_name)
                    prod_class_id = _remap_class_id(cls_id)

                    shelf_records.append((
                        storeid,
                        shelf_image_name,
                        iterationid,
                        prod_class_id,
                        x1_px,
                        x2_px,
                        y1_px,
                        y2_px,
                        shelf_id,
                        brand_name,
                        s3path_annotated,
                    ))

            if shelf_records:
                all_records.extend(shelf_records)

            # Save + upload the annotated shelf crop regardless of whether
            # any box survived should_ignore_class filtering (matches the
            # 605 flow's SKU-annotation convention).
            try:
                out_path = os.path.join(output_folder_path, f"segmented_{shelf_image_name}")
                cv2.imwrite(out_path, annotated_crop)
                s3_handler.upload_file_to_s3(out_path, s3path_annotated)
                logger.info(f"Uploaded planogram shelf-annotated image: segmented_{shelf_image_name}")
            except Exception as annot_err:
                logger.warning(f"Planogram annotation upload failed for {filename} shelf {shelf_id}: {annot_err}")

    if not all_records:
        logger.info("Planogram detector: no SKU boxes detected across target shelf regions")
        return 0

    conn, cur = initialize_db_connection(db_config)
    try:
        ensure_sku_planogram_temp_table(cur)
        conn.commit()

        insert_sql = """
            INSERT INTO temp.sku_planogram_temp (
                store_id, image_file_name, iteration_id, prod_class_id,
                x1, x2, y1, y2, shelfnumber, brand_name, s3path_annotated_file
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (store_id, image_file_name, iteration_id, prod_class_id, x1, x2, y1, y2)
            DO NOTHING
        """
        cur.executemany(insert_sql, all_records)
        conn.commit()
        logger.info(f"Inserted {len(all_records)} planogram SKU rows into temp.sku_planogram_temp")
        return len(all_records)
    except Exception as e:
        logger.error(f"Planogram DB insert failed: {e}", exc_info=True)
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        close_db_connection(conn, cur)