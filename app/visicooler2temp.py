import os
import cv2
from ultralytics import YOLO
import logging
from app.db_handler import initialize_db_connection, close_db_connection, get_classtext
from app.config_loader import load_config
from app.s3_handler import S3Handler
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_overlapping_boxes(boxes, threshold=10):
    """Merge overlapping shelf bounding boxes based on y-coordinates."""
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


def check_visibilitydetails_schema(cur):
    """Check if all required columns exist in orgi.visibilitydetails with correct data types."""
    required_columns = {
        'cyclecountid': {'expected_types': ['integer', 'bigint']},
        'imagefilename': {'expected_types': ['varchar', 'text', 'character varying']},
        'numshelf': {'expected_types': ['integer', 'bigint']},
        'numproducts': {'expected_types': ['integer', 'bigint']},
        'numpureshelf': {'expected_types': ['integer', 'bigint']},
        'coolersize': {'expected_types': ['varchar', 'text', 'character varying']},
        'percentrgb': {'expected_types': ['double precision', 'numeric', 'real']},
        'chilleditems': {'expected_types': ['integer', 'bigint']},
        'warmitems': {'expected_types': ['integer', 'bigint']},
        'skus_detected': {'expected_types': ['varchar', 'text', 'character varying']},
        'share_chilled': {'expected_types': ['double precision', 'numeric', 'real']},
        'share_warm': {'expected_types': ['double precision', 'numeric', 'real']},
        'present_no_facings': {'expected_types': ['varchar', 'text', 'character varying']}
    }
    try:
        columns_list = ', '.join(f"'{col}'" for col in required_columns.keys())
        query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'orgi' AND table_name = 'visibilitydetails'
            AND column_name IN ({columns_list})
        """
        cur.execute(query)
        columns = {row[0]: row[1].lower() for row in cur.fetchall()}

        for col, props in required_columns.items():
            if col not in columns:
                logger.error(
                    f"Column {col} not found in orgi.visibilitydetails. Please add it with: ALTER TABLE orgi.visibilitydetails ADD COLUMN {col} {props['expected_types'][0]};")
                return False
            if columns[col] not in props['expected_types']:
                logger.error(
                    f"Column {col} has unexpected data type: {columns[col]}. Expected one of {props['expected_types']}. Please alter with: ALTER TABLE orgi.visibilitydetails ALTER COLUMN {col} TYPE {props['expected_types'][0]};")
                return False
        return True
    except Exception as e:
        logger.error(f"Failed to check visibilitydetails schema: {e}")
        return False


def upload_to_visibilitydetails(conn, cur, records, cyclecountid):
    """Upload visicooler analysis results to orgi.visibilitydetails."""
    try:
        has_valid_schema = check_visibilitydetails_schema(cur)
        if not has_valid_schema:
            logger.error("Invalid schema for orgi.visibilitydetails. Please ensure all required columns exist.")
            raise Exception("Invalid schema for orgi.visibilitydetails")

        insert_query = """
        INSERT INTO orgi.visibilitydetails
        (imagefilename, numshelf, numproducts, numpureshelf, coolersize, percentrgb,
         chilleditems, warmitems, skus_detected, share_chilled, share_warm, present_no_facings, cyclecountid)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        formatted_records = []
        for record in records:
            (filename, num_shelves, total_products, pure_shelf_count, visicooler_size,
             percent_rgb, chilled_items, warm_items, skus_detected,
             share_chilled, share_warm, present_no_facings) = record
            if not isinstance(filename, str):
                logger.warning(f"Invalid imagefilename type for {filename}: {type(filename)}. Skipping.")
                continue
            if not isinstance(num_shelves, int):
                logger.warning(f"Invalid numshelf type for {filename}: {type(num_shelves)}. Using 0.")
                num_shelves = 0
            if not isinstance(total_products, int):
                logger.warning(f"Invalid numproducts type for {filename}: {type(total_products)}. Using 0.")
                total_products = 0
            if not isinstance(pure_shelf_count, int):
                logger.warning(f"Invalid numpureshelf type for {filename}: {type(pure_shelf_count)}. Using 0.")
                pure_shelf_count = 0
            if not isinstance(visicooler_size, str):
                logger.warning(f"Invalid coolersize type for {filename}: {type(visicooler_size)}. Using 'Unknown'.")
                visicooler_size = "Unknown"
            if not isinstance(percent_rgb, (int, float)):
                logger.warning(f"Invalid percentrgb type for {filename}: {type(percent_rgb)}. Using 0.0.")
                percent_rgb = 0.0
            if not isinstance(chilled_items, int):
                logger.warning(f"Invalid chilleditems type for {filename}: {type(chilled_items)}. Using 0.")
                chilled_items = 0
            if not isinstance(warm_items, int):
                logger.warning(f"Invalid warmitems type for {filename}: {type(warm_items)}. Using 0.")
                warm_items = 0
            if not isinstance(skus_detected, str):
                logger.warning(f"Invalid skus_detected type for {filename}: {type(skus_detected)}. Using 'N'.")
                skus_detected = "N"
            if not isinstance(share_chilled, (int, float)):
                logger.warning(f"Invalid share_chilled type for {filename}: {type(share_chilled)}. Using 0.0.")
                share_chilled = 0.0
            if not isinstance(share_warm, (int, float)):
                logger.warning(f"Invalid share_warm type for {filename}: {type(share_warm)}. Using 0.0.")
                share_warm = 0.0
            if not isinstance(present_no_facings, str):
                logger.warning(
                    f"Invalid present_no_facings type for {filename}: {type(present_no_facings)}. Using 'N'.")
                present_no_facings = "N"
            formatted_records.append((
                filename, num_shelves, total_products, pure_shelf_count, visicooler_size,
                float(percent_rgb), chilled_items, warm_items, skus_detected,
                float(share_chilled), float(share_warm), present_no_facings,
                cyclecountid
            ))

        if not formatted_records:
            logger.warning("No valid records to upload to orgi.visibilitydetails.")
            return

        cur.executemany(insert_query, formatted_records)
        conn.commit()
        logger.info(
            f"Uploaded {len(formatted_records)} records to orgi.visibilitydetails with cyclecountid {cyclecountid}")
    except Exception as e:
        logger.error(f"Failed to upload to visibilitydetails: {e}")
        conn.rollback()
        raise


def run_visicooler_analysis(image_paths, config, s3_handler, conn, cur, output_folder_path, cyclecountid):
    """Run visicooler analysis on images and upload results to database."""
    try:
        shelf_model_path = config['visicooler_config']['caps_model_path']
        sku_model_path = config['yolo_config']['model_path']
        shelf_class_id = config['visicooler_config']['shelf_class_id']
        conf_threshold = config['visicooler_config']['conf_threshold']

        shelf_model = YOLO(shelf_model_path)
        sku_model = YOLO(sku_model_path)

        coca_cola_classes = {
            "Aquafina 1000ml PET",
            "Aquafina 250ml",
            "Aquafina 500ml PET",
            "Coca-Cola 1000ml PET",
            "Coca-Cola 1500ml PET",
            "Coca-Cola 175ml Glass",
            "Coca-Cola 2250ml PET",
            "Coca-Cola 250ml PET",
            "Coca-Cola 500ml PET",
            "Coca-Cola 750ml PET",
            "Coca-Cola Can",
            "Coca-Cola Zero 1000ml PET",
            "Coca-Cola Zero 250ml PET",
            "Coca-Cola Zero 500ml PET",
            "Coca-Cola Zero 750ml Pet",
            "Coca-Cola Zero Can",
            "Coke Cap",
            "Coke Glass Caps",
            "Fanta Apple 250ml PET",
            "Fanta Apple 500ml PET",
            "Fanta Cap",
            "Fanta Glass Cap",
            "Fanta Lemon 250ml PET",
            "Fanta Orange 1000ml PET",
            "Fanta Orange 1500ml PET",
            "Fanta Orange 2250ml PET",
            "Fanta Orange 250ml PET",
            "Fanta Orange 500ml PET",
            "Fanta Orange 700ml PET",
            "Fanta green cap",
            "Fanta lemon 2250ml PET",
            "Fanta red Cap",
            "Kinley Glass cap",
            "Kinley Soda 250ml PET",
            "Kinley Soda 500ml PET",
            "Kinley Soda 750ml PET",
            "Kinley Water 1000ml PET",
            "Kinley Water 500ml PET",
            "Kinley bottle cap",
            "Kinley soda cap",
            "Limca 250ml",
            "Limca 250ml PET",
            "Maaza 1000ml PET",
            "Maaza 250ml PET",
            "Maaza 500ml PET",
            "Sprite  glassCap",
            "Sprite 1000ml PET",
            "Sprite 1500ml PET",
            "Sprite 2250ml PET",
            "Sprite 250ml PET",
            "Sprite 500ml PET",
            "Sprite 750ml PET",
            "Sprite Can",
            "Sprite Cap",
            "Sprite Glass Cap",
            "Sprite Visicooler",
            "Thums Up 250ml",
            "Thums Up 250ml PET",
            "Slice 1000ml PET Bottle",
            "Slice 250ml PET",
            "Slice 500ml PET"
        }

        pepsico_classes = {
            "7Up Nimbooz 2250ml PET",
            "7Up Nimbooz 250ml PET",
            "7up 250ml PET",
            "7up 500ML PET",
            "AGER BEER",
            "Carlsberg Visicooler",
            "Charged Energy Drink 250ml PET",
            "Evervess Club Soda 600ml PET",
            "Lehar Soda 600ml PET",
            "Mirinda Orange 1000ml PET",
            "Mirinda Orange 1500ml PET",
            "Mirinda Orange 2250ml PET",
            "Mirinda Orange 250ml Glass",
            "Mirinda Orange 250ml PET",
            "Mirinda Orange 500ml PET",
            "Mountain Dew 1000ml PET",
            "Mountain Dew 1250ml PET",
            "Mountain Dew 1500ml PET",
            "Mountain Dew 2250ml PET",
            "Mountain Dew 250ml PET",
            "Mountain Dew 500ml PET",
            "Mountain Dew 600ml PET",
            "Mountain Dew 750ml PET",
            "Pepsi 1000ml PET",
            "Pepsi 1500ml PET",
            "Pepsi 2250ml PET",
            "Pepsi 250ml PET",
            "Pepsi 500ml PET",
            "Pepsi 600ml PET",
            "Pepsi 700ml PET",
            "Pepsi Visicooler",
            "Sting 250ml PET",
            "Teem Soda 250ml PET",
            "Tuborg visicooler",
            "Other  Caps",
            "Other CAN",
            "Other Caps",
            "Other Glass Caps",
            "Other PET",
            "Other RGB",
            "Other TPK",
            "Other Water",
            "Other Water Caps",
        }

        rgb_classes = {
            'Coca-Cola 250ml Glass', 'Sprite 250ml Glass', 'Fanta Orange 250ml Glass', 'Pepsi 250ml Glass',
            'Mountain Dew 250ml Glass', 'Mirinda Orange 250ml Glass', 'Slice 250ml Glass', '7Up 250ml Glass Bottle',
            '7up 250ml glass'
        }

        bulk_records = []
        for filesequenceid, storename, filename, local_path, s3_key, storeid in image_paths:
            try:
                image = cv2.imread(local_path)
                if image is None:
                    logger.error(f"Failed to load image: {local_path}")
                    continue
                image_height, image_width = image.shape[:2]

                shelf_results = shelf_model(local_path, conf=conf_threshold)
                shelves = []
                for result in shelf_results:
                    scale_w = image_width / result.orig_shape[1]
                    scale_h = image_height / result.orig_shape[0]
                    for box in result.boxes:
                        if int(box.cls[0]) == shelf_class_id:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
                            shelves.append({
                                "top_y": y1,
                                "bottom_y": y2
                            })

                merged_shelves = merge_overlapping_boxes(shelves)
                num_shelves = len(merged_shelves)
                shelf_regions = []
                for i in range(num_shelves):
                    if i == 0:
                        shelf_regions.append({
                            "shelf_id": i + 1,
                            "top": 0,
                            "bottom": merged_shelves[i]["bottom_y"]
                        })
                    elif i == num_shelves - 1:
                        shelf_regions.append({
                            "shelf_id": i + 1,
                            "top": merged_shelves[i - 1]["top_y"],
                            "bottom": image_height
                        })
                    else:
                        shelf_regions.append({
                            "shelf_id": i + 1,
                            "top": merged_shelves[i - 1]["top_y"],
                            "bottom": merged_shelves[i]["bottom_y"]
                        })

                sku_results = sku_model(local_path, conf=0.35)
                sku_detections = []
                for result in sku_results:
                    scale_w = image_width / result.orig_shape[1]
                    scale_h = image_height / result.orig_shape[0]
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        name = sku_model.names[cls_id]
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
                        center_y = (y1 + y2) // 2
                        sku_detections.append({
                            "name": name,
                            "conf": float(box.conf[0]),
                            "bbox": (x1, y1, x2, y2),
                            "center_y": center_y
                        })

                shelf_sku_map = {region["shelf_id"]: [] for region in shelf_regions}
                shelf_rgb_count = {region["shelf_id"]: 0 for region in shelf_regions}
                shelf_total_count = {region["shelf_id"]: 0 for region in shelf_regions}
                coke_pepsi_chilled = 0
                coke_pepsi_warm = 0
                has_coke_pepsi = False

                for sku in sku_detections:
                    for region in shelf_regions:
                        if region["top"] <= sku["center_y"] <= region["bottom"]:
                            shelf_sku_map[region["shelf_id"]].append(sku)
                            # Count "cap" class vs others per shelf
                            if "cap" in sku["name"].lower():
                                shelf_total_count[region["shelf_id"]] += 1
                            else:
                                shelf_total_count[region["shelf_id"]] += 1
                            if sku["name"] in rgb_classes:
                                shelf_rgb_count[region["shelf_id"]] += 1
                            if sku["name"] in coca_cola_classes or sku["name"] in pepsico_classes:
                                has_coke_pepsi = True
                                if visicooler_size != "Unknown":
                                    coke_pepsi_chilled += 1
                                else:
                                    coke_pepsi_warm += 1
                            break

                pure_shelf_count = 0
                for region in shelf_regions:
                    skus = shelf_sku_map[region["shelf_id"]]
                    if skus:
                        coca_count = sum(1 for sku in skus if sku["name"] in coca_cola_classes)
                        total = len(skus)

                        if total > 0 and ((coca_count / total) * 100) > 60:
                            pure_shelf_count += 1

                if num_shelves <= 3 and num_shelves > 0:
                    visicooler_size = "Small"
                elif num_shelves == 4:
                    visicooler_size = "Medium"
                elif num_shelves >= 5:
                    visicooler_size = "Large"
                else:
                    visicooler_size = "Unknown"

                total_rgb = sum(shelf_rgb_count.values())
                total_products = 0
                for shelf_id in shelf_sku_map:
                    caps_count = sum(1 for s in shelf_sku_map[shelf_id] if "cap" in s["name"].lower())
                    other_count = len(shelf_sku_map[shelf_id]) - caps_count

                    if caps_count > other_count:
                        total_products += max(caps_count, len(shelf_sku_map[shelf_id]))
                    else:
                        total_products += len(shelf_sku_map[shelf_id])
                percent_rgb = (total_rgb / total_products * 100) if total_products > 0 else 0

                if visicooler_size == "Unknown":
                    chilled_items = 0
                    warm_items = len(sku_detections)
                else:
                    chilled_items = len(sku_detections)
                    warm_items = 0

                share_chilled = (coke_pepsi_chilled / chilled_items * 100) if chilled_items > 0 else 0.0
                share_warm = (coke_pepsi_warm / warm_items * 100) if warm_items > 0 else 0.0

                skus_detected = "Y" if sku_detections else "N"
                present_no_facings = "Y" if has_coke_pepsi and total_products == 0 else "N"

                rendered_image = sku_results[0].plot()
                for region in shelf_regions:
                    cv2.line(rendered_image, (0, region["top"]), (image_width, region["top"]), (0, 255, 0), 2)
                    cv2.line(rendered_image, (0, region["bottom"]), (image_width, region["bottom"]), (0, 0, 255), 2)
                output_path = os.path.join(output_folder_path, f"segmented_{filename}")
                cv2.imwrite(output_path, rendered_image)

                s3_key_annotated = f"ModelResults/Visicooler_{cyclecountid}/segmented_{filename}"
                s3_handler.upload_file_to_s3(output_path, s3_key_annotated)
                logger.info(f"Uploaded segmented image to S3: {s3_key_annotated}")

                bulk_records.append((
                    filename, num_shelves, total_products, pure_shelf_count, visicooler_size,
                    percent_rgb, chilled_items, warm_items, skus_detected,
                    share_chilled, share_warm, present_no_facings
                ))

            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")
                continue

        if bulk_records:
            upload_to_visibilitydetails(conn, cur, bulk_records, cyclecountid)
        else:
            logger.warning("No visicooler records to upload.")
        return bulk_records
    except Exception as e:
        logger.error(f"Error in visicooler analysis: {e}")
        raise