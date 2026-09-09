import os
import json
import csv
import logging
import re
import requests
from datetime import datetime
from ollama import Client
from ultralytics import YOLO

from app.config_loader import load_config, load_json_classes
from app.db_handler import (
    initialize_db_connection,
    close_db_connection,
    get_classtext,
    insert_ollama_results
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


_OLLAMA_CLIENT = None

def get_ollama_client(ollama_host):
    """
    Reuse same Ollama client connection to reduce overhead.
    """
    global _OLLAMA_CLIENT
    if _OLLAMA_CLIENT is None:
        _OLLAMA_CLIENT = Client(host=ollama_host)
        logger.info(f"Initialized persistent Ollama client for {ollama_host}")
    return _OLLAMA_CLIENT


def check_ollama_server(ollama_host, model_name):
    try:
        r = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        models = r.json().get("models", [])
        return any(m["name"] == model_name for m in models)
    except Exception as e:
        logger.error(f"Ollama server check failed: {e}")
        return False


def extract_json(text):
    if not text:
        return {}
    txt = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def ollama_generate(ollama_host, model_name, prompt, image_path):
    """
    Call Ollama with persistent client connection.
    """
    client = get_ollama_client(ollama_host)
    try:
        r = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return ONLY a JSON object."},
                {"role": "user", "content": prompt, "images": [image_path]},
            ],
            format="json",
            options={"temperature": 0}
        )
        return extract_json(r["message"]["content"])
    except Exception as e:
        logger.warning(f"Ollama retry: {e}")
        try:
            r = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Return ONLY a JSON object."},
                    {"role": "user", "content": prompt, "images": [image_path]},
                ],
                options={"temperature": 0}
            )
            return extract_json(r["message"]["content"])
        except Exception:
            return {}

_ACTIVATION_YOLO = None

def get_activation_yolo(model_path):
    global _ACTIVATION_YOLO
    if _ACTIVATION_YOLO is None:
        logger.info(f"Loading activation YOLO: {model_path}")
        _ACTIVATION_YOLO = YOLO(model_path)
    return _ACTIVATION_YOLO


def run_activation_yolo(image_path, model_path, conf_threshold=0.3):
    """
    Run activation YOLO to quickly detect common items.
    Returns set of detected class names.
    """
    model = get_activation_yolo(model_path)
    results = model(image_path, conf=conf_threshold, verbose=False)

    detected = set()
    for r in results:
        for cls in r.boxes.cls:
            class_name = model.names[int(cls)].lower().replace(" ", "_")
            detected.add(class_name)
            logger.info(f"Activation YOLO detected: {class_name} in {os.path.basename(image_path)}")
    
    return detected


def analyze_image(image_path, ollama_host, prompts, class_ids, model_name):
    
    merged = {}

    if "extended_visibility_all" in prompts and prompts["extended_visibility_all"]:
        # Single Ollama call for all visibility items
        result = ollama_generate(ollama_host, model_name, prompts["extended_visibility_all"], image_path)
        merged.update(result)
        logger.debug(f"Used merged prompt for {os.path.basename(image_path)}")
    else:
        # Multiple prompts if needed
        for p in ["extended_visibility_group1", "extended_visibility_group2"]:
            if p in prompts and prompts[p]:
                result = ollama_generate(ollama_host, model_name, prompts[p], image_path)
                merged.update(result)

    # Return only class IDs in scope (1018-1046)
    return {k: v for k, v in merged.items() if k in class_ids}


def process_single_image(image_info, ollama_host, prompts, class_ids, model_name, 
                         activation_yolo_model, activation_conf_threshold, 
                         activation_mappings, s3_handler, s3_annotated_folder, 
                         classtext_cache):
    
    fileseqid, storename, filename, local_path, s3_key, storeid, subcategory_id = image_info
    
    # Skip visicooler-specific subcategories
    if subcategory_id in [601, 602, 603, 604, 605]:
        return []
    
    if not os.path.exists(local_path):
        return []

    results = []
    
    try:
        logger.info(f"Processing: {filename}")
        
        # Fast pre-filter with activation YOLO
        activation_detected = set()
        if activation_yolo_model:
            activation_detected = run_activation_yolo(
                local_path, 
                activation_yolo_model, 
                conf_threshold=activation_conf_threshold
            )
        
        # Check activation YOLO detections against mappings
        # Only skip Ollama if we have at least one confidently-mapped detection
        # AND no unmapped/unknown detections that Ollama should evaluate
        ollama_output = {}
        has_mapped = False
        has_unmapped = False

        for detected_name in activation_detected:
            if detected_name in activation_mappings:
                cid = str(activation_mappings[detected_name])
                ollama_output[cid] = "Y"
                has_mapped = True
                logger.info(f"   → Activation mapped '{detected_name}' → class {cid}")
            else:
                # 'others' or any unknown class — let Ollama decide
                has_unmapped = True
                logger.info(f"   → Unmapped detection '{detected_name}' → will run Ollama")

        skip_ollama = has_mapped and not has_unmapped and len(activation_detected) > 0

        # Run Ollama only if needed
        if skip_ollama:
            logger.info(f"  FAST PATH: Skipped Ollama — all detections mapped by activation model")
        else:
            if not activation_detected:
                logger.info(f"  No activation detections — running Ollama")
            ollama_output = analyze_image(
                local_path, ollama_host, prompts, class_ids, model_name
            )
            logger.info(f"    Ollama completed")
        
        # Generate results
        now = datetime.now()
        s3_annot = f"{s3_annotated_folder}/{filename}"
        rowid = 1
        
        for cid, val in ollama_output.items():
            if cid not in class_ids:
                continue
            
            inference = 1.0
            
            results.append({
                "rowid": rowid,
                "modelname": model_name,
                "imagefilename": filename,
                "classid": int(cid),
                "classtext": classtext_cache.get(int(cid), 'Unknown'),  
                "value": val,
                "inference": inference,
                "modelrun": now,
                "processed_flag": "N",
                "storeid": storeid,
                "storename": storename,
                "s3path_actual_file": s3_key,
                "s3path_annotated_file": s3_annot
            })
            rowid += 1
        
        # Upload to S3
        s3_handler.upload_file_to_s3(local_path, s3_annot)
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
    
    return results


def run_ollama_analysis(
    image_paths,
    image_folder,
    output_csv,
    config_path,
    class_ids_path,
    ollama_host,
    s3_handler,
    s3_annotated_folder,
    db_config,
    cyclecountid,
    stagingid
):
    """
    Run Ollama analysis on images and insert results into database.
    stagingid is now passed in from main.py (shared across all batches).
    """
    
    config = load_config(config_path)
    ollama_cfg = config["ollama_config"]
    model_name = ollama_cfg["ollama_model"]
    prompts = ollama_cfg["prompts"]
    activation_yolo_model = ollama_cfg.get("activation_yolo_model")
    activation_conf_threshold = ollama_cfg.get("activation_conf_threshold", 0.3)

    if not check_ollama_server(ollama_host, model_name):
        logger.error("Ollama not available")
        return [], None

    class_ids = load_json_classes(class_ids_path)

    # Open database briefly to cache classtext values
    import pg8000.dbapi as pg
    conn = pg.connect(
        host=db_config['host'],
        port=db_config['port'],
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password']
    )
    cur = conn.cursor()
    logger.info("Database connection established")
    
    # Pre-fetch all classtext values (1018-1053)
    classtext_cache = {}
    for cid in range(1018, 1054):
        classtext_cache[cid] = get_classtext(cur, cid)
    
    # Close database before inference
    close_db_connection(conn, cur)
    logger.info("Database closed before inference")

    # Map activation YOLO class names → visibility class IDs
    
    activation_mappings = {
        "bar_signage":      1024,
        "signage":          1024,   # same as bar signage
        "combo_board":      1022,
        "dps":              1053,
        "menu_board":       1023,
        "pillar_branding":  1040,
        "poster":           1019,
        "rgb_crate_stacking": 1056,
        "shelf_display":    1064,
        "table_sticker":    1027,
        "streamer":         1020,
        # 'others' intentionally omitted 
    }

    all_results = []

    # Filter images for processing
    images_to_process = [
        img for img in image_paths 
        if img[6] not in [601, 602, 603, 604, 605] and os.path.exists(img[3])
    ]
    
    total_images = len(images_to_process)
    logger.info(f"Processing {total_images} images sequentially")

    # Run inference without database connection
    for img_info in images_to_process:
        results = process_single_image(
            img_info,
            ollama_host,
            prompts,
            class_ids,
            model_name,
            activation_yolo_model,
            activation_conf_threshold,
            activation_mappings,
            s3_handler,
            s3_annotated_folder,
            classtext_cache
        )
        all_results.extend(results)

    # Log statistics
    logger.info("=" * 60)
    logger.info("OLLAMA ANALYSIS STATISTICS:")
    logger.info(f"Total images processed: {total_images}")
    logger.info(f"Total results generated: {len(all_results)}")
    logger.info(f"Using stagingid: {stagingid} (shared across all batches)")
    logger.info(f"Scope: Extended visibility items only (class IDs 1018-1046)")
    logger.info("=" * 60)

    # Reopen database for insert
    logger.info("Reopening database for insert...")
    conn = pg.connect(
        host=db_config['host'],
        port=db_config['port'],
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password']
    )
    cur = conn.cursor()

    # Verify connection is alive
    try:
        cur.execute("SELECT 1")
    except Exception as e:
        logger.warning(f"Connection issue, reconnecting: {e}")
        close_db_connection(conn, cur)
        conn = pg.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        cur = conn.cursor()

    # Fetch max rowid already stored for this stagingid so subsequent batches
    # never collide with rows from earlier batches.
    # pg8000 may return a decimal.Decimal — cast to int explicitly.
    cur.execute(
        "SELECT COALESCE(MAX(rowid), 0) FROM orgi.visibilityitemsstaging WHERE stagingid = %s",
        (stagingid,)
    )
    max_existing_rowid = int(cur.fetchone()[0])
    logger.info(
        f"Max existing rowid for stagingid {stagingid}: {max_existing_rowid} "
        f"— new rows will start from {max_existing_rowid + 1}"
    )

    # Assign globally-unique rowids continuing from where the last batch left off
    for idx, result in enumerate(all_results, start=max_existing_rowid + 1):
        result['rowid'] = idx

    # Save CSV (after rowids are finalised)
    if all_results:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    # Insert into database using the shared stagingid.
    # Pass conn + db_config so insert_ollama_results can reconnect mid-insert
    # if the TCP connection is dropped during a long executemany call (WinError 10053).
    insert_ollama_results(
        cur, stagingid, all_results, model_name, s3_annotated_folder, image_paths,
        conn=conn, db_config=db_config
    )
    # Final commit covers any stragglers; per-chunk commits already happened inside
    try:
        conn.commit()
    except Exception:
        pass  # connection may have been replaced during chunked insert
    close_db_connection(conn, cur)

    logger.info("Database insert completed")

    return all_results, output_csv