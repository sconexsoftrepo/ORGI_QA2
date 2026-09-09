import base64
import csv
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime, date

import boto3
import requests
from botocore.exceptions import ClientError
import pg8000.dbapi as pg

# ---------------------------------------------------------------------------
# Logging  — FIX #1: force UTF-8 on both handlers so the arrow character
#            (and any other non-ASCII) never causes a UnicodeEncodeError on
#            Windows cp1252 consoles.
# ---------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("outputs/llm_rescorer.log", encoding="utf-8"),
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
ACTIVATION_SCORE_THRESHOLD = 10       # stores with score < this value are re-scored
LLM_CATEGORY_IDS = (1, 2)             # fetch images where category_id IN (1, 2)
MODELNAME = "gpt-4o-rescorer"

# Class IDs this LLM scores.
# MUST exactly match the 19 classes defined in extended_group1.txt —
# the LLM returns class_id integers from this set; any ID not listed here
# is silently dropped, which is why all 19 must be present.
LLM_CLASS_IDS = {
    "1019": "POSTER",
    "1020": "STREAMER",
    "1081": "DANGLER",
    "1082": "WOBBLER",
    "1022": "WALL_BRANDING",
    "1083": "FRONT_WINDOW_BRANDING",
    "1084": "CEILING_PILLAR_BRANDING",
    "1040": "PILLAR_BRANDING",
    "1085": "COUNTER_FRONT",
    "1086": "SHELF_RACK_BRANDING",
    "1027": "OFFER_RIBBON_STICKER",
    "1087": "MT_RACK_DISPLAY",
    "1092": "SHOPPER_GATE",
    "1091": "LED_ELEMENT",
    "1064": "SHELF_DISPLAY_MT",
    "1088": "NECK_RINGER",
    "1089": "BOX_DISPLAY",
    "1090": "COOLER_STRIPS",
    "1080": "FOAM_BANNER",
}


# ===========================================================================
# Config loader
# ===========================================================================
def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def load_prompt(prompt_path):
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.info(f"Loaded prompt from {prompt_path}")
    return content


# ===========================================================================
# Database helpers
# ===========================================================================
def db_connect(db_config):
    return pg.connect(
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["database"],
        user=db_config["user"],
        password=db_config["password"],
        timeout=60,
    )


def db_close(conn, cur):
    try:
        if cur:
            cur.close()
        if conn:
            conn.close()
    except Exception:
        pass


def get_low_activation_stores(db_config, threshold=ACTIVATION_SCORE_THRESHOLD):
    """
    Return a list of store IDs from orgi.scorecard where, in the current
    month and year, activation_score < threshold.
    """
    today = date.today()
    current_month = today.month
    current_year = today.year

    conn = db_connect(db_config)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT DISTINCT storeid
            FROM orgi.scorecard
            WHERE month = %s
              AND year  = %s
              AND activation_score < %s
            ORDER BY storeid
            """,
            (current_month, current_year, threshold),
        )
        rows = cur.fetchall()
        store_ids = [row[0] for row in rows]
        logger.info(
            f"Found {len(store_ids)} stores with activation_score < {threshold} "
            f"in {current_month}/{current_year}"
        )
        return store_ids
    except Exception as e:
        logger.error(f"Failed to fetch low-activation stores: {e}")
        raise
    finally:
        db_close(conn, cur)


def get_images_for_stores(db_config, store_ids):
    """
    Return rows from orgi.fileupload for the given store_ids where
    category_id IN (1, 2).

    Returns list of tuples:
        (filesequenceid, storename, filename, storeid, category_id)
    """
    if not store_ids:
        return []

    conn = db_connect(db_config)
    cur = conn.cursor()
    try:
        placeholders = ",".join(["%s"] * len(store_ids))
        cat_placeholders = ",".join(["%s"] * len(LLM_CATEGORY_IDS))
        cur.execute(
            f"""
            SELECT filesequenceid,
                   storename,
                   filename,
                   storeid,
                   category_id
            FROM orgi.fileupload
            WHERE storeid IN ({placeholders})
              AND category_id IN ({cat_placeholders})
            ORDER BY storeid, uploadtimestamp ASC
            """,
            (*store_ids, *LLM_CATEGORY_IDS),
        )
        rows = cur.fetchall()
        logger.info(
            f"Found {len(rows)} images across {len(store_ids)} stores "
            f"(category_id IN {LLM_CATEGORY_IDS})"
        )
        return rows
    except Exception as e:
        logger.error(f"Failed to fetch images for stores: {e}")
        raise
    finally:
        db_close(conn, cur)


def get_stagingid(db_config):
    """Return max(stagingid) + 1 from orgi.visibilityitemsstaging."""
    conn = db_connect(db_config)
    cur = conn.cursor()
    try:
        cur.execute("SELECT COALESCE(MAX(stagingid), 0) FROM orgi.visibilityitemsstaging")
        val = int(cur.fetchone()[0])
        stagingid = val + 1
        logger.info(f"Using stagingid: {stagingid}")
        return stagingid
    finally:
        db_close(conn, cur)


def get_max_rowid(conn, cur, stagingid):
    """Return max rowid already inserted for this stagingid (0 if none)."""
    cur.execute(
        "SELECT COALESCE(MAX(rowid), 0) FROM orgi.visibilityitemsstaging WHERE stagingid = %s",
        (stagingid,),
    )
    return int(cur.fetchone()[0])


def get_already_inserted(conn, cur, stagingid):
    """Return set of (imagefilename, classid) already in DB for this stagingid."""
    cur.execute(
        "SELECT imagefilename, classid FROM orgi.visibilityitemsstaging WHERE stagingid = %s",
        (stagingid,),
    )
    return set((row[0], int(row[1])) for row in cur.fetchall())


def insert_results_chunked(conn, cur, stagingid, results, db_config, chunk_size=20):
    """
    Insert result rows into orgi.visibilityitemsstaging in chunks.
    Commits after each chunk. Auto-reconnects on network errors.
    """
    insert_query = """
    INSERT INTO orgi.visibilityitemsstaging
    (stagingid, rowid, modelname, imagefilename, classid, classtext, value, inference,
     modelrun, processed_flag, storeid, storename, s3path_actual_file, s3path_annotated_file)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    records = [
        (
            stagingid,
            r["rowid"],
            r["modelname"],
            r["imagefilename"],
            r["classid"],
            r["classtext"],
            r["value"],
            r["inference"],
            r["modelrun"],
            r["processed_flag"],
            r["storeid"],
            r["storename"],
            r["s3path_actual_file"],
            r["s3path_annotated_file"],
        )
        for r in results
    ]

    total = len(records)
    inserted = 0
    network_keywords = [
        "network", "connection", "timeout", "broken pipe",
        "reset by peer", "timed out", "10053", "10054", "10060",
    ]

    for chunk_start in range(0, total, chunk_size):
        chunk = records[chunk_start: chunk_start + chunk_size]
        for attempt in range(1, 4):
            try:
                cur.executemany(insert_query, chunk)
                conn.commit()
                inserted += len(chunk)
                logger.info(
                    f"Inserted rows {chunk_start + 1}-{chunk_start + len(chunk)} of {total}"
                )
                break
            except Exception as e:
                is_net = any(kw in str(e).lower() for kw in network_keywords)
                if is_net and attempt < 3:
                    wait = 5 * attempt
                    logger.warning(f"Network error on chunk, reconnecting in {wait}s: {e}")
                    time.sleep(wait)
                    try:
                        db_close(conn, cur)
                    except Exception:
                        pass
                    new_conn = db_connect(db_config)
                    new_cur = new_conn.cursor()
                    conn.__dict__.update(new_conn.__dict__)
                    cur.__dict__.update(new_cur.__dict__)
                else:
                    logger.error(f"Failed to insert chunk: {e}")
                    raise

    logger.info(f"Total inserted: {inserted} rows (stagingid={stagingid})")


# ===========================================================================
# S3 helpers
# ===========================================================================
def build_s3_client(s3_config):
    return boto3.client(
        "s3",
        aws_access_key_id=s3_config["access_key"],
        aws_secret_access_key=s3_config["secret_key"],
        region_name=s3_config["region"],
    )


def sanitize_filename(filename):
    base = os.path.basename(filename)
    base = re.sub(r"\.rf\.[0-9a-fA-F]+$", "", base)
    base = re.sub(r'[\\:*?"<>|]', "", base)
    return base


def download_images(s3_client, s3_config, temp_dir, image_rows):
    """
    Download images from S3 for the given fileupload rows.

    image_rows : list of (filesequenceid, storename, filename, storeid, category_id)

    Returns
    -------
    image_paths : list of
        (filesequenceid, storename, clean_filename, local_path, s3_key, storeid, category_id)
    failed      : list of (filesequenceid, storename, filename)
    """
    bucket = s3_config["bucket_name"]
    image_folder = s3_config["image_folder_s3"]

    image_paths = []
    failed = []
    total = len(image_rows)
    downloaded = 0

    for filesequenceid, storename, filename, storeid, category_id in image_rows:
        try:
            clean_filename = sanitize_filename(filename)
            local_path = os.path.join(temp_dir, clean_filename)
            base_filename = os.path.basename(filename)
            clean_storename = re.sub(r'[\\:*?"<>|]', "", storename.replace(":", ".").strip())

            candidate_keys = [
                f"{image_folder}{clean_storename}/{base_filename}",
                f"{image_folder}{base_filename}",
                base_filename,
            ]

            success = False
            for s3_key in candidate_keys:
                try:
                    s3_client.download_file(bucket, s3_key, local_path)
                    downloaded += 1
                    if downloaded % 10 == 0 or downloaded == total:
                        logger.info(f"Downloaded {downloaded}/{total} images")
                    image_paths.append(
                        (filesequenceid, storename, clean_filename,
                         local_path, s3_key, storeid, category_id)
                    )
                    success = True
                    break
                except ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        continue
                    logger.error(f"S3 error for {s3_key}: {e}")
                    failed.append((filesequenceid, storename, filename))
                    break

            if not success:
                logger.warning(f"File not found in S3: {filename}")
                failed.append((filesequenceid, storename, filename))

        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            failed.append((filesequenceid, storename, filename))

    logger.info(f"Download complete: {downloaded}/{total} successful, {len(failed)} failed")
    return image_paths, failed


def upload_to_s3(s3_client, s3_config, local_path, s3_key):
    import mimetypes
    bucket = s3_config["bucket_name"]
    content_type, _ = mimetypes.guess_type(local_path)
    extra = {"ContentType": content_type} if content_type else {}
    s3_client.upload_file(local_path, bucket, s3_key, ExtraArgs=extra)


# ===========================================================================
# LLM helpers
# ===========================================================================
def encode_image(image_path, max_px=1024):
    """
    Resize image so long side <= max_px and return (base64_str, mime_type).
    Falls back to raw bytes if Pillow is not installed.
    """
    try:
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        w, h = img.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)

        buf = io.BytesIO()
        fmt = img.format or "JPEG"
        if fmt not in ("JPEG", "PNG", "GIF", "WEBP"):
            fmt = "JPEG"
        img.save(buf, format=fmt)
        buf.seek(0)
        mime = {"JPEG": "image/jpeg", "PNG": "image/png",
                "GIF": "image/gif", "WEBP": "image/webp"}.get(fmt, "image/jpeg")
        return base64.b64encode(buf.read()).decode("utf-8"), mime

    except ImportError:
        ext = os.path.splitext(image_path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp"}.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime


def extract_json(text):
    """
    FIX #2: Robust JSON extraction that handles:
      - Markdown fences (```json ... ```)
      - Truncated responses (max_tokens cut off mid-array)

    Strategy:
      1. Strip markdown fences.
      2. Try full json.loads on the largest {...} block.
      3. If that fails (truncated), salvage complete detection objects
         from the partial text using regex on individual items.
    """
    if not text:
        return {}

    # Strip markdown code fences
    txt = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()

    # --- Attempt 1: parse the full JSON blob ---
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass  # fall through to partial salvage

    # --- Attempt 2: partial salvage for truncated detections arrays ---
    # Extract every complete {...} object inside a "detections" list.
    # Each item looks like:
    #   {"class_name": "POSTER", "class_id": 1019, "result": "Y", "confidence": 90}
    item_pattern = re.compile(
        r'\{\s*"class_name"\s*:\s*"[^"]+"\s*,\s*"class_id"\s*:\s*(\d+)\s*,'
        r'\s*"result"\s*:\s*"([YN])"\s*,\s*"confidence"\s*:\s*(\d+)\s*\}',
        re.DOTALL,
    )
    items = item_pattern.findall(txt)
    if items:
        detections = []
        for class_id, result, confidence in items:
            detections.append({
                "class_id":   int(class_id),
                "result":     result,
                "confidence": int(confidence),
            })
        logger.debug(
            f"  [extract_json] Partial salvage: recovered {len(detections)} "
            f"detection(s) from truncated response"
        )
        return {"detections": detections}

    return {}


def normalize_llm_response(raw):
    """
    Parse the LLM response into a flat dict:
        { str_class_id: {"result": "Y"/"N", "confidence": int} }

    Primary format — detections array (current prompt):
        {
          "detections": [
            {"class_name": "POSTER", "class_id": 1019, "result": "Y", "confidence": 92},
            {"class_name": "STREAMER", "class_id": 1020, "result": "N", "confidence": 0},
            ...
          ]
        }

    Legacy fallback — flat dict:
        {"1019": "Y", "1020": "N"}
    """
    if not raw:
        return {}

    # Primary: detections array
    if "detections" in raw and isinstance(raw["detections"], list):
        normalized = {}
        for item in raw["detections"]:
            try:
                cid = item.get("class_id")
                result = str(item.get("result", "N")).upper().strip()
                confidence = int(item.get("confidence", 0))
                if cid is None:
                    continue
                if result not in ("Y", "N"):
                    result = "N"
                normalized[str(int(cid))] = {
                    "result":     result,
                    "confidence": confidence if result == "Y" else 0,
                }
            except (ValueError, TypeError):
                continue
        return normalized

    # Legacy fallback: flat {class_id: result} dict
    normalized = {}
    for k, v in raw.items():
        try:
            result = str(v).upper().strip()
            if result not in ("Y", "N"):
                result = "N"
            normalized[str(int(k))] = {"result": result, "confidence": 0}
        except (ValueError, TypeError):
            pass
    return normalized


def call_llm(api_key, model, prompt, image_path,
             max_tokens=2048, temperature=0,
             image_detail="auto", max_image_px=1024,
             max_retries=3, base_delay=5):
    """
    Send one vision request to OpenAI and return a parsed dict.
    Retries on 429, 5xx, and timeouts with exponential back-off.

    FIX #2: default max_tokens raised to 2048 (was 512) so the full
    detections array is never truncated mid-response.
    """
    b64, mime = encode_image(image_path, max_px=max_image_px)
    data_url = f"data:{mime};base64,{b64}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise image analysis system. "
                    "Return ONLY a valid JSON object -- "
                    "no markdown, no explanation, no extra keys."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": image_detail},
                    },
                ],
            },
        ],
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(OPENAI_CHAT_URL, headers=headers,
                                 json=payload, timeout=120)

            if resp.status_code == 200:
                raw_text = resp.json()["choices"][0]["message"]["content"]
                parsed = extract_json(raw_text)
                if parsed:
                    logger.debug(f"  [LLM] OK attempt {attempt} -- {len(parsed)} keys")
                    return parsed
                logger.warning(f"  [LLM] Non-JSON response: {raw_text[:200]}")
                return {}

            elif resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", base_delay * attempt))
                logger.warning(f"  [LLM] 429 rate-limited -- waiting {wait:.0f}s")
                time.sleep(wait)

            elif resp.status_code in (500, 502, 503, 504):
                wait = base_delay * attempt
                logger.warning(f"  [LLM] HTTP {resp.status_code} -- retrying in {wait}s")
                time.sleep(wait)

            else:
                logger.error(f"  [LLM] Unrecoverable HTTP {resp.status_code}: {resp.text[:300]}")
                return {}

        except requests.exceptions.Timeout:
            wait = base_delay * attempt
            logger.warning(f"  [LLM] Timeout attempt {attempt} -- retrying in {wait}s")
            time.sleep(wait)
        except Exception as e:
            logger.error(f"  [LLM] Unexpected error: {e}")
            return {}

    logger.error(f"  [LLM] All {max_retries} attempts exhausted for {os.path.basename(image_path)}")
    return {}


# ===========================================================================
# Per-image processing
# ===========================================================================
def process_image(
    image_info,
    prompt,
    api_key,
    model,
    max_tokens,
    temperature,
    image_detail,
    max_image_px,
    s3_annotated_folder,
    s3_client,
    s3_config,
):
    """
    Run LLM on one image and return a list of result-row dicts.

    image_info : (filesequenceid, storename, filename, local_path,
                  s3_key, storeid, category_id)
    """
    fileseqid, storename, filename, local_path, s3_key, storeid, category_id = image_info

    if not os.path.exists(local_path):
        logger.warning(f"  Local file missing, skipping: {local_path}")
        return []

    logger.info(f"  Processing [category={category_id}]: {filename}")

    results = []
    s3_annot_key = f"{s3_annotated_folder}/{filename}"

    try:
        # -- LLM call ---------------------------------------------------------
        raw = call_llm(
            api_key, model, prompt, local_path,
            max_tokens=max_tokens,
            temperature=temperature,
            image_detail=image_detail,
            max_image_px=max_image_px,
        )
        llm_output = normalize_llm_response(raw)
        y_count = sum(1 for v in llm_output.values() if v["result"] == "Y")
        logger.info(
            f"  LLM returned {len(llm_output)} class keys for {filename} "
            f"({y_count} detected as Y)"
        )

        # -- Build result rows ------------------------------------------------
        now = datetime.now()
        rowid = 1   # placeholder -- globally-unique rowids assigned after dedup
        seen = set()

        for cid_str, val in llm_output.items():
            if cid_str not in LLM_CLASS_IDS:
                continue
            int_cid = int(cid_str)
            if int_cid in seen:
                continue
            seen.add(int_cid)
            result     = val["result"]        # "Y" or "N"
            confidence = val["confidence"]    # 40-100 for Y, 0 for N
            results.append({
                "rowid":                 rowid,
                "modelname":             MODELNAME,
                "imagefilename":         filename,
                "classid":               int_cid,
                "classtext":             LLM_CLASS_IDS[cid_str],
                "value":                 result,
                "inference":             confidence / 100.0,
                "modelrun":              now,
                "processed_flag":        "N",
                "storeid":               storeid,
                "storename":             storename,
                "s3path_actual_file":    s3_key,
                "s3path_annotated_file": s3_annot_key,
            })
            rowid += 1

        # FIX #1: use ASCII arrow '->' to avoid UnicodeEncodeError on Windows
        try:
            upload_to_s3(s3_client, s3_config, local_path, s3_annot_key)
            logger.info(f"  Uploaded annotated image -> {s3_annot_key}")
        except Exception as up_err:
            logger.warning(f"  S3 upload failed for {filename}: {up_err}")

    except Exception as e:
        logger.error(f"  Error processing {filename}: {e}", exc_info=True)

    return results


# ===========================================================================
# CSV writer
# ===========================================================================
def write_csv(output_csv, results):
    if not results:
        logger.warning("No results -- CSV not written.")
        return None
    csv_dir = os.path.dirname(output_csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"CSV written -> {output_csv}  ({len(results)} rows)")
    return output_csv


# ===========================================================================
# Main pipeline
# ===========================================================================
def run_llm_rescorer(config_path="config.json"):
    """
    Full end-to-end LLM re-scoring pipeline.

    Steps
    -----
    1.  Load config
    2.  Find low-activation stores from orgi.scorecard (current month)
    3.  Fetch image rows for those stores (category_id 1 & 2)
    4.  Get stagingid
    5.  Download images from S3 to temp dir
    6.  Run LLM on each image
    7.  Log statistics
    8.  Reopen DB -- fetch max_rowid + already-inserted pairs (dedup guard)
    9.  Assign globally-unique rowids
    10. Write CSV
    11. Insert into orgi.visibilityitemsstaging (chunked, auto-reconnect)
    12. Upload CSV to S3
    """
    os.makedirs("outputs", exist_ok=True)

    # -- 1. Load config -------------------------------------------------------
    config = load_config(config_path)
    db_config      = config["db_config"]
    s3_config      = config["s3_config"]
    ollama_cfg     = config.get("ollama_config", {})
    chatgpt_cfg    = config.get("chatgpt_config", {})

    api_key = chatgpt_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error(
            "OpenAI API key not found. Add 'api_key' in chatgpt_config "
            "or set the OPENAI_API_KEY environment variable."
        )
        return

    model        = chatgpt_cfg.get("model",        "gpt-4o")
    # FIX #2: read max_tokens from config (should be 2048+); fall back to 2048
    max_tokens   = int(chatgpt_cfg.get("max_tokens",   2048))
    temperature  = float(chatgpt_cfg.get("temperature", 0))
    image_detail = chatgpt_cfg.get("image_detail", "auto")
    max_image_px = int(chatgpt_cfg.get("max_image_px", 1024))
    output_csv   = chatgpt_cfg.get("output_csv",   "outputs/llm_rescorer_results.csv")
    output_s3_folder = ollama_cfg.get("output_s3_folder", "ModelResults/")

    # Prompt file -- check chatgpt_config first, fall back to ollama_config prompt_path
    prompt_file = chatgpt_cfg.get("prompt_file")
    if not prompt_file:
        prompt_path = ollama_cfg.get("prompt_path", "data/prompts/")
        prompt_file = os.path.join(prompt_path, "prompt.txt")

    prompt = load_prompt(prompt_file)

    logger.info("=" * 60)
    logger.info("LLM ACTIVATION RE-SCORER STARTING")
    logger.info(f"  Model        : {model}")
    logger.info(f"  Max tokens   : {max_tokens}")
    logger.info(f"  Threshold    : activation_score < {ACTIVATION_SCORE_THRESHOLD}")
    logger.info(f"  Categories   : {LLM_CATEGORY_IDS}")
    logger.info(f"  Prompt file  : {prompt_file}")
    logger.info("=" * 60)

    # -- 2. Find low-activation stores ----------------------------------------
    store_ids = get_low_activation_stores(db_config, threshold=ACTIVATION_SCORE_THRESHOLD)
    if not store_ids:
        logger.info("No low-activation stores found. Nothing to do.")
        return

    # -- 3. Fetch image rows ---------------------------------------------------
    image_rows = get_images_for_stores(db_config, store_ids)
    if not image_rows:
        logger.info("No images found for low-activation stores. Nothing to do.")
        return

    # -- 4. Get stagingid ------------------------------------------------------
    stagingid = get_stagingid(db_config)

    # -- 5. Download images from S3 --------------------------------------------
    s3_client = build_s3_client(s3_config)
    s3_annotated_folder = f"{output_s3_folder}LLMRescore_{stagingid}"

    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Temp dir: {temp_dir}")
        image_paths, failed_downloads = download_images(
            s3_client, s3_config, temp_dir, image_rows
        )

        if not image_paths:
            logger.error("No images downloaded. Aborting.")
            return

        if failed_downloads:
            logger.warning(f"{len(failed_downloads)} images failed to download")

        total = len(image_paths)

        # -- 6. LLM inference --------------------------------------------------
        all_results = []
        for idx, img_info in enumerate(image_paths, start=1):
            logger.info(f"[{idx}/{total}] store={img_info[5]} cat={img_info[6]}")
            rows = process_image(
                img_info,
                prompt,
                api_key, model, max_tokens, temperature, image_detail, max_image_px,
                s3_annotated_folder,
                s3_client,
                s3_config,
            )
            all_results.extend(rows)

        # -- 7. Statistics -----------------------------------------------------
        logger.info("=" * 60)
        logger.info("LLM RESCORER STATISTICS:")
        logger.info(f"  Stores queried      : {len(store_ids)}")
        logger.info(f"  Images downloaded   : {len(image_paths)}")
        logger.info(f"  Images failed DL    : {len(failed_downloads)}")
        logger.info(f"  Result rows raw     : {len(all_results)}")
        logger.info(f"  stagingid           : {stagingid}")
        logger.info("=" * 60)

        # -- 8. Reopen DB -- max rowid + dedup guard ---------------------------
        conn = db_connect(db_config)
        cur = conn.cursor()

        try:
            cur.execute("SELECT 1")
        except Exception as e:
            logger.warning(f"DB ping failed, reconnecting: {e}")
            db_close(conn, cur)
            conn = db_connect(db_config)
            cur = conn.cursor()

        max_existing_rowid = get_max_rowid(conn, cur, stagingid)
        logger.info(
            f"Max existing rowid for stagingid {stagingid}: {max_existing_rowid} "
            f"-- new rows start from {max_existing_rowid + 1}"
        )

        already_inserted = get_already_inserted(conn, cur, stagingid)
        if already_inserted:
            logger.info(
                f"Cross-batch dedup: {len(already_inserted)} pairs already in DB"
            )

        deduped = []
        skipped = 0
        for r in all_results:
            key = (r["imagefilename"], r["classid"])
            if key in already_inserted:
                skipped += 1
            else:
                deduped.append(r)
                already_inserted.add(key)

        if skipped:
            logger.warning(f"Skipped {skipped} duplicate (imagefilename, classid) rows")
        all_results = deduped

        # -- 9. Assign globally-unique rowids ----------------------------------
        for i, result in enumerate(all_results, start=max_existing_rowid + 1):
            result["rowid"] = i

        # -- 10. Write CSV -----------------------------------------------------
        final_csv = write_csv(output_csv, all_results)

        # -- 11. Insert into DB ------------------------------------------------
        if all_results:
            insert_results_chunked(conn, cur, stagingid, all_results, db_config)
        else:
            logger.warning("No results to insert into DB.")

        try:
            conn.commit()
        except Exception:
            pass

        db_close(conn, cur)
        logger.info("DB insert completed.")

        # -- 12. Upload CSV to S3 ----------------------------------------------
        if final_csv and os.path.exists(final_csv):
            csv_s3_key = f"{output_s3_folder}LLMRescore_{stagingid}/llm_rescorer_results.csv"
            try:
                upload_to_s3(s3_client, s3_config, final_csv, csv_s3_key)
                logger.info(f"CSV uploaded to S3 -> {csv_s3_key}")
            except Exception as e:
                logger.error(f"CSV S3 upload failed: {e}")

    logger.info("=" * 60)
    logger.info(f"LLM RE-SCORER COMPLETE  |  stagingid={stagingid}")
    logger.info(f"  Final result rows inserted : {len(all_results)}")
    logger.info("=" * 60)


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    run_llm_rescorer(config_path="config.json")