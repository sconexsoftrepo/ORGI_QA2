"""
visicooler_presence_detector.py — Visicooler-availability YOLO detector.

Runs the dedicated visicooler-detection model (visicooler_config.model_path
/ visicooler_model_path) over the subcategory-601 images — the ones that
`chatgpt_analyzer.py` deliberately skips via `_VISICOOLER_SUBCATS` — and
writes ONE row per image into orgi.visibilityitemsstaging:

    classid = 1001  ("Visicooler - Visicooler available")
    value   = 'Y' if a visicooler is found, 'N' otherwise

Detection order per image:
  1. Primary visicooler model, at `conf_threshold` (default 0.1).
  2. If (1) found nothing: fall back to a generic `yolov8x.pt` model,
     restricted to ONLY its "refrigerator" class (every other COCO class
     it can detect is ignored), at `fallback_conf_threshold`.
  3. If neither found anything: value = 'N'.

It follows the exact same DB-insert pattern as
`app.chatgpt_analyzer.run_yolo_analysis`:
  - builds result dicts with the same keys `insert_ollama_results` expects
  - assigns globally-unique rowids by reading MAX(rowid) for this stagingid
  - de-dupes against rows already in orgi.visibilityitemsstaging for this
    stagingid (same image+classid, N->Y upgrade allowed, Y never overwritten)
  - inserts via `app.db_handler.insert_ollama_results` (chunked, retry-safe)
  - optionally uploads an annotated copy to S3, same folder convention as
    the activation YOLO step

Usage (mirrors run_yolo_analysis's call shape):

    from visicooler_presence_detector import run_visicooler_presence_analysis

    results, csv_path = run_visicooler_presence_analysis(
        image_paths=image_paths,           # same list returned by
                                            # s3_handler.download_images_from_s3
        db_config=db_config,
        stagingid=stagingid,
        visicooler_model_path=visicooler_config['model_path'],  # or
                                            # visicooler_config['visicooler_model_path']
        conf_threshold=0.1,
        fallback_model_path="yolov8x.pt",  # generic COCO model, "refrigerator" class only
        fallback_conf_threshold=0.1,
        s3_handler=s3_handler,
        s3_annotated_folder=f"{ollama_cfg['output_s3_folder']}Visicooler_{cyclecountid}",
        output_csv="outputs/visicooler_presence_results.csv",
    )
"""

import copy
import csv
import logging
import os
import tempfile
from datetime import datetime

from app.db_handler import (
    initialize_db_connection,
    close_db_connection,
    get_classtext,
    insert_ollama_results,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Subcategory that chatgpt_analyzer.py's _VISICOOLER_SUBCATS skips and that
# this module is responsible for instead. Kept in sync with that set's
# convention (601 = visicooler presence photos).
DEFAULT_VISICOOLER_SUBCATEGORY_ID = 601

VISICOOLER_CLASSID = 1001  # "Visicooler - Visicooler available"

# COCO class name the fallback yolov8x.pt model must match — every other
# class it can detect (person, chair, bottle, etc.) is ignored.
REFRIGERATOR_CLASS_NAME = "refrigerator"

DEFAULT_FALLBACK_MODEL_PATH = "yolov8x.pt"

# Singleton, lazy-loaded once per process — same pattern as
# chatgpt_analyzer.get_activation_yolo
_VISICOOLER_PRESENCE_YOLO = None
_REFRIGERATOR_FALLBACK_YOLO = None


def get_visicooler_presence_yolo(model_path):
    global _VISICOOLER_PRESENCE_YOLO
    if _VISICOOLER_PRESENCE_YOLO is None:
        from ultralytics import YOLO
        logger.info(f"Loading visicooler-presence YOLO: {model_path}")
        _VISICOOLER_PRESENCE_YOLO = YOLO(model_path)
    return _VISICOOLER_PRESENCE_YOLO


def get_refrigerator_fallback_yolo(model_path):
    global _REFRIGERATOR_FALLBACK_YOLO
    if _REFRIGERATOR_FALLBACK_YOLO is None:
        from ultralytics import YOLO
        logger.info(f"Loading refrigerator-fallback YOLO: {model_path}")
        _REFRIGERATOR_FALLBACK_YOLO = YOLO(model_path)
    return _REFRIGERATOR_FALLBACK_YOLO


def run_visicooler_presence_yolo(image_path, model_path, conf_threshold=0.1):
    """
    Run the visicooler-presence YOLO model on one image.

    Returns
    -------
    detected   : bool       — True if at least one box was found at/above conf_threshold
    max_conf   : float      — highest confidence among detected boxes (0.0 if none)
    annot_bgr  : ndarray|None — annotated frame with detections drawn (None if none/failed)
    """
    model = get_visicooler_presence_yolo(model_path)
    results = model(image_path, conf=conf_threshold, verbose=False)

    detected = False
    max_conf = 0.0
    annot_bgr = None

    for r in results:
        confs = [float(c) for c in r.boxes.conf] if r.boxes is not None else []
        if confs:
            detected = True
            max_conf = max(max_conf, max(confs))
            logger.info(
                f"  [YOLO] visicooler detected in {os.path.basename(image_path)} "
                f"(max_conf={max_conf:.3f}, boxes={len(confs)})"
            )
            try:
                annot_bgr = r.plot()
            except Exception as plot_err:
                logger.debug(f"  YOLO .plot() failed: {plot_err}")
                annot_bgr = None

    return detected, max_conf, annot_bgr


def run_refrigerator_fallback_yolo(image_path, model_path, conf_threshold=0.1):
    """
    Fallback detector: run a generic COCO-pretrained YOLO (default
    yolov8x.pt) on one image, but only ever look at its "refrigerator"
    class — every other class it can detect (person, bottle, chair, TV,
    etc.) is ignored entirely, both for the Y/N decision and for what gets
    drawn on the annotated image.

    Returns
    -------
    detected   : bool
    max_conf   : float
    annot_bgr  : ndarray|None — annotated frame with ONLY refrigerator boxes drawn
    """
    model = get_refrigerator_fallback_yolo(model_path)
    results = model(image_path, conf=conf_threshold, verbose=False)

    detected = False
    max_conf = 0.0
    annot_bgr = None

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        fridge_idx = [
            i for i, c in enumerate(r.boxes.cls)
            if model.names[int(c)].lower().replace(" ", "_") == REFRIGERATOR_CLASS_NAME
        ]
        if not fridge_idx:
            continue

        confs = [float(r.boxes.conf[i]) for i in fridge_idx]
        detected = True
        max_conf = max(max_conf, max(confs))
        logger.info(
            f"  [fallback YOLO] refrigerator detected in {os.path.basename(image_path)} "
            f"(max_conf={max_conf:.3f}, boxes={len(fridge_idx)})"
        )
        try:
            import torch
            r_fridge = copy.deepcopy(r)
            r_fridge.boxes = r_fridge.boxes[torch.tensor(fridge_idx, dtype=torch.long)]
            annot_bgr = r_fridge.plot()
        except Exception as plot_err:
            logger.debug(f"  fallback YOLO .plot() failed: {plot_err}")
            annot_bgr = None

    return detected, max_conf, annot_bgr


def process_single_visicooler_presence_image(
    image_info,
    visicooler_model_path,
    conf_threshold,
    classtext,
    s3_handler=None,
    s3_annotated_folder=None,
    modelname="yolo_visicooler_presence",
    fallback_model_path=DEFAULT_FALLBACK_MODEL_PATH,
    fallback_conf_threshold=None,
):
    """
    Run the visicooler-presence model on one image and build its single
    result row (classid=1001, value='Y'/'N').

    If the primary model finds nothing, falls back to a generic
    `yolov8x.pt` model looking ONLY for its "refrigerator" class before
    settling on 'N'. Pass fallback_model_path=None to disable the fallback.

    image_info: (fileseqid, storename, filename, local_path, s3_key, storeid, subcategory_id)
                — same tuple shape as s3_handler.download_images_from_s3 returns.

    Returns: dict | None  (None if the local file is missing)
    """
    fileseqid, storename, filename, local_path, s3_key, storeid, subcategory_id = image_info

    if not os.path.exists(local_path):
        logger.warning(f"  Local file missing, skipping: {local_path}")
        return None

    if fallback_conf_threshold is None:
        fallback_conf_threshold = conf_threshold

    try:
        logger.info(f"Processing (visicooler presence): {filename}")
        detected, max_conf, annot_bgr = run_visicooler_presence_yolo(
            local_path, visicooler_model_path, conf_threshold
        )
        used_model = modelname

        if not detected and fallback_model_path:
            logger.info(
                f"  Primary visicooler model found nothing on {filename} — "
                f"trying refrigerator fallback ({fallback_model_path})"
            )
            detected, max_conf, annot_bgr = run_refrigerator_fallback_yolo(
                local_path, fallback_model_path, fallback_conf_threshold
            )
            if detected:
                used_model = f"{modelname}_fallback_refrigerator"

        if not detected:
            logger.info(f"  No visicooler and no refrigerator found on {filename} — value=N")

        value = "Y" if detected else "N"
        inference = max_conf if detected else 0.0

        s3_annot = None
        if s3_handler is not None and s3_annotated_folder:
            s3_annot = f"{s3_annotated_folder}/{filename}"
            try:
                if annot_bgr is not None:
                    import cv2 as _cv2
                    ext = os.path.splitext(filename)[1] or ".jpg"
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        if _cv2.imwrite(tmp_path, annot_bgr):
                            s3_handler.upload_file_to_s3(tmp_path, s3_annot)
                            logger.info(f"  Uploaded visicooler-annotated image → {s3_annot}")
                        else:
                            s3_handler.upload_file_to_s3(local_path, s3_annot)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                else:
                    s3_handler.upload_file_to_s3(local_path, s3_annot)
            except Exception as upload_err:
                logger.warning(f"  S3 upload failed for {filename}: {upload_err}")
                s3_annot = None

        return {
            "rowid": 1,  # placeholder — globally unique rowids assigned by caller
            "modelname": used_model,
            "imagefilename": filename,
            "classid": VISICOOLER_CLASSID,
            "classtext": classtext,
            "value": value,
            "inference": inference,
            "modelrun": datetime.now(),
            "processed_flag": "N",
            "storeid": storeid,
            "storename": storename,
            "s3path_actual_file": s3_key,
            "s3path_annotated_file": s3_annot,
        }

    except Exception as e:
        logger.error(f"  Error processing {filename}: {e}", exc_info=True)
        return None


def run_visicooler_presence_analysis(
    image_paths,
    db_config,
    stagingid,
    visicooler_model_path,
    conf_threshold=0.1,
    s3_handler=None,
    s3_annotated_folder=None,
    output_csv=None,
    subcategory_id=DEFAULT_VISICOOLER_SUBCATEGORY_ID,
    fallback_model_path=DEFAULT_FALLBACK_MODEL_PATH,
    fallback_conf_threshold=None,
):
    """
    End-to-end: filter image_paths down to `subcategory_id` (default 601),
    run the visicooler-presence model on each, and insert results into
    orgi.visibilityitemsstaging under `stagingid` — same table/pattern as
    app.chatgpt_analyzer.run_yolo_analysis.

    If the primary model finds nothing on an image, it falls back to a
    generic `yolov8x.pt` model looking ONLY for its "refrigerator" class
    before the row is finally written as 'N'. Pass fallback_model_path=None
    to disable this fallback and go straight to 'N'.

    Returns (results: list[dict], output_csv: str|None)
    """
    if not visicooler_model_path:
        logger.error(
            "No visicooler model path configured "
            "(visicooler_config.model_path / visicooler_model_path in config.json)."
        )
        return [], None

    target_images = [img for img in image_paths if img[6] == subcategory_id]
    logger.info(
        f"Visicooler-presence: {len(target_images)} of {len(image_paths)} images "
        f"are subcategory {subcategory_id}"
    )
    if not target_images:
        return [], None

    # Pre-fetch classtext once (short-lived connection, same convention as
    # chatgpt_analyzer's classtext pre-fetch step).
    conn, cur = initialize_db_connection(db_config)
    classtext = get_classtext(cur, VISICOOLER_CLASSID)
    close_db_connection(conn, cur)

    all_results = []
    for image_info in target_images:
        row = process_single_visicooler_presence_image(
            image_info,
            visicooler_model_path,
            conf_threshold,
            classtext,
            s3_handler=s3_handler,
            s3_annotated_folder=s3_annotated_folder,
            fallback_model_path=fallback_model_path,
            fallback_conf_threshold=fallback_conf_threshold,
        )
        if row is not None:
            all_results.append(row)

    detected_count = sum(1 for r in all_results if r["value"] == "Y")
    fallback_hits = sum(1 for r in all_results if r["modelname"].endswith("_fallback_refrigerator"))
    logger.info(
        f"Visicooler-presence detection complete: {detected_count}/{len(all_results)} "
        f"images show a visicooler (conf>={conf_threshold}), of which {fallback_hits} "
        f"were found only via the refrigerator fallback"
    )

    if not all_results:
        return [], None

    # ── Reopen DB: assign globally-unique rowids + cross-batch dedup ──────
    # Same dedup semantics as chatgpt_analyzer.run_yolo_analysis: an
    # existing 'Y' for (imagefilename, classid) is never overwritten; an
    # existing 'N' can be upgraded to 'Y'.
    conn, cur = initialize_db_connection(db_config)
    try:
        cur.execute(
            "SELECT COALESCE(MAX(rowid), 0) FROM orgi.visibilityitemsstaging "
            "WHERE stagingid = %s",
            (stagingid,),
        )
        max_existing_rowid = int(cur.fetchone()[0])

        cur.execute(
            "SELECT imagefilename, classid, value FROM orgi.visibilityitemsstaging "
            "WHERE stagingid = %s AND classid = %s",
            (stagingid, VISICOOLER_CLASSID),
        )
        already_inserted = {(row[0], int(row[1])): row[2] for row in cur.fetchall()}

        deduped_results = []
        skipped_dupes = 0
        for r in all_results:
            key = (r["imagefilename"], r["classid"])
            existing_val = already_inserted.get(key)

            if existing_val is None:
                deduped_results.append(r)
                already_inserted[key] = r["value"]
            elif existing_val == "Y":
                skipped_dupes += 1
            elif existing_val == "N" and r["value"] == "Y":
                deduped_results.append(r)
                already_inserted[key] = "Y"
            else:
                skipped_dupes += 1

        if skipped_dupes:
            logger.info(f"Visicooler-presence: skipped {skipped_dupes} duplicate rows")

        all_results = deduped_results
        for global_idx, result in enumerate(all_results, start=max_existing_rowid + 1):
            result["rowid"] = global_idx

        if output_csv and all_results:
            csv_dir = os.path.dirname(output_csv)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
            logger.info(f"CSV written → {output_csv} ({len(all_results)} rows)")

        if all_results:
            insert_ollama_results(
                cur, stagingid, all_results,
                s3_annotated_folder, image_paths,
                conn=conn, db_config=db_config,
            )
            conn.commit()
            logger.info(
                f"Inserted {len(all_results)} visicooler-presence rows "
                f"into orgi.visibilityitemsstaging (stagingid={stagingid})"
            )

        return all_results, output_csv if all_results else None

    except Exception as e:
        logger.error(f"Visicooler-presence DB insert failed: {e}", exc_info=True)
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        close_db_connection(conn, cur)