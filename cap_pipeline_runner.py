from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import psycopg2

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    step: int
    name: str
    status: str                   # "success" | "failed" | "skipped"
    rows_affected: Optional[int]
    duration_ms: float
    error: Optional[str] = None

    def __str__(self) -> str:
        rows = f"{self.rows_affected} rows" if self.rows_affected is not None else "n/a"
        return (
            f"  Step {self.step:>2}  [{self.status.upper():<7}]  "
            f"{self.name:<55}  {rows:<12}  {self.duration_ms:>8.1f} ms"
        )


@dataclass
class PipelineResult:
    iteration_id: int
    overall_status: str           # "success" | "failed"
    total_duration_ms: float
    steps: list[StepResult] = field(default_factory=list)
    pipeline_label: str = "CAP PREDICTION PIPELINE"
    total_steps: int = 11

    # ── Pretty summary ────────────────────────────────────────────────────────
    def log_summary(self) -> None:
        sep = "=" * 100
        logger.info(sep)
        logger.info(f"  {self.pipeline_label}  –  SUMMARY ({self.total_steps} steps)")
        logger.info(sep)
        logger.info(f"  Iteration ID   : {self.iteration_id}")
        logger.info(f"  Overall status : {self.overall_status.upper()}")
        logger.info(f"  Total duration : {self.total_duration_ms:.1f} ms")
        logger.info("-" * 100)
        logger.info(
            f"  {'Step':>4}  {'Status':<9}  {'Name':<55}  {'Rows':<12}  {'Duration':>10}"
        )
        logger.info("-" * 100)
        for s in self.steps:
            logger.info(str(s))
        logger.info(sep)

        # Surface any errors
        failed = [s for s in self.steps if s.status == "failed"]
        if failed:
            logger.error(f"  {len(failed)} step(s) failed:")
            for s in failed:
                logger.error(f"    Step {s.step} – {s.name}")
                if s.error:
                    first_line = s.error.strip().splitlines()[-1]
                    logger.error(f"      {first_line}")
            logger.info(sep)


# ── SQL definitions ────────────────────────────────────────────────────────────
def _build_queries(iteration_id: int, staging_id: int) -> list[dict]:
    """Return all pipeline SQL statements in execution order.

    Steps removed vs. the original 8-step version
    -----------------------------------------------
    Old Step 2 – "Insert CAP predictions from SKU table"
        cap_sku_mapper (Python, called in visicooler.py) already inserts caps
        into temp.cap_prediction_temp with prod_class_id pre-filled via
        4-phase brand+package matching.  Copying SKU rows a second time here
        created ghost rows.

    Old Steps 3 & 4 – SQL vertical-match / centroid-distance remapping
        The Python mapper is brand+package-aware and IoU-deduped; the SQL
        fallback is no longer needed.

    Old Step 5 – "Remove small caps inside SKU bounding box (< 15 % area)"
        With Steps 2-4 gone, cap rows are *only* cap-model detections.
        The Python IoU-dedup in cap_sku_mapper already handles overlapping
        duplicate caps before insertion.

    Remaining pipeline (renumbered 1-11):
        1  Create unique index          (guard against any race-condition dups)
        2  Remove duplicate rows        (clean-up safety net)
        3  Backfill missing cap rows for uploaded images with no cap prediction
        4  Populate coolermetricsmaster
        5  Populate coolermetricstransaction
        6  Insert "other"-brand SKU rows      (productclassid = 59)
        7  Insert "alcohol"-brand SKU rows    (productclassid = 25)
        8  Update caserid on coolermetricsmaster
        9  Insert structural-count extra rows  ← BUG FIXED HERE
        10 Insert into reference_table            (marks iteration as AI-processed)
        11 Validate reference_table against visibilityitemsstaging

    Steps 10 & 11 only run if Steps 1-9 all succeeded, since the runner
    aborts and rolls back on the first failed step (see run_cap_pipeline).

    NOTE: the low-detection-count IMPURE->PURE reclassification is NOT a
    step in this pipeline — see reclassify_low_detection_impure() below.
    It depends on orgi.shelfsequencecompliance.purity_status and
    orgi.dino_nonbeverage_detections, both of which are only populated by
    run_shelf_sequence_check() — which main.py runs AFTER run_cap_pipeline().
    Running it as a step here means it always executes before that data
    exists for the current iteration, so it would silently match 0 rows
    every time. Call reclassify_low_detection_impure() from main.py
    after run_shelf_sequence_check() completes instead.
    """
    iid = int(iteration_id)          # guard against injection
    sid = int(staging_id)            # guard against injection
    return [
        # ── 1 ─────────────────────────────────────────────────────────────────
        {
            "step": 1,
            "name": "Create unique index on cap_prediction_temp",
            "sql": """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_cap_box
                ON temp.cap_prediction_temp (
                    store_id, image_file_name, s3path_annotated_file,
                    iteration_id, cap_class_id, x1, y1, x2, y2
                );
            """,
        },
        # ── 2 ─────────────────────────────────────────────────────────────────
        {
            "step": 2,
            "name": "Remove duplicate rows",
            "sql": """
                DELETE FROM temp.cap_prediction_temp c
                USING (
                    SELECT
                        store_id, image_file_name, s3path_annotated_file,
                        iteration_id, cap_class_id, x1, y1,
                        MIN(ctid) AS keep_ctid
                    FROM temp.cap_prediction_temp
                    GROUP BY
                        store_id, image_file_name, s3path_annotated_file,
                        iteration_id, cap_class_id, x1, y1
                    HAVING COUNT(*) > 1
                ) d
                WHERE c.store_id              = d.store_id
                  AND c.image_file_name       = d.image_file_name
                  AND c.s3path_annotated_file = d.s3path_annotated_file
                  AND c.iteration_id          = d.iteration_id
                  AND c.cap_class_id          = d.cap_class_id
                  AND c.x1                   = d.x1
                  AND c.y1                   = d.y1
                  AND c.ctid                 <> d.keep_ctid;
            """,
        },
        # ── 3 ─────────────────────────────────────────────────────────────────
        # Backfill a synthetic "no-cap" row for any uploaded image in this
        # store/subcategory/date-window that has no row yet in
        # temp.cap_prediction_temp for this iteration, so downstream steps
        # (coolermetricsmaster/coolermetricstransaction population) still
        # produce a row for that image.
        {
            "step": 3,
            "name": "Backfill missing cap rows for images with no cap prediction",
            "sql": f"""
                INSERT INTO temp.cap_prediction_temp (
                    store_id,
                    image_file_name,
                    iteration_id,
                    cap_class_id,
                    x1,
                    x2,
                    y1,
                    y2,
                    prod_class_id,
                    shelfnumber,
                    brand_name,
                    s3path_annotated_file
                )
                SELECT
                    f.storeid,
                    f.filename,
                    {iid} AS iteration_id,
                    10028 AS cap_class_id,
                    -1 AS x1,
                    -1 AS x2,
                    -1 AS y1,
                    -1 AS y2,
                    10028 AS prod_class_id,
                    f.shelfnumber,
                    NULL AS brand_name,
                    NULL AS s3path_annotated_file
                FROM orgi.fileupload f
                -- Only consider stores that already belong to this iteration
                INNER JOIN (
                    SELECT DISTINCT store_id
                    FROM temp.cap_prediction_temp
                    WHERE iteration_id = {iid}
                ) iter_stores
                    ON iter_stores.store_id = f.storeid
                -- Check whether this image already exists for this iteration
                LEFT JOIN temp.cap_prediction_temp cp
                    ON cp.store_id = f.storeid
                   AND cp.image_file_name = f.filename
                   AND cp.iteration_id = {iid}
                WHERE
                    f.subcategory_id = 605
                    AND f.uploadtimestamp >= '2026-09-01'
                    AND f.uploadtimestamp < '2026-10-01'
                    AND cp.store_id IS NULL
                ON CONFLICT DO NOTHING;
            """,
        },
        # ── 4 ─────────────────────────────────────────────────────────────────
        {
            "step": 4,
            "name": "Populate orgi.coolermetricsmaster",
            "sql": f"""
                WITH image_map AS (
                    SELECT DISTINCT
                        cpt.iteration_id  AS iterationid,
                        cpt.store_id      AS storeid,
                        cpt.image_file_name,
                        cpt.s3path_annotated_file,
                        DENSE_RANK() OVER (
                            PARTITION BY cpt.iteration_id
                            ORDER BY
                                cpt.store_id,
                                cpt.image_file_name,
                                cpt.s3path_annotated_file
                        ) AS iterationtranid
                    FROM temp.cap_prediction_temp cpt
                    WHERE cpt.iteration_id = {iid}
                )
                INSERT INTO orgi.coolermetricsmaster (
                    iterationid, iterationtranid, storeid,
                    caserid, modelrun, processed_flag
                )
                SELECT
                    i.iterationid,
                    i.iterationtranid,
                    i.storeid,
                    pm.caserid,
                    NOW(),
                    'N'
                FROM image_map i
                JOIN orgi.puritymapping pm ON pm.caserid IS NOT NULL
                ON CONFLICT (iterationid, iterationtranid) DO NOTHING;
            """,
        },
        # ── 5 ─────────────────────────────────────────────────────────────────
        {
            "step": 5,
            "name": "Populate orgi.coolermetricstransaction",
            "sql": f"""
                WITH image_map AS (
                    SELECT DISTINCT
                        cpt.iteration_id  AS iterationid,
                        cpt.store_id      AS storeid,
                        cpt.image_file_name,
                        cpt.s3path_annotated_file,
                        DENSE_RANK() OVER (
                            PARTITION BY cpt.iteration_id
                            ORDER BY
                                cpt.store_id,
                                cpt.image_file_name,
                                cpt.s3path_annotated_file
                        ) AS iterationtranid
                    FROM temp.cap_prediction_temp cpt
                    WHERE cpt.iteration_id = {iid}
                )
                INSERT INTO orgi.coolermetricstransaction (
                    iterationid, iterationtranid, shelfnumber,
                    productsequenceno, productclassid,
                    x1, y1, x2, y2, confidence,
                    imagefilename, s3path_actual_file, s3path_annotated_file
                )
                SELECT
                    cpt.iteration_id,
                    im.iterationtranid,
                    cpt.shelfnumber,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            cpt.iteration_id,
                            cpt.store_id,
                            cpt.image_file_name,
                            cpt.s3path_annotated_file,
                            cpt.shelfnumber
                        ORDER BY cpt.prod_class_id, cpt.x1
                    ) AS productsequenceno,
                    cpt.prod_class_id,
                    cpt.x1, cpt.y1, cpt.x2, cpt.y2,
                    NULL,
                    cpt.image_file_name,
                    'June_store_images/' || cpt.image_file_name,
                    cpt.s3path_annotated_file
                FROM temp.cap_prediction_temp cpt
                JOIN image_map im
                  ON  im.iterationid           = cpt.iteration_id
                 AND im.storeid                = cpt.store_id
                 AND im.image_file_name        = cpt.image_file_name
                 AND im.s3path_annotated_file  = cpt.s3path_annotated_file
                WHERE cpt.iteration_id = {iid};
            """,
        },
        # ── 6 ─────────────────────────────────────────────────────────────────
        # Insert "other" brand SKU detections from temp.sku_prediction_temp into
        # orgi.coolermetricstransaction with productclassid=26.
        # iterationtranid is resolved per image via the same DENSE_RANK() over
        # cap_prediction_temp that Steps 4 and 5 use — prevents fan-out when a
        # store has multiple images.
        # productsequenceno is offset past the existing max for that
        # (iterationtranid, shelfnumber) slot to avoid PK collisions with
        # cap rows already written by Step 5.
        {
            "step": 6,
            "name": "Insert other-brand SKU rows into coolermetricstransaction",
            "sql": f"""
                WITH image_map AS (
                    SELECT DISTINCT
                        cpt.iteration_id,
                        cpt.store_id,
                        cpt.image_file_name,
                        DENSE_RANK() OVER (
                            PARTITION BY cpt.iteration_id
                            ORDER BY
                                cpt.store_id,
                                cpt.image_file_name,
                                cpt.s3path_annotated_file
                        ) AS iterationtranid
                    FROM temp.cap_prediction_temp cpt
                    WHERE cpt.iteration_id = {iid}
                ),
                max_seq AS (
                    SELECT
                        iterationid,
                        iterationtranid,
                        shelfnumber,
                        MAX(productsequenceno) AS max_seq
                    FROM orgi.coolermetricstransaction
                    WHERE iterationid = {iid}
                    GROUP BY iterationid, iterationtranid, shelfnumber
                )
                INSERT INTO orgi.coolermetricstransaction (
                    iterationid,
                    iterationtranid,
                    shelfnumber,
                    productsequenceno,
                    productclassid,
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence,
                    imagefilename,
                    s3path_actual_file,
                    s3path_annotated_file
                )
                SELECT
                    im.iteration_id,
                    im.iterationtranid,
                    s.shelfnumber,
                    COALESCE(mx.max_seq, 0)
                      + ROW_NUMBER() OVER (
                            PARTITION BY im.iterationtranid, s.shelfnumber
                            ORDER BY s.image_file_name, s.x1, s.y1
                        ) AS productsequenceno,
                    59                          AS productclassid,
                    s.x1,
                    s.y1,
                    s.x2,
                    s.y2,
                    1.0000                      AS confidence,
                    s.image_file_name,
                    NULL                        AS s3path_actual_file,
                    s.s3path_annotated_file
                FROM temp.sku_prediction_temp s
                JOIN image_map im
                  ON  im.iteration_id    = s.iteration_id
                 AND  im.image_file_name = s.image_file_name
                JOIN orgi.coolermetricsmaster m
                  ON  m.iterationid     = im.iteration_id
                 AND  m.iterationtranid = im.iterationtranid
                LEFT JOIN max_seq mx
                  ON  mx.iterationid     = im.iteration_id
                 AND  mx.iterationtranid = im.iterationtranid
                 AND  mx.shelfnumber     = s.shelfnumber
                WHERE s.iteration_id = {iid}
                  AND LOWER(s.brand_name) = 'other'
                ON CONFLICT (iterationid, iterationtranid, shelfnumber, productsequenceno)
                DO NOTHING;
            """,
        },
        # ── 7 ─────────────────────────────────────────────────────────────────
        # Insert "alcohol" brand SKU detections from temp.sku_prediction_temp
        # into orgi.coolermetricstransaction with productclassid=25.
        # Identical structure to Step 6 (other-brand insert): iterationtranid
        # resolved via the same DENSE_RANK() over cap_prediction_temp, and
        # productsequenceno offset past the current max for that
        # (iterationtranid, shelfnumber) slot — recomputed here so it also
        # accounts for the other-brand rows Step 6 just inserted.
        {
            "step": 7,
            "name": "Insert alcohol-brand SKU rows into coolermetricstransaction",
            "sql": f"""
                WITH image_map AS (
                    SELECT DISTINCT
                        cpt.iteration_id,
                        cpt.store_id,
                        cpt.image_file_name,
                        DENSE_RANK() OVER (
                            PARTITION BY cpt.iteration_id
                            ORDER BY
                                cpt.store_id,
                                cpt.image_file_name,
                                cpt.s3path_annotated_file
                        ) AS iterationtranid
                    FROM temp.cap_prediction_temp cpt
                    WHERE cpt.iteration_id = {iid}
                ),
                max_seq AS (
                    SELECT
                        iterationid,
                        iterationtranid,
                        shelfnumber,
                        MAX(productsequenceno) AS max_seq
                    FROM orgi.coolermetricstransaction
                    WHERE iterationid = {iid}
                    GROUP BY iterationid, iterationtranid, shelfnumber
                )
                INSERT INTO orgi.coolermetricstransaction (
                    iterationid,
                    iterationtranid,
                    shelfnumber,
                    productsequenceno,
                    productclassid,
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence,
                    imagefilename,
                    s3path_actual_file,
                    s3path_annotated_file
                )
                SELECT
                    im.iteration_id,
                    im.iterationtranid,
                    s.shelfnumber,
                    COALESCE(mx.max_seq, 0)
                      + ROW_NUMBER() OVER (
                            PARTITION BY im.iterationtranid, s.shelfnumber
                            ORDER BY s.image_file_name, s.x1, s.y1
                        ) AS productsequenceno,
                    25                          AS productclassid,
                    s.x1,
                    s.y1,
                    s.x2,
                    s.y2,
                    1.0000                      AS confidence,
                    s.image_file_name,
                    NULL                        AS s3path_actual_file,
                    s.s3path_annotated_file
                FROM temp.sku_prediction_temp s
                JOIN image_map im
                  ON  im.iteration_id    = s.iteration_id
                 AND  im.image_file_name = s.image_file_name
                JOIN orgi.coolermetricsmaster m
                  ON  m.iterationid     = im.iteration_id
                 AND  m.iterationtranid = im.iterationtranid
                LEFT JOIN max_seq mx
                  ON  mx.iterationid     = im.iteration_id
                 AND  mx.iterationtranid = im.iterationtranid
                 AND  mx.shelfnumber     = s.shelfnumber
                WHERE s.iteration_id = {iid}
                  AND LOWER(s.brand_name) = 'alcohol'
                ON CONFLICT (iterationid, iterationtranid, shelfnumber, productsequenceno)
                DO NOTHING;
            """,
        },
        # ── 8 ─────────────────────────────────────────────────────────────────
        {
            "step": 8,
            "name": "Update caserid on coolermetricsmaster",
            "sql": f"""
                UPDATE orgi.coolermetricsmaster c
                SET caserid = p.caserid
                FROM orgi.storemaster s
                JOIN orgi.puritymapping p
                  ON lower(p.casername) = lower(s.cooler)
                WHERE s.storeid      = c.storeid
                  AND c.iterationid  = {iid};
            """,
        },
        # ── 9 ─────────────────────────────────────────────────────────────────
        # BUG FIX: Extra rows for structural counts were being duplicated across
        # multiple images of the same store.
        #
        # Root cause (old code)
        # ─────────────────────
        # The old Step 6 joined structural_count_temp directly to
        # coolermetricsmaster ON (iterationid, storeid) with NO image_file_name
        # filter.  A store with 2 images has 2 rows in coolermetricsmaster
        # (iterationtranid=1 and iterationtranid=2).  Every row in
        # structural_count_temp (which is per-image) therefore matched BOTH
        # coolermetricsmaster rows and generated extra_count rows × 2.
        #
        # Example (this run):
        #   Coca-Cola img2 → extra_count=4, matched iterationtranid 1 AND 2
        #   → 8 extra rows attempted → some survived ON CONFLICT → DB got 24
        #   instead of the correct 20.
        #
        #   Sprite img1   → extra_count=7, matched iterationtranid 1 AND 2
        #   → 14 rows attempted → productsequenceno collisions → some dropped
        #   → DB got 55 instead of the correct 56.
        #
        # Fix
        # ───
        # Resolve iterationtranid per image by joining structural_count_temp
        # through cap_prediction_temp (which carries image_file_name and the
        # same DENSE_RANK ordering used in Steps 3/4).  This guarantees exactly
        # one iterationtranid per structural_count_temp row regardless of how
        # many images the store has.
        #
        # productsequenceno offset
        # ────────────────────────
        # MAX() is intentionally NOT filtered by productclassid.  The PK of
        # coolermetricstransaction is (iterationid, iterationtranid, shelfnumber,
        # productsequenceno) with no productclassid column, so we must offset
        # past the global max for that (iterationtranid, shelfnumber=0) slot to
        # avoid collisions with rows already written by Step 5 for other SKUs.
        {
            "step": 9,
            "name": "Insert structural-count extra rows",
            "sql": f"""
                INSERT INTO orgi.coolermetricstransaction (
                    iterationid,
                    iterationtranid,
                    shelfnumber,
                    productsequenceno,
                    productclassid,
                    imagefilename,
                    s3path_actual_file,
                    s3path_annotated_file
                )
                SELECT
                    sc.iteration_id,
                    im.iterationtranid,
                    0                           AS shelfnumber,
                    ROW_NUMBER() OVER (
                        PARTITION BY sc.iteration_id, im.iterationtranid
                        ORDER BY sc.prod_class_id
                    ) + COALESCE((
                        SELECT MAX(t2.productsequenceno)
                        FROM orgi.coolermetricstransaction t2
                        WHERE t2.iterationid     = sc.iteration_id
                          AND t2.iterationtranid = im.iterationtranid
                          AND t2.shelfnumber     = 0
                        -- Intentionally NO productclassid filter here.
                        -- The PK includes productsequenceno but NOT productclassid,
                        -- so we must offset past the global max for this
                        -- (iterationtranid, shelfnumber) slot to avoid collisions
                        -- with rows inserted for other SKUs in Step 5.
                    ), 0)                       AS productsequenceno,
                    sc.prod_class_id,
                    sc.image_file_name,
                    'June_store_images/' || sc.image_file_name,
                    NULL                        AS s3path_annotated_file
                FROM temp.structural_count_temp sc
                -- ── FIX: resolve iterationtranid per image ───────────────────
                -- Join through cap_prediction_temp to get the DENSE_RANK value
                -- for this specific image_file_name, matching exactly what
                -- Steps 4 and 5 computed.  Without image_file_name in the join
                -- a store with N images fans out to N iterationtranid matches
                -- and inserts N × extra_count rows instead of extra_count.
                JOIN (
                    SELECT DISTINCT
                        cpt.iteration_id,
                        cpt.store_id,
                        cpt.image_file_name,
                        DENSE_RANK() OVER (
                            PARTITION BY cpt.iteration_id
                            ORDER BY
                                cpt.store_id,
                                cpt.image_file_name,
                                cpt.s3path_annotated_file
                        ) AS iterationtranid
                    FROM temp.cap_prediction_temp cpt
                    WHERE cpt.iteration_id = {iid}
                ) im
                  ON  im.iteration_id    = sc.iteration_id
                 AND  im.store_id::TEXT  = sc.store_id::TEXT
                 AND  im.image_file_name = sc.image_file_name
                -- ── belt-and-braces: verify master row exists ─────────────────
                JOIN orgi.coolermetricsmaster cm
                  ON  cm.iterationid     = im.iteration_id
                 AND  cm.iterationtranid = im.iterationtranid
                CROSS JOIN generate_series(1, sc.extra_count)
                WHERE sc.iteration_id = {iid}
                  AND EXISTS (
                        SELECT 1 FROM temp.structural_count_temp
                        WHERE iteration_id = {iid}
                  )
                ON CONFLICT (iterationid, iterationtranid, shelfnumber, productsequenceno)
                DO NOTHING;
            """,
        },
        # ── 10 ────────────────────────────────────────────────────────────────
        {
            "step": 10,
            "name": "Insert into reference_table",
            "sql": f"""
                INSERT INTO orgi.reference_table
                (
                    iteration_id,
                    staging_id,
                    ai_processed,
                    java_batch_processed,
                    model_run
                )
                VALUES
                (
                    {iid},
                    {sid},
                    TRUE,
                    FALSE,
                    CURRENT_DATE::text
                )
                ON CONFLICT (iteration_id) DO NOTHING
            """,
        },
        # ── 11 ────────────────────────────────────────────────────────────────
        {
            "step": 11,
            "name": "Validate reference_table against visibilityitemsstaging",
            "sql": f"""
                UPDATE orgi.reference_table rt
                SET staging_id = NULL
                WHERE rt.staging_id = (
                    SELECT MAX(staging_id)
                    FROM orgi.reference_table
                )
                AND (
                    SELECT COUNT(*)
                    FROM orgi.visibilityitemsstaging vs
                    WHERE vs.stagingid = (
                        SELECT MAX(staging_id)
                        FROM orgi.reference_table
                    )
                ) = 0;
            """,
        },
    ]


# ── Core runner ────────────────────────────────────────────────────────────────
def run_cap_pipeline(db_config: dict, iteration_id: int, staging_id: int) -> PipelineResult:
    """
    Execute all 11 CAP post-processing steps sequentially.

    Steps 1-9 are the existing cap/SKU post-processing SQL (including the
    Step 3 backfill of missing cap rows). Steps 10-11 (reference_table
    insert + validation) only run if steps 1-9 all succeed — the loop
    below aborts and rolls back on the first failure, so a failed
    step 1-9 means steps 10-11 are simply never reached.

    Parameters
    ----------
    db_config : dict
        Database credentials from config.json  (keys: host, port, database,
        user, password).
    iteration_id : int
        The iteration ID used in the current pipeline run (same value passed
        to visicooler / ollama steps).
    staging_id : int
        The staging ID used in the current pipeline run (same value passed
        to execute_models / insert_ollama_results). Needed for Step 9.

    Returns
    -------
    PipelineResult
        Dataclass containing per-step outcomes and the overall status.
        Always logs a formatted summary table to the application logger.
    """
    pipeline_start = time.monotonic()
    step_results: list[StepResult] = []
    overall_status = "success"

    logger.info("=" * 100)
    logger.info(f"  CAP PREDICTION PIPELINE  –  START  (iteration_id={iteration_id})")
    logger.info("=" * 100)

    # ── Open connection ────────────────────────────────────────────────────────
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            dbname=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
        )
        conn.autocommit = False
        logger.info(
            f"  Connected to {db_config['host']}:{db_config['port']} / {db_config['database']}"
        )
    except Exception as exc:
        logger.error(f"  Database connection failed: {exc}")
        result = PipelineResult(
            iteration_id=iteration_id,
            overall_status="failed",
            total_duration_ms=round((time.monotonic() - pipeline_start) * 1000, 2),
            steps=[
                StepResult(
                    step=0,
                    name="Database connection",
                    status="failed",
                    rows_affected=None,
                    duration_ms=0.0,
                    error=str(exc),
                )
            ],
        )
        result.log_summary()
        return result

    # ── Execute steps ──────────────────────────────────────────────────────────
    try:
        with conn:
            with conn.cursor() as cur:
                for query in _build_queries(iteration_id, staging_id):
                    step_start = time.monotonic()
                    step_num = query["step"]
                    step_name = query["name"]

                    logger.info(f"  ▶  Step {step_num}/11 – {step_name} …")

                    try:
                        cur.execute(query["sql"])
                        rows = cur.rowcount if cur.rowcount >= 0 else None
                        duration = round((time.monotonic() - step_start) * 1000, 2)

                        step_results.append(
                            StepResult(
                                step=step_num,
                                name=step_name,
                                status="success",
                                rows_affected=rows,
                                duration_ms=duration,
                            )
                        )
                        rows_label = f"{rows} rows" if rows is not None else ""
                        logger.info(
                            f"     ✓  Completed in {duration:.1f} ms  {rows_label}"
                        )

                    except Exception as exc:
                        duration = round((time.monotonic() - step_start) * 1000, 2)
                        err_trace = traceback.format_exc()

                        step_results.append(
                            StepResult(
                                step=step_num,
                                name=step_name,
                                status="failed",
                                rows_affected=None,
                                duration_ms=duration,
                                error=err_trace,
                            )
                        )
                        logger.error(
                            f"     ✗  FAILED in {duration:.1f} ms – rolling back transaction"
                        )
                        logger.error(err_trace)

                        overall_status = "failed"
                        conn.rollback()
                        break  # abort remaining steps

    finally:
        try:
            conn.close()
        except Exception:
            pass

    # ── Assemble and return result ─────────────────────────────────────────────
    total_ms = round((time.monotonic() - pipeline_start) * 1000, 2)

    result = PipelineResult(
        iteration_id=iteration_id,
        overall_status=overall_status,
        total_duration_ms=total_ms,
        steps=step_results,
    )
    result.log_summary()
    return result


# ── Post-shelf-sequence-check reclassification ──────────────────────────────────
def reclassify_low_detection_impure(db_config: dict, iteration_id: int) -> int:
    """
    Reclassify low-detection-count IMPURE rows back to PURE, for THIS
    iteration_id: any orgi.shelfsequencecompliance row currently marked
    IMPURE whose orgi.dino_nonbeverage_detections total between 1 and 3
    (inclusive) is treated as false-positive noise and reset to PURE.

    IMPORTANT — call this AFTER run_shelf_sequence_check(), not as a step
    inside run_cap_pipeline(). run_cap_pipeline() executes before
    run_shelf_sequence_check() in main.py's orchestration, and it's
    run_shelf_sequence_check() that actually populates purity_status on
    shelfsequencecompliance and inserts rows into
    dino_nonbeverage_detections for the current iteration. Running this
    query any earlier means both tables are still empty/stale for this
    iteration_id, so it will silently match 0 rows every time — which is
    exactly what happened when this was briefly wired in as
    run_cap_pipeline's Step 12.

    Parameters
    ----------
    db_config : dict
        Database credentials from config.json (keys: host, port, database,
        user, password).
    iteration_id : int
        The iteration ID whose shelfsequencecompliance rows should be
        reclassified. Only rows for this iteration are touched.

    Returns
    -------
    int
        Number of shelfsequencecompliance rows flipped from IMPURE to
        PURE. Returns 0 on any failure — logs the error but does not
        raise, since a failure here should not fail the overall pipeline
        run (the purity_status values just stay as run_shelf_sequence_check
        left them).
    """
    iid = int(iteration_id)  # guard against injection

    conn = None
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            dbname=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
        )
        conn.autocommit = False

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE orgi.shelfsequencecompliance s
                    SET purity_status = 'PURE'
                    WHERE (s.iterationid, s.iterationtranid) IN (
                        SELECT
                            s1.iterationid,
                            s1.iterationtranid
                        FROM orgi.shelfsequencecompliance s1
                        LEFT JOIN orgi.dino_nonbeverage_detections d
                            ON s1.iterationid = d.iterationid
                           AND s1.iterationtranid = d.iterationtranid
                        WHERE s1.iterationid = {iid}
                          AND s1.purity_status = 'IMPURE'
                        GROUP BY
                            s1.iterationid,
                            s1.iterationtranid
                        HAVING COUNT(d.productname) > 0
                           AND COUNT(d.productname) <= 1
                    );
                    """
                )
                rows = cur.rowcount if cur.rowcount >= 0 else 0

        logger.info(
            f"  Reclassify low-detection-count IMPURE rows -> PURE "
            f"(iteration_id={iid}): {rows} row(s) updated"
        )
        return rows

    except Exception as exc:
        logger.error(
            f"  Reclassify low-detection-count IMPURE rows failed "
            f"(iteration_id={iid}): {exc}"
        )
        logger.error(traceback.format_exc())
        return 0

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ── Planogram (subcategory-603) pipeline ────────────────────────────────────────
def _build_planogram_queries(iteration_id: int, base_tranid: int) -> list[dict]:
    """
    SQL steps that populate orgi.coolermetricsmaster_planogram /
    orgi.coolermetricstransaction_planogram from temp.sku_planogram_temp —
    the planogram-detector's SKU-only equivalent of steps 1, 4 and 5 of the
    main cap pipeline. There is no cap<->SKU matching step here: every row
    in temp.sku_planogram_temp is already a final, resolved detection (see
    planogram_detector.py), and there are no "other"/"alcohol"/structural-
    count filler steps because sku_planogram_temp only ever holds real
    resolved-brand detections (per product decision).

    iterationtranid offset — MUST be a fixed literal, not a live subquery
    ------------------------------------------------------------------------
    orgi.shelfsequencecompliance's PK is (iterationid, iterationtranid), and
    both this table and orgi.coolermetricsmaster share the same iterationid
    for a given pipeline run. To avoid tranid collisions when both flows'
    results land in that one shared compliance table, planogram
    iterationtranids continue on from `base_tranid` (MAX(iterationtranid)
    already used in orgi.coolermetricsmaster for this iterationid, computed
    ONCE by the caller — see run_planogram_pipeline).

    This value is passed in as a plain int and inlined as a literal in both
    Step 2 and Step 3 below — it must NOT be recomputed independently in
    each step via `SELECT MAX(...) FROM orgi.coolermetricsmaster_planogram`,
    because Step 2 itself inserts new rows into that table. If Step 3 were
    to re-read that MAX() live, it would see Step 2's own inserts and
    compute a HIGHER base, so its DENSE_RANK-derived iterationtranids
    would silently drift out of step with the tranids Step 2 actually
    wrote — leaving master rows with no matching transaction rows (and vice
    versa), which surfaces downstream as NULL imagefilename/purity_status
    in orgi.shelfsequencecompliance. Using one literal for both steps keeps
    the DENSE_RANK output identical between them.

    Ordering
    --------
    Each row in temp.sku_planogram_temp already carries the TRUE physical
    shelf position (shelfnumber, 1 = topmost) assigned by
    planogram_detector.py's shelf-segmentation step — a single
    subcategory-603 image can contain several shelves. So the DENSE_RANK
    that assigns iterationtranid (one per physical shelf, i.e. one per
    (store_id, image_file_name, shelfnumber)) orders by shelfnumber
    numerically — an alphabetical sort on "..._shelf10..." vs
    "..._shelf2..." would rank shelf 10 above shelf 2, corrupting the
    top-to-bottom ordering the shelf-sequence checker relies on.

    image_file_name / s3path_actual_file
    -------------------------------------
    temp.sku_planogram_temp.image_file_name is the per-SHELF name (e.g.
    "IMG-XXXX_shelf1.jpg" — see planogram_detector.py), not the original
    uploaded filename, since each shelf crop is analyzed and annotated
    independently. That per-shelf name is what's stored as
    coolermetricstransaction_planogram.imagefilename (matches the
    annotated-crop naming 1:1). s3path_actual_file, however, should still
    point at the real RAW uploaded photo in S3 — there is no separate raw
    file per shelf — so the "_shelfN" suffix is stripped back out before
    building that path.
    """
    iid = int(iteration_id)
    base = int(base_tranid)
    return [
        {
            "step": 1,
            "name": "Create unique index on sku_planogram_temp",
            "sql": """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_sku_planogram_box
                ON temp.sku_planogram_temp (
                    store_id, image_file_name, iteration_id, prod_class_id, x1, x2, y1, y2
                );
            """,
        },
        {
            "step": 2,
            "name": "Populate orgi.coolermetricsmaster_planogram",
            "sql": f"""
                WITH image_map AS (
                    SELECT DISTINCT
                        spt.iteration_id AS iterationid,
                        spt.store_id     AS storeid,
                        spt.image_file_name,
                        spt.shelfnumber,
                        {base} + DENSE_RANK() OVER (
                            PARTITION BY spt.iteration_id
                            ORDER BY
                                spt.store_id,
                                spt.image_file_name,
                                spt.shelfnumber
                        ) AS iterationtranid
                    FROM temp.sku_planogram_temp spt
                    WHERE spt.iteration_id = {iid}
                )
                INSERT INTO orgi.coolermetricsmaster_planogram (
                    iterationid, iterationtranid, storeid,
                    caserid, modelrun, processed_flag
                )
                SELECT
                    i.iterationid,
                    i.iterationtranid,
                    i.storeid,
                    pm.caserid,
                    NOW(),
                    'N'
                FROM image_map i
                JOIN orgi.puritymapping pm ON pm.caserid IS NOT NULL
                ON CONFLICT (iterationid, iterationtranid) DO NOTHING;
            """,
        },
        {
            "step": 3,
            "name": "Populate orgi.coolermetricstransaction_planogram",
            "sql": f"""
                WITH image_map AS (
                    SELECT DISTINCT
                        spt.iteration_id AS iterationid,
                        spt.store_id     AS storeid,
                        spt.image_file_name,
                        spt.shelfnumber,
                        {base} + DENSE_RANK() OVER (
                            PARTITION BY spt.iteration_id
                            ORDER BY
                                spt.store_id,
                                spt.image_file_name,
                                spt.shelfnumber
                        ) AS iterationtranid
                    FROM temp.sku_planogram_temp spt
                    WHERE spt.iteration_id = {iid}
                )
                INSERT INTO orgi.coolermetricstransaction_planogram (
                    iterationid, iterationtranid, shelfnumber,
                    productsequenceno, productclassid,
                    x1, y1, x2, y2, confidence,
                    imagefilename, s3path_actual_file, s3path_annotated_file
                )
                SELECT
                    spt.iteration_id,
                    im.iterationtranid,
                    spt.shelfnumber,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            spt.iteration_id,
                            spt.store_id,
                            spt.image_file_name,
                            spt.shelfnumber
                        ORDER BY spt.prod_class_id, spt.x1
                    ) AS productsequenceno,
                    spt.prod_class_id,
                    spt.x1, spt.y1, spt.x2, spt.y2,
                    NULL,
                    spt.image_file_name,
                    'June_store_images/' || regexp_replace(spt.image_file_name, '_shelf[0-9]+', ''),
                    spt.s3path_annotated_file
                FROM temp.sku_planogram_temp spt
                JOIN image_map im
                  ON  im.iterationid     = spt.iteration_id
                 AND im.storeid         = spt.store_id
                 AND im.image_file_name = spt.image_file_name
                 AND im.shelfnumber     = spt.shelfnumber
                WHERE spt.iteration_id = {iid}
                ON CONFLICT (iterationid, iterationtranid, shelfnumber, productsequenceno)
                DO NOTHING;
            """,
        },
        {
            "step": 4,
            "name": "Update caserid on coolermetricsmaster_planogram",
            "sql": f"""
                UPDATE orgi.coolermetricsmaster_planogram c
                SET caserid = p.caserid
                FROM orgi.storemaster s
                JOIN orgi.puritymapping p
                  ON lower(p.casername) = lower(s.cooler)
                WHERE s.storeid      = c.storeid
                  AND c.iterationid  = {iid};
            """,
        },
    ]


def _ensure_planogram_tables(cur) -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orgi.coolermetricsmaster_planogram (
            iterationid int4 NOT NULL,
            iterationtranid int4 NOT NULL,
            storeid int8 NOT NULL,
            caserid int4 NOT NULL,
            modelrun timestamp NULL,
            processed_flag bpchar(1) DEFAULT 'N'::bpchar NULL,
            CONSTRAINT coolermetricsmaster_planogram_pkey PRIMARY KEY (iterationid, iterationtranid)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orgi.coolermetricstransaction_planogram (
            iterationid int4 NOT NULL,
            iterationtranid int4 NOT NULL,
            shelfnumber int4 NOT NULL,
            productsequenceno int4 NOT NULL,
            productclassid int4 NULL,
            x1 numeric(10, 4) NULL,
            y1 numeric(10, 4) NULL,
            x2 numeric(10, 4) NULL,
            y2 numeric(10, 4) NULL,
            confidence numeric(10, 4) NULL,
            imagefilename varchar NULL,
            s3path_actual_file varchar NULL,
            s3path_annotated_file varchar NULL,
            CONSTRAINT coolermetricstransaction_planogram_pkey PRIMARY KEY (
                iterationid, iterationtranid, shelfnumber, productsequenceno
            )
        )
    """)
    logger.info("Verified orgi.coolermetricsmaster_planogram / coolermetricstransaction_planogram exist")


def _compute_planogram_base_tranid(cur, iteration_id: int) -> int:
    """
    Single, one-time computation of the tranid offset planogram shelves
    should continue from — see the big comment in _build_planogram_queries
    for why this must be computed exactly once and reused as a literal
    rather than re-queried live inside each SQL step.

    IMPORTANT — must depend ONLY on orgi.coolermetricsmaster (605), never
    on orgi.coolermetricsmaster_planogram's own max. The 605 table's max
    tranid for a given iteration_id is stable across repeat calls (its
    DENSE_RANK has no offset, so re-running run_cap_pipeline for the same
    iteration always recomputes the identical values and ON CONFLICT DO
    NOTHING no-ops correctly). If this function also folded in
    orgi.coolermetricsmaster_planogram's current max, the base would grow
    every time run_planogram_pipeline is invoked for that same
    iteration_id (retry, re-run, resumed run — same iteration_id reused),
    so a second invocation would compute an entirely NEW, non-colliding
    tranid range and insert the exact same shelves again as duplicates —
    which is exactly what caused the doubled rows in
    orgi.shelfsequencecompliance. Keying only off the 605 table makes this
    call idempotent: the same iteration_id always yields the same base,
    so re-running safely collides with (and is skipped by) the existing
    ON CONFLICT DO NOTHING inserts instead of duplicating.
    """
    cur.execute(
        """
        SELECT COALESCE(
            (SELECT MAX(iterationtranid) FROM orgi.coolermetricsmaster
             WHERE iterationid = %s),
            0
        )
        """,
        (iteration_id,),
    )
    return int(cur.fetchone()[0])


def run_planogram_pipeline(db_config: dict, iteration_id: int) -> PipelineResult:
    """
    Execute the 4-step planogram post-processing pipeline, populating
    orgi.coolermetricsmaster_planogram / orgi.coolermetricstransaction_planogram
    from temp.sku_planogram_temp for subcategory-603 images.

    Call this AFTER run_cap_pipeline() for the same iteration_id, since the
    iterationtranid-offset logic reads orgi.coolermetricsmaster's current
    max tranid for this iteration to avoid colliding with it in the shared
    orgi.shelfsequencecompliance table downstream.
    """
    pipeline_start = time.monotonic()
    step_results: list[StepResult] = []
    overall_status = "success"

    logger.info("=" * 100)
    logger.info(f"  PLANOGRAM PIPELINE  –  START  (iteration_id={iteration_id})")
    logger.info("=" * 100)

    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            dbname=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
        )
        conn.autocommit = False
    except Exception as exc:
        logger.error(f"  Database connection failed: {exc}")
        result = PipelineResult(
            iteration_id=iteration_id,
            overall_status="failed",
            total_duration_ms=round((time.monotonic() - pipeline_start) * 1000, 2),
            steps=[StepResult(step=0, name="Database connection", status="failed",
                               rows_affected=None, duration_ms=0.0, error=str(exc))],
            pipeline_label="PLANOGRAM PIPELINE",
            total_steps=4,
        )
        result.log_summary()
        return result

    try:
        with conn:
            with conn.cursor() as cur:
                _ensure_planogram_tables(cur)
                base_tranid = _compute_planogram_base_tranid(cur, iteration_id)
                logger.info(f"  Planogram iterationtranid base: {base_tranid}")

                for query in _build_planogram_queries(iteration_id, base_tranid):
                    step_start = time.monotonic()
                    step_num = query["step"]
                    step_name = query["name"]
                    logger.info(f"  ▶  Step {step_num}/4 – {step_name} …")
                    try:
                        cur.execute(query["sql"])
                        rows = cur.rowcount if cur.rowcount >= 0 else None
                        duration = round((time.monotonic() - step_start) * 1000, 2)
                        step_results.append(StepResult(
                            step=step_num, name=step_name, status="success",
                            rows_affected=rows, duration_ms=duration,
                        ))
                        logger.info(f"     ✓  Completed in {duration:.1f} ms")
                    except Exception as exc:
                        duration = round((time.monotonic() - step_start) * 1000, 2)
                        err_trace = traceback.format_exc()
                        step_results.append(StepResult(
                            step=step_num, name=step_name, status="failed",
                            rows_affected=None, duration_ms=duration, error=err_trace,
                        ))
                        logger.error(f"     ✗  FAILED in {duration:.1f} ms – rolling back transaction")
                        logger.error(err_trace)
                        overall_status = "failed"
                        conn.rollback()
                        break
    finally:
        try:
            conn.close()
        except Exception:
            pass

    total_ms = round((time.monotonic() - pipeline_start) * 1000, 2)
    result = PipelineResult(
        iteration_id=iteration_id,
        overall_status=overall_status,
        total_duration_ms=total_ms,
        steps=step_results,
        pipeline_label="PLANOGRAM PIPELINE",
        total_steps=4,
    )
    result.log_summary()
    return result