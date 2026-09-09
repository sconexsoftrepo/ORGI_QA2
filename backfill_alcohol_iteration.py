"""
backfill_alcohol_iteration.py
==============================
One-off backfill for iterations that already completed the CAP pipeline
BEFORE the alcohol-insert step existed (e.g. iteration_id = 260).

It re-runs *only* the alcohol-brand SKU insert (the same query as Step 6 in
cap_pipeline_runner.py) for the given iteration_id. It does NOT touch
Steps 1-5, 7, 8 — coolermetricsmaster / coolermetricstransaction rows
already written for that iteration are left exactly as they are; this only
adds the missing alcohol rows (productclassid = 25) that Step 6 would have
inserted.

Safe to re-run: the INSERT uses ON CONFLICT DO NOTHING on the same PK as
the live table, so running this twice for the same iteration is a no-op
the second time.

Usage:
    python backfill_alcohol_iteration.py 260
    python backfill_alcohol_iteration.py 260 --dry-run
"""

import sys
import argparse
import logging

import psycopg2

from app.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_alcohol_backfill_sql(iteration_id: int) -> str:
    iid = int(iteration_id)  # guard against injection
    return f"""
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
    """


def preflight_checks(cur, iteration_id: int) -> None:
    """Sanity checks so we fail loudly instead of silently inserting 0 rows."""
    iid = int(iteration_id)

    cur.execute(
        "SELECT COUNT(*) FROM temp.sku_prediction_temp "
        "WHERE iteration_id = %s AND LOWER(brand_name) = 'alcohol'",
        (iid,),
    )
    alcohol_count = cur.fetchone()[0]
    logger.info(f"temp.sku_prediction_temp alcohol rows for iteration {iid}: {alcohol_count}")

    cur.execute(
        "SELECT COUNT(*) FROM orgi.coolermetricsmaster WHERE iterationid = %s",
        (iid,),
    )
    master_count = cur.fetchone()[0]
    logger.info(f"orgi.coolermetricsmaster rows for iteration {iid}: {master_count}")

    if alcohol_count == 0:
        raise RuntimeError(
            f"No 'alcohol' rows found in temp.sku_prediction_temp for iteration_id={iid}. "
            f"Nothing to backfill — check the iteration_id or whether the temp table "
            f"was cleared since that run."
        )
    if master_count == 0:
        raise RuntimeError(
            f"No rows in orgi.coolermetricsmaster for iteration_id={iid}. "
            f"Steps 3/4 of the pipeline must have run for this iteration first."
        )


def run_backfill(db_config: dict, iteration_id: int, dry_run: bool = False) -> int:
    conn = psycopg2.connect(
        host=db_config["host"],
        port=db_config["port"],
        dbname=db_config["database"],
        user=db_config["user"],
        password=db_config["password"],
    )
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            preflight_checks(cur, iteration_id)

            sql = build_alcohol_backfill_sql(iteration_id)
            cur.execute(sql)
            rows_inserted = cur.rowcount

            if dry_run:
                logger.info(f"[DRY RUN] Would insert {rows_inserted} alcohol rows — rolling back.")
                conn.rollback()
            else:
                conn.commit()
                logger.info(f"Inserted {rows_inserted} alcohol rows for iteration_id={iteration_id}.")

            return rows_inserted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill missing alcohol-brand SKU rows for one iteration.")
    parser.add_argument("iteration_id", type=int, help="Iteration ID to backfill, e.g. 260")
    parser.add_argument("--dry-run", action="store_true", help="Run inside a transaction and roll back (no writes).")
    parser.add_argument("--config", default="config.json", help="Path to config.json (default: ./config.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    db_config = config["db_config"]

    try:
        rows = run_backfill(db_config, args.iteration_id, dry_run=args.dry_run)
        logger.info(f"Done. rows_affected={rows}")
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()