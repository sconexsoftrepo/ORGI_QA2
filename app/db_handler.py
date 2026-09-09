from ctypes.wintypes import RGB
import logging
import pg8000.dbapi as pg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_db_connection(db_config):
    # Initialize database connection with 30 second timeout
    try:
        conn = pg.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password'],
            timeout=60
        )
        cur = conn.cursor()
        logger.info("Database connection established.")
        return conn, cur
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {e}")
        raise


def close_db_connection(conn, cur):
    # Close database connection safely
    try:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
        logger.info("Database connection closed.")
    except Exception as e:
        # Check if it's just an already closed connection
        error_msg = str(e).lower()
        if 'closed' in error_msg or 'invalid' in error_msg:
            logger.debug(f"Connection was already closed: {e}")
        else:
            logger.error(f"Failed to close database connection: {e}")

def get_max_stagingid(cur):
    """Get the maximum stagingid from orgi.visibilityitemsstaging."""
    try:
        cur.execute("SELECT MAX(stagingid) FROM orgi.visibilityitemsstaging")
        result = cur.fetchone()
        return result[0] if result[0] is not None else 0
    except Exception as e:
        logger.error(f"Failed to get max stagingid: {e}")
        raise

def get_classtext(cur, classid):
    """Get classtext from orgi.productmaster or predefined mappings."""
    predefined_mappings = {
        1001: "Visicooler - Visicooler available",
        1002: "Visicooler - Brand of Visicooler",
        1003: "Coca-Cola Company visicooler - Size of Visicooler",
        1004: "Coca-Cola Company visicooler - Visicooler working",
        1005: "Coca-Cola Company visicooler - Visicooler pics allowed",
        1006: "Coca-Cola Company visicooler - Visicooler visibility from street",
        1007: "Coca-Cola Company visicooler - Visicooler visibility in shop Partial/Fully/Not visible",
        1008: "Coca-Cola Company visicooler - Visicooler accessible",
        1009: "Coca-Cola Company visicooler - Brand shelf strip",
        1010: "Coca-Cola Company visicooler - As per planogram",
        1011: "Coca-Cola Company visicooler - Cooler Purity",
        1012: "Coca-Cola Company visicooler - Number of shelves in cooler",
        1013: "Coca-Cola Company visicooler - Number of pure shelves",
        1014: "Coca-Cola Company visicooler - RGB space occupied",

        1018: "Neck ringer",
        1019: "Poster",
        1020: "Steamer",
        1022: "Combo Board",
        1023: "Menu Board",
        1027: "Table Sticker",
        1034: "Neon Signage",
        1036: "Waiter Apron",
        1040: "Pillar branding",
        1043: "Aerial Hanger",
        1053: "DPS",
        1054: "Crate cover",
        1055: "Window display",
        1056: "RGB Crate Stacking",
        1057: "Flange",
        1059: "Standee",
        1061: "Countertop filled bottle",
        1063: "Wobbler",
        1064: "Shelf Display",
        1074: "RGB Countertop",
        1075: "PET Floor Stacking",
        1076: "PET Rack",
        1077: "Countertop display",
        1078: "3 Tier Rack",
        1079: "ASSP CounterTop",
        1080: "Box Display",
        1081: "Ceiling Pillar Branding",
        1082: "Counter Front",
        1083: "Dangler",
        1084: "Foam Banner",
        1085: "Front Window Branding",
        1086: "Led Wall Or Shelf Mount Element",
        1087: "MT Rack Display",
        1088: "Offer Ribbon Sticker",
        1089: "One Pager Combo Menu",
        1090: "Others",
        1091: "Shelf Display MT",
        1092: "Shopper Gate",
        1093: "Umbrella",
        1094: "Wall Branding",
        1095: "Warm display of 3 Bottles at visible place",
        1096: "cooler strips",
        1097: "shelf branding",
        1098: "Printed Communication Materials",
    }
    if classid in predefined_mappings:
        return predefined_mappings[classid]
    try:
        cur.execute("SELECT productname FROM orgi.productmaster WHERE productclassid = %s", (classid,))
        result = cur.fetchone()
        return result[0] if result else 'Unknown'
    except Exception as e:
        logger.error(f"Failed to get classtext for classid {classid}: {e}")
        return 'Unknown'

def insert_ollama_results(cur, stagingid, results, s3_annotated_folder, image_paths,
                          conn=None, db_config=None, chunk_size=20):
    """Insert Ollama results into orgi.visibilityitemsstaging table.

    Inserts in chunks of `chunk_size` rows to avoid TCP timeout on long inserts.
    If a connection error occurs mid-insert, reconnects automatically (requires
    conn and db_config to be passed in) and retries the failed chunk up to 3 times.

    Note: stagingid is shared across all batches in a single pipeline run,
    so multiple inserts under the same stagingid are expected and correct.
    """
    insert_query = """
    INSERT INTO orgi.visibilityitemsstaging
    (stagingid, rowid, modelname, imagefilename, classid, classtext, value, inference,
     modelrun, processed_flag, storeid, storename, s3path_actual_file, s3path_annotated_file)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    formatted_records = [
        (
            stagingid,
            result['rowid'],
            result['modelname'],
            result['imagefilename'],
            result['classid'],
            result['classtext'],
            result['value'],
            result['inference'],
            result['modelrun'],
            result['processed_flag'],
            result['storeid'],
            result['storename'],
            result['s3path_actual_file'],
            result['s3path_annotated_file'],
        )
        for result in results
    ]

    total = len(formatted_records)
    inserted = 0

    # Split into chunks so each round-trip is short and the TCP connection stays alive
    for chunk_start in range(0, total, chunk_size):
        chunk = formatted_records[chunk_start: chunk_start + chunk_size]
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                cur.executemany(insert_query, chunk)
                # Commit after every chunk so partial progress is saved
                if conn is not None:
                    conn.commit()
                inserted += len(chunk)
                logger.info(
                    f"Inserted rows {chunk_start + 1}–{chunk_start + len(chunk)} "
                    f"of {total} (stagingid {stagingid})"
                )
                break  # chunk succeeded, move on

            except Exception as e:
                error_msg = str(e).lower()
                network_keywords = [
                    'network', 'connection', 'timeout', 'broken pipe',
                    'reset by peer', 'timed out', '10053', '10054', '10060',
                    'winerror', 'connection was aborted', 'connection attempt failed',
                    'host has failed to respond',
                ]
                is_network_error = any(kw in error_msg for kw in network_keywords)

                if is_network_error and conn is not None and db_config is not None and attempt < max_attempts:
                    wait = 5 * attempt
                    logger.warning(
                        f"Network error on chunk {chunk_start}–{chunk_start + len(chunk)} "
                        f"(attempt {attempt}/{max_attempts}): {e}. "
                        f"Reconnecting in {wait}s…"
                    )
                    import time
                    time.sleep(wait)
                    try:
                        close_db_connection(conn, cur)
                    except Exception:
                        pass
                    # Reconnect and update the caller's cursor/connection references
                    new_conn, new_cur = initialize_db_connection(db_config)
                    # Replace references in-place so the rest of the insert uses them
                    conn.__dict__.update(new_conn.__dict__)
                    cur.__dict__.update(new_cur.__dict__)
                    logger.info("Reconnected successfully, retrying chunk…")
                else:
                    logger.error(
                        f"Failed to insert chunk {chunk_start}–{chunk_start + len(chunk)} "
                        f"into orgi.visibilityitemsstaging: {e}"
                    )
                    raise

    logger.info(f"Inserted {inserted} rows into orgi.visibilityitemsstaging with stagingid {stagingid}.")