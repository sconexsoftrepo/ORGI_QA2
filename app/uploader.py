import pandas as pd
import logging
from app.db_handler import initialize_db_connection, close_db_connection, insert_ollama_results, get_max_stagingid, get_classtext
from app.visicooler import upload_to_visibilitydetails
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_ollama_results(csv_path, db_config, modelname, image_paths, s3_annotated_folder, cyclecountid, s3_handler):
    """
    Upload Ollama results to orgi.visibilityitemsstaging and update orgi.visibilitydetails.
    This function is now optional as ollama_analyzer.py handles DB insertion directly.
    """
    conn = None
    cur = None
    try:
        conn, cur = initialize_db_connection(db_config)
        stagingid = get_max_stagingid(cur) + 1
        logger.info(f"Using stagingid: {stagingid}")

        results = []
        if not pd.io.common.file_exists(csv_path):
            logger.warning(f"CSV file {csv_path} does not exist.")
        else:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                results.append({
                    'rowid': int(row['rowid']),
                    'modelname': modelname,
                    'imagefilename': row['imagefilename'],
                    'classid': int(row['classid']),
                    'classtext': row['classtext'],
                    'value': str(row['value']),
                    'inference': float(row['inference']),
                    'modelrun': row['modelrun'],
                    'processed_flag': row['processed_flag'],
                    'storeid': row['storeid'] if pd.notna(row['storeid']) else None,
                    'storename': row['storename'] if pd.notna(row['storename']) else None,
                    's3path_actual_file': row['s3path_actual_file'],
                    's3path_annotated_file': row['s3path_annotated_file']
                })

        if results:
            insert_ollama_results(cur, stagingid, results, modelname, s3_annotated_folder, image_paths)
            conn.commit()
            logger.info(f"Successfully uploaded {len(results)} Ollama results to orgi.visibilityitemsstaging with stagingid {stagingid}.")
        else:
            logger.warning("No valid Ollama results to upload to database.")

        # Process visicooler-specific data
        image_data = defaultdict(lambda: {
            'numshelf': 0,
            'numpureshelf': 0,
            'visicooler_size': "",
            'percent_rgb': 0.0,
            'chilled_items': 0,
            'warm_items': 0,
            'skus_detected': 'N',
            'share_chilled': 0.0,
            'share_warm': 0.0,
            'present_no_facings': 'N',
            'coke_pepsi_chilled': 0,
            'coke_pepsi_warm': 0
        })

        for result in results:
            image = result['imagefilename']
            classid = result['classid']
            value = result['value']

            if classid == 1012:
                image_data[image]['numshelf'] = int(value) if value.isdigit() else 0
            elif classid == 1013:
                image_data[image]['numpureshelf'] = int(value) if value.isdigit() else 0
            elif classid == 1003:
                image_data[image]['visicooler_size'] = value
            elif classid == 1014:
                try:
                    percent_rgb = float(value.strip('%')) if isinstance(value, str) and '%' in value else float(value)
                    image_data[image]['percent_rgb'] = percent_rgb
                except ValueError:
                    image_data[image]['percent_rgb'] = 0.0
            elif classid == 1048:
                image_data[image]['chilled_items'] = int(value) if value.isdigit() else 0
            elif classid == 1049:
                image_data[image]['warm_items'] = int(value) if value.isdigit() else 0
            elif classid == 1047:
                image_data[image]['skus_detected'] = value if value in ['Y', 'N'] else 'N'
            elif classid == 1050:
                image_data[image]['coke_pepsi_chilled'] = int(value) if value.isdigit() else 0
            elif classid == 1051:
                image_data[image]['coke_pepsi_warm'] = int(value) if value.isdigit() else 0
            elif classid == 1052:
                image_data[image]['present_no_facings'] = value if value in ['Y', 'N'] else 'N'

        visicooler_records = []
        image_fnames = {fname for _, _, fname, _, _, _, _ in image_paths}
        for image, data in image_data.items():
            if image not in image_fnames:
                logger.warning(f"Image {image} not found in image_paths. Skipping.")
                continue

            total_chilled = data['chilled_items']
            data['share_chilled'] = (data['coke_pepsi_chilled'] / total_chilled * 100) if total_chilled > 0 else 0.0

            total_warm = data['warm_items']
            data['share_warm'] = (data['coke_pepsi_warm'] / total_warm * 100) if total_warm > 0 else 0.0

            visicooler_records.append((
                image,
                data['numshelf'],
                0,
                data['numpureshelf'],
                data['visicooler_size'],
                data['percent_rgb'],
                data['chilled_items'],
                data['warm_items'],
                data['skus_detected'],
                data['share_chilled'],
                data['share_warm'],
                data['present_no_facings']
            ))

        if visicooler_records:
            upload_to_visibilitydetails(conn, cur, visicooler_records, cyclecountid)
        else:
            logger.warning("No visicooler records to upload to orgi.visibilitydetails.")

        return stagingid
    except Exception as e:
        logger.error(f"Failed to upload Ollama results to database: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            close_db_connection(conn, cur)