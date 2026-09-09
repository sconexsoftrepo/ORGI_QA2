import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileUploader:
    def __init__(self, config=None):
        self.config = config

    def update_processed_flag(self, conn, processed_files):
        """Update processed_flag in orgi.fileupload for processed files."""
        failed_updates = []
        try:
            cur = conn.cursor()
            for file_tuple in processed_files:
                try:
                    length = len(file_tuple)

                    # Handle tuples with 5, 6 OR 7 values
                    if length == 5:
                        filesequenceid, storename, filename, local_path, s3_key = file_tuple
                        storeid = None
                        subcategory_id = None

                    elif length == 6:
                        filesequenceid, storename, filename, local_path, s3_key, storeid = file_tuple
                        subcategory_id = None

                    elif length == 7:
                        filesequenceid, storename, filename, local_path, s3_key, storeid, subcategory_id = file_tuple

                    else:
                        logger.error(
                            f"Invalid tuple length: {length} "
                            f"for file: {file_tuple}"
                        )
                        failed_updates.append(file_tuple)
                        continue

                    # UPDATE using ONLY filesequenceid (correct approach)
                    cur.execute(
                        """
                        UPDATE orgi.fileupload 
                        SET processed_flag = 'Y' 
                        WHERE filesequenceid = %s
                        """,
                        (filesequenceid,)
                    )

                    logger.info(
                        f"Updated processed_flag for filesequenceid={filesequenceid} "
                        f"(storeid={storeid}, subcategory={subcategory_id})"
                    )

                except Exception as e:
                    logger.error(f"Failed to update processed_flag for {file_tuple}: {e}")
                    failed_updates.append(file_tuple)

            conn.commit()

        except Exception as e:
            logger.error(f"Failed to update processed_flag globally: {e}")

            # return safe tuple parts only
            for f in processed_files:
                if len(f) >= 3:
                    failed_updates.append((f[0], f[1], f[2]))

        finally:
            if 'cur' in locals():
                cur.close()

        return failed_updates
