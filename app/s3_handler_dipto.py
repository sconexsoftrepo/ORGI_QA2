import os
import boto3
import re
import mimetypes
from botocore.exceptions import ClientError, NoCredentialsError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class S3Handler:
    def __init__(self, s3_config, db_config):
        self.bucket_name = s3_config['bucket_name']
        self.image_folder_s3 = s3_config['image_folder_s3']
        self.db_config = db_config
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=s3_config['access_key'],
                aws_secret_access_key=s3_config['secret_key'],
                region_name=s3_config['region']
            )
            logger.info(f"Initialized S3 client for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

    def sanitize_filename(self, filename):
        """Remove invalid characters, Roboflow suffixes, and directory prefixes from filename."""
        # Remove directory prefix (e.g., Testing-Images/) and keep only the base filename
        base_filename = os.path.basename(filename)
        # Remove Roboflow suffix (e.g., .rf.<hash>)
        base_filename = re.sub(r'\.rf\.[0-9a-fA-F]+$', '', base_filename)
        # Remove invalid characters for filenames
        base_filename = re.sub(r'[\\:*?"<>|]', '', base_filename)
        return base_filename

    def download_images_from_s3(self, temp_dir):
        """Download images from S3 (from the specified image_folder_s3) to the provided temporary folder."""
        try:
            from app.db_handler import initialize_db_connection, close_db_connection
            conn, cur = initialize_db_connection(self.db_config)
            # Modified query to include processed_flag = '0.0' and remove strict filename prefix requirement
            cur.execute(
                """
                SELECT filesequenceid, storename, filename, storeid, subcategory_id
                FROM orgi.fileupload
                WHERE 
                (
                    processed_flag IN ('N', '0.0') 
                    OR processed_flag IS NULL
                );
                """
            )
            image_data = cur.fetchall()
            close_db_connection(conn, cur)
            logger.info(f"Fetched {len(image_data)} unprocessed images from orgi.fileupload")

            image_paths = []
            failed_files = []
            for filesequenceid, storename, filename, storeid, subcategory_id in image_data:
                try:
                    logger.debug(f"Processing filesequenceid: {filesequenceid}, storename: {storename}, filename: {filename}")
                    clean_filename = self.sanitize_filename(filename)
                    local_path = os.path.join(temp_dir, clean_filename)
                    base_filename = os.path.basename(filename)
                    clean_storename = re.sub(r'[\\:*?"<>|]', '', storename.replace(":", ".").strip())
                    possible_s3_keys = [
                        f"{self.image_folder_s3}{clean_storename}/{base_filename}",  # Subfolder: Testing-Images/storename/filename
                        f"{self.image_folder_s3}{base_filename}",  # Root: Testing-Images/filename
                        base_filename,  # Root: filename (in case database stores only base filename)
                    ]

                    downloaded = False
                    s3_key_used = None
                    for s3_key in possible_s3_keys:
                        try:
                            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
                            logger.info(f"Downloaded {s3_key} to {local_path}")
                            image_paths.append((filesequenceid, storename, clean_filename, local_path, s3_key, storeid, subcategory_id))
                            downloaded = True
                            s3_key_used = s3_key
                            break  # Stop trying other keys after a successful download
                        except ClientError as e:
                            if e.response['Error']['Code'] == '404':
                                logger.debug(f"File not found in S3: {s3_key}")
                                continue  # Try the next key
                            else:
                                logger.error(f"Failed to download {s3_key}: {e}")
                                failed_files.append((filesequenceid, storename, filename))
                                break  # Non-404 error, skip to next file

                    if not downloaded:
                        logger.warning(f"File not found in any S3 location for {filename}: tried {possible_s3_keys}")
                        failed_files.append((filesequenceid, storename, filename))

                except OSError as e:
                    logger.error(f"Invalid filename {filename}: {e}")
                    failed_files.append((filesequenceid, storename, filename))
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error downloading {filename}: {e}")
                    failed_files.append((filesequenceid, storename, filename))
                    continue

            if failed_files:
                logger.info(f"Failed to download {len(failed_files)} files: {[f[2] for f in failed_files]}")
            return image_paths, failed_files
        except Exception as e:
            logger.error(f"Failed to fetch image paths: {e}")
            raise

    def upload_file_to_s3(self, file_path, s3_key):
        """Upload a file to S3 bucket with correct content type for web viewing."""
        try:
            content_type, _ = mimetypes.guess_type(file_path)
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            logger.info(
                f"Uploaded {file_path} to S3: {s3_key} "
                f"(ContentType={content_type})"
            )
        except NoCredentialsError:
            logger.error("Invalid AWS credentials provided.")
            raise
        except ClientError as e:
            logger.error(f"Failed to upload {file_path} to S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {e}")
            raise
