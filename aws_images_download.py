import os
import json
import boto3
import psycopg2
from botocore.exceptions import ClientError

#####################################################
# LOAD CONFIG
#####################################################

with open("config.json", "r") as f:
    config = json.load(f)

db = config["db_config"]
s3_cfg = config["s3_config"]

#####################################################
# OUTPUT DIRECTORY
#####################################################

DOWNLOAD_DIR = "downloaded_images"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

#####################################################
# CONNECT DATABASE
#####################################################

conn = psycopg2.connect(
    host=db["host"],
    port=db["port"],
    database=db["database"],
    user=db["user"],
    password=db["password"]
)

cursor = conn.cursor()

#####################################################
# FETCH 50 RECORDS
#####################################################

query = """
SELECT
    filesequenceid,
    filename,
    s3_image_path
FROM orgi.fileupload
WHERE subcategory_id = 605
  AND storeid IN (3110111722,2110112061,2110110358)
  AND uploadtimestamp >= '2026-07-01'
  AND uploadtimestamp < '2026-08-01';
"""

cursor.execute(query)

rows = cursor.fetchall()

print(f"Found {len(rows)} images.\n")

#####################################################
# CREATE S3 CLIENT
#####################################################

s3 = boto3.client(
    "s3",
    aws_access_key_id=s3_cfg["access_key"],
    aws_secret_access_key=s3_cfg["secret_key"],
    region_name=s3_cfg["region"]
)

bucket = s3_cfg["bucket_name"]

#####################################################
# DOWNLOAD IMAGES
#####################################################

downloaded = 0
failed = 0

for idx, (fileid, filename, s3_path) in enumerate(rows, start=1):

    print(f"[{idx}/{len(rows)}] {filename}")

    local_path = os.path.join(DOWNLOAD_DIR, filename)

    try:

        s3.download_file(
            bucket,
            s3_path,
            local_path
        )

        print("   ✓ Downloaded")

        downloaded += 1

    except ClientError as e:

        print("   ✗ Failed")
        print(e)

        failed += 1

#####################################################
# SUMMARY
#####################################################

cursor.close()
conn.close()

print("\n==============================")
print("DOWNLOAD SUMMARY")
print("==============================")
print(f"Downloaded : {downloaded}")
print(f"Failed     : {failed}")
print("==============================")