import boto3
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------- CONFIG --------
AWS_ACCESS_KEY = "AKIA6D6JBNORLYOWHTZZ"
AWS_SECRET_KEY = "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"
REGION = "ap-south-1"
BUCKET_NAME = "imageinsightimagesbeverages"
STORE_FOLDER = "STORE_IMAGES"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

MAX_WORKERS = 20
# ------------------------

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)


def list_root_images():
    paginator = s3.get_paginator("list_objects_v2")
    images = []

    for page in paginator.paginate(Bucket=BUCKET_NAME):
        for obj in page.get("Contents", []):

            key = obj["Key"]

            # only root files
            if "/" in key:
                continue

            if key.lower().endswith(IMAGE_EXTENSIONS):
                images.append(key)

    return images


def fix_store_image(img):

    store_key = f"{STORE_FOLDER}/{img}"
    content_type = mimetypes.guess_type(img)[0]

    if not content_type:
        return None

    try:
        s3.copy_object(
            Bucket=BUCKET_NAME,
            CopySource={"Bucket": BUCKET_NAME, "Key": store_key},
            Key=store_key,
            MetadataDirective="REPLACE",
            ContentType=content_type,
            ContentDisposition="inline"
        )

        return store_key

    except Exception:
        return None


def run_pipeline():

    root_images = list_root_images()

    print(f"\nRoot images found: {len(root_images)}\n")

    processed = 0

    with ThreadPoolExecutor(MAX_WORKERS) as executor:

        futures = [executor.submit(fix_store_image, img) for img in root_images]

        for future in as_completed(futures):

            result = future.result()

            if result:
                processed += 1
                print(f"Fixed: {result}")

    print(f"\nFinished fixing {processed} images.")


if __name__ == "__main__":
    run_pipeline()



