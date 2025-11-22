import os
import boto3
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .config import BUCKET, PREFIX, LOCAL_DIR, CLIENT_CONFIG

def list_s3_files():
    """Lists files in the S3 bucket."""
    s3 = boto3.client('s3', config=CLIENT_CONFIG)
    print("Listing files...")
    paginator = s3.get_paginator('list_objects_v2')
    tasks = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('/'): continue

                local_path = os.path.join(LOCAL_DIR, key.replace(PREFIX, ""))
                tasks.append((key, local_path))
    
    print(f"Total files found: {len(tasks)}")
    return tasks

def download_task(task):
    """Downloads a single file if it doesn't exist."""
    key, path = task
    s3 = boto3.client('s3', config=CLIENT_CONFIG)
    # Only download if file is missing
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s3.download_file(BUCKET, key, path)

def download_files_parallel(tasks):
    """Downloads files in parallel using ThreadPoolExecutor."""
    print(f"Downloading {len(tasks)} images...")
    with ThreadPoolExecutor(max_workers=20) as pool:
        list(tqdm(pool.map(download_task, tasks), total=len(tasks)))

    # Verify and Retry Missing Files
    print("Verifying files...")
    missing_tasks = [t for t in tasks if not os.path.exists(t[1])]

    if len(missing_tasks) > 0:
        print(f"Retrying {len(missing_tasks)} missing files...")
        # Simple sequential retry for any failures
        for task in tqdm(missing_tasks):
            download_task(task)

    print(f"Verified. Total files on disk: {len([t for t in tasks if os.path.exists(t[1])])}")

