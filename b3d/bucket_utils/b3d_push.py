import argparse
import json
import os
import subprocess
import b3d
from pathlib import Path


## Paths.
GCLOUD_BUCKET_NAME = b3d.get_gcloud_bucket_ref()
LOCAL_BUCKET_PATH = str(b3d.get_shared())  # returns path to the data subdirectory in dcolmap/assets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ow', '--overwrite', action='store_true', help='Overwrite local copy of data with bucket data', default=False)
    parser.add_argument('-fn', '--filename', type=str, help='Path of specific file to download, relative to the local bucket root.\
                         If not specified, defaults to downloading all bucket data.', default='')

    opt = parser.parse_args()

    return opt

def upload_all(overwrite):
    """
    Upload all local contents to the bucket.
    """
    if overwrite:
        confirm = input("WARNING:This will overwrite any existing copy of the data on the GCP bucket with the version in your local directory.\
                        \nThis will affect ALL USERS of the shared bucket.\
                        \nContinue? [y/[n]]: ")
        if confirm != 'y':
            print("Aborting.")
            return -1
        
    print("===============================================")
    print(f"Uploading all contents from {LOCAL_BUCKET_PATH} into {GCLOUD_BUCKET_NAME}...\n(overwrite={overwrite})")
    print("===============================================")

    if overwrite:
        upload_cmd = ["gcloud", "storage", "cp", "-r", "*", GCLOUD_BUCKET_NAME]
    else:
        upload_cmd = ["gcloud", "storage", "cp", "-r", "-n", "*", GCLOUD_BUCKET_NAME]


    popen = subprocess.Popen(upload_cmd, stdout=subprocess.PIPE, universal_newlines=True, cwd=LOCAL_BUCKET_PATH)
    for stdout_line in iter(popen.stdout.readline, ""):
        print(stdout_line)
    popen.stdout.close()
    return_code = popen.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, upload_cmd)
    
    return 0


def upload_item(overwrite, filename):
    """
    Upload a specified item (file or directory) to the bucket.
    """
    # clean trailing slash to prevent wrong local nesting
    while filename[-1] == '/': filename = filename[:-1]

    if overwrite:
        confirm = input("WARNING:This will overwrite any existing copy of the data on the GCP bucket with the version in your local directory.\
                        \nThis will affect ALL USERS of the shared bucket.\
                        \nContinue? [y/[n]]: ")
        if confirm != 'y':
            print("Aborting.")
            return -1
        
    print("===============================================")
    print(f"Uploading {filename} from {LOCAL_BUCKET_PATH + '/' + filename} into {GCLOUD_BUCKET_NAME}...\n(overwrite={overwrite})")
    print("===============================================")

    # upload a single item
    if "." in filename:
        if overwrite:
            upload_cmd = ["gcloud", "storage", "cp", "-r", LOCAL_BUCKET_PATH + "/" + filename, GCLOUD_BUCKET_NAME + "/" + filename]
        else:
            upload_cmd = ["gcloud", "storage", "cp", "-r", "-n", LOCAL_BUCKET_PATH + "/" + filename, GCLOUD_BUCKET_NAME + "/" + filename]

    # upload a directory (preserve directory structures)
    else:
        if filename.rfind('/') != -1:  # nested directory   
            parent_dir = filename[:filename.rfind('/')]
            # if not os.path.exists(LOCAL_BUCKET_PATH + "/" + parent_dir):
            #     os.makedirs(LOCAL_BUCKET_PATH + "/" + parent_dir)
        else:  # directory at root of bucket
            parent_dir = ''

        if overwrite:
            upload_cmd = ["gcloud", "storage", "cp", "-r", LOCAL_BUCKET_PATH + "/" + filename, GCLOUD_BUCKET_NAME + "/" + parent_dir]
        else:
            upload_cmd = ["gcloud", "storage", "cp", "-r", "-n", LOCAL_BUCKET_PATH + "/" + filename, GCLOUD_BUCKET_NAME + "/" + parent_dir]

    popen = subprocess.Popen(upload_cmd, stdout=subprocess.PIPE, universal_newlines=True, cwd=LOCAL_BUCKET_PATH)
    for stdout_line in iter(popen.stdout.readline, ""):
        print(stdout_line)
    popen.stdout.close()
    return_code = popen.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, upload_cmd)
    
    return 0


def upload_to_bucket(overwrite=False):
    opts = parse_args()
    overwrite = opts.overwrite
    filename = opts.filename

    if filename == '':  
        # download the whole bucket
        return_code = upload_all(overwrite)
    else:
        # download the specified item in bucket
        return_code = upload_item(overwrite, filename)

    if return_code == 0:
        print(f"\nSuccess: Check updated bucket contents at https://console.cloud.google.com/storage/browser/{GCLOUD_BUCKET_NAME.split('://')[-1]}.")

if __name__ == "__main__":
    upload_to_bucket()