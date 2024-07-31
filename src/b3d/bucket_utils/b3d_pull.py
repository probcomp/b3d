# gcloud storage cp --recursive gs://hgps_data_bucket/shared .
import argparse
import os
import subprocess

import b3d

## Paths.
GCLOUD_BUCKET_NAME = b3d.get_gcloud_bucket_ref()
LOCAL_BUCKET_PATH = str(
    b3d.get_shared()
)  # returns path to the data subdirectory in dcolmap/assets
print(f"B3D Downloading data from {GCLOUD_BUCKET_NAME} to {LOCAL_BUCKET_PATH}.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ow",
        "--overwrite",
        action="store_true",
        help="Overwrite local copy of data with bucket data",
        default=False,
    )
    parser.add_argument(
        "-fn",
        "--filename",
        type=str,
        help="Path of specific file to download, relative to the bucket root.\
                         If not specified, defaults to downloading all bucket data.",
        default="",
    )

    opt = parser.parse_args()

    return opt


def download_all(overwrite):
    """
    Download all contents from the bucket.
    """
    if overwrite:
        confirm = input(
            "WARNING:This will overwrite any existing local copy of the specified data with the version on the GCP bucket.\nContinue? [y/[n]]: "
        )
        if confirm != "y":
            print("Aborting.")
            return -1

    print("===============================================")
    print(
        f"Downloading all contents from {GCLOUD_BUCKET_NAME} into {LOCAL_BUCKET_PATH}...\n(overwrite={overwrite})"
    )
    print("===============================================")

    flags = [] if overwrite else ["-u"]
    download_cmd = (
        ["gsutil", "-m", "rsync", "-r", "-x", ".*\\.gstmp$"]
        + flags
        + [
            GCLOUD_BUCKET_NAME,
            LOCAL_BUCKET_PATH,
        ]
    )

    popen = subprocess.Popen(
        download_cmd, stdout=subprocess.PIPE, universal_newlines=True
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        print(stdout_line)
    popen.stdout.close()
    return_code = popen.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, download_cmd)

    return 0


def download_item(overwrite, filename):
    """
    Download a specified item (file or directory) from the bucket.
    """
    # clean trailing slash to prevent wrong local nesting
    while filename[-1] == "/":
        filename = filename[:-1]

    if overwrite:
        confirm = input(
            "WARNING:This will overwrite any existing local copy of the specified data with the version on the GCP bucket.\nContinue? [y/[n]]: "
        )
        if confirm != "y":
            print("Aborting.")
            return -1

    print("===============================================")
    print(
        f"Downloading {filename} from {GCLOUD_BUCKET_NAME} into {LOCAL_BUCKET_PATH + '/' + filename}...\n(overwrite={overwrite})"
    )
    print("===============================================")

    # download a single item
    if "." in filename:
        if overwrite:
            download_cmd = [
                "gcloud",
                "storage",
                "cp",
                "--recursive",
                GCLOUD_BUCKET_NAME + "/" + filename,
                LOCAL_BUCKET_PATH + "/" + filename,
            ]
        else:
            download_cmd = [
                "gcloud",
                "storage",
                "cp",
                "-n",
                "--recursive",
                GCLOUD_BUCKET_NAME + "/" + filename,
                LOCAL_BUCKET_PATH + "/" + filename,
            ]

    # download a directory (preserve directory structures)
    else:
        # preprocessing
        if filename.rfind("/") != -1:  # nested directory
            parent_dir = filename[: filename.rfind("/")]
            if not os.path.exists(LOCAL_BUCKET_PATH + "/" + parent_dir):
                os.makedirs(LOCAL_BUCKET_PATH + "/" + parent_dir)
        else:  # directory at root of bucket
            parent_dir = ""

        if overwrite:
            download_cmd = [
                "gcloud",
                "storage",
                "cp",
                "--recursive",
                GCLOUD_BUCKET_NAME + "/" + filename,
                LOCAL_BUCKET_PATH + "/" + parent_dir,
            ]
        else:
            download_cmd = [
                "gcloud",
                "storage",
                "cp",
                "-n",
                "--recursive",
                GCLOUD_BUCKET_NAME + "/" + filename,
                LOCAL_BUCKET_PATH + "/" + parent_dir,
            ]

    popen = subprocess.Popen(
        download_cmd, stdout=subprocess.PIPE, universal_newlines=True
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        print(stdout_line)
    popen.stdout.close()
    return_code = popen.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, download_cmd)

    return 0


def download_from_bucket():
    opts = parse_args()
    overwrite = opts.overwrite
    filename = opts.filename

    if filename == "":
        # download the whole bucket
        return_code = download_all(overwrite)
    else:
        # download the specified item in bucket
        return_code = download_item(overwrite, filename)

    if return_code == 0:
        print(
            f"\nSuccess: Downloaded all contents from https://console.cloud.google.com/storage/browser/{GCLOUD_BUCKET_NAME.split('://')[-1]}."
        )


if __name__ == "__main__":
    download_from_bucket()
