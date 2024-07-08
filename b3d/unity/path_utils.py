from pathlib import Path
import os
import b3d


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path().resolve().parents[1]

def get_gcloud_bucket_ref()-> str: return "gs://hgps_data_bucket"

def get_root() -> Path: return Path(Path(b3d.__file__).parents[1])

def get_assets() -> Path:
    """The absolute path of the assets directory on current machine"""
    assets_dir_path = get_root() / 'assets'

    if not os.path.exists(assets_dir_path):
        os.makedirs(assets_dir_path)
        print(f"Initialized empty directory for shared bucket data at {assets_dir_path}.")

    return assets_dir_path

def get_shared() -> Path:
    """The absolute path of the assets directory on current machine"""
    data_dir_path = get_assets() / 'shared_data_bucket'

    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
        print(f"Initialized empty directory for shared bucket data at {data_dir_path}.")

    return data_dir_path

def get_assets_path(data_class: str, scenefolder: str = None, basefolder: str = None) -> str:
    """Returns the absolute path of the assets directory for the given data class on the current machine.

    Args:
        data_class (str): Short code for the data class ('k' for keypoints, 'v' for videoinput, 'd' for deformableMeshes).
        folder (str, optional): Specific folder within the data class directory.

    Returns:
        Path: The absolute path to the assets directory.
    """
    data_class_map = {
        'f': 'feature_track_data',
        'v': 'videoinput',
    }

    if data_class not in data_class_map:
        raise ValueError(f"Invalid data class '{data_class}'. Expected one of {list(data_class_map.keys())}.")

    data_class_dir = data_class_map[data_class]
    root_path = get_project_root()
    assets_dir_path = root_path / 'assets' / 'shared_data_bucket' / 'input_data' / 'unity' / data_class_dir

    if scenefolder is not None:
        assets_dir_path = assets_dir_path / scenefolder

    if basefolder is not None:
        assets_dir_path = assets_dir_path / basefolder

    if not assets_dir_path.exists():
        assets_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Initialized empty directory for shared bucket data at {assets_dir_path}.")

    return str(assets_dir_path) + '/'