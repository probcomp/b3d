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

def get_assets_path(
        data_class: str, 
        scenefolder: str = None, 
        basefolder: str = None) -> Path:
    """Returns the absolute path of the assets directory for the given data on the current machine.

    Args:
        data_class (str): Short code for the data class ('f' for feature track data, 's' for segmented video input, 'v' for video input).
        scenefolder (str, optional): Specific folder within the data class directory.
        basefolder (str, optional): Specific base folder within the scene folder.

    Returns:
        Path: The absolute path to the assets directory.
    """
    data_class_map = {
        'f': 'feature_track_data',
        'feature_track_data': 'feature_track_data',
        's': 'segmented_video_input',
        'segmented_video_input': 'segmented_video_input',
        'v': 'video_input',
        'video_input': 'video_input',
    }

    if data_class not in data_class_map:
        raise ValueError(f"Invalid data class '{data_class}'. Expected one of {list(data_class_map.keys())}.")

    data_class_dir = data_class_map[data_class]
    root_path = get_project_root()
    assets_dir_path = root_path / 'assets' / 'shared_data_bucket' / 'input_data' / 'unity' / data_class_dir

    if scenefolder is not None:
        assets_dir_path /= scenefolder

    if basefolder is not None:
        assets_dir_path /= basefolder

    if not assets_dir_path.exists():
        assets_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Initialized empty directory for shared bucket data at {assets_dir_path}.")

    return assets_dir_path

def find_unity_data_folder_path(
        base_name: str, 
        data_class: str= 'f', 
        scene_folder: str = None) -> Path:
    """Searches for the unity data base name folder inside the data class folder within the scene folders."""
    data_class_map = {
        'f': 'feature_track_data',
        's': 'segmented_video_input',
        'v': 'video_input',
        # Add other data classes here as needed
    }

    if data_class not in data_class_map:
        raise ValueError(f"Invalid data class '{data_class}'. Expected one of {list(data_class_map.keys())}.")

    data_class_dir = data_class_map[data_class]
    root_path = get_project_root()
    assets_dir_path = root_path / 'assets' / 'shared_data_bucket' / 'input_data' / 'unity' / data_class_dir

    # If scene_folder is specified, look only in that folder
    if scene_folder:
        specific_scene_path = assets_dir_path / scene_folder
        if specific_scene_path.exists() and specific_scene_path.is_dir():
            for folder in specific_scene_path.iterdir():
                if folder.is_dir() and folder.name == base_name:
                    return folder
    else:
        # Search for the base name folder inside all scene folders
        for scenefolder in assets_dir_path.iterdir():
            if scenefolder.is_dir():
                for folder in scenefolder.iterdir():
                    if folder.is_dir() and folder.name == base_name:
                        return folder

    raise FileNotFoundError(f"Base name '{base_name}' not found in any scenefolder under data class '{data_class}'.")

def find_unity_data_file_path(
        base_name: str, 
        data_type: str = 'f', 
        resolution: int = 200, 
        background: bool = True, 
        scene_folder: str = None, 
        light: bool = True) -> str:
    
    # Look for the data folder in shared_data_bucket' / 'input_data' / 'unity'
    assets_path = find_unity_data_folder_path(base_name, data_type, scene_folder)

    # Construct the file name based on the specified settings
    light_setting = 'lit' if (light == True) else 'unlit'
    background_setting = 'bg' if (background == True) else 'nobg'
    file_name = f"{light_setting}_{background_setting}_{resolution}p.input.npz"
    file_path = assets_path / file_name
    
    # Check if the file exists
    if file_path.exists():
        return str(file_path)
    else:
        print(f"Specified file '{file_name}' does not exist in '{assets_path}'. Existing files:")
        for existing_file in assets_path.glob('*.npz'):
            if existing_file.is_file():
                print(base_name + "/" + existing_file.name)
        raise FileNotFoundError(f"Specified file '{file_name}' not found in '{assets_path}'.")
