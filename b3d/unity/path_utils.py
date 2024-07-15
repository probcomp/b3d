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

def get_input() -> Path:
    """The absolute path of the input_data directory on current machine"""
    input_dir_path = get_shared() / 'input_data'

    if not os.path.exists(input_dir_path):
        os.makedirs(input_dir_path)
        print(f"Initialized empty directory for input data at {input_dir_path}.")

    return input_dir_path

def get_unity() -> Path:
    """The absolute path of the unity directory on current machine"""
    unity_dir_path = get_input() / 'unity'

    if not os.path.exists(unity_dir_path):
        os.makedirs(unity_dir_path)
        print(f"Initialized empty directory for unity data at {unity_dir_path}.")

    return unity_dir_path

def data_class_dir_map(data_class: str) -> str:
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
    
    return data_class_map[data_class]


def get_unity_dataclass(data_class: str) -> Path:
    
    data_class_dir = data_class_dir_map(data_class)
    data_dir_path = get_unity() / data_class_dir

    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
        print(f"Initialized empty directory for unity data at {data_dir_path}.")

    return data_dir_path

def get_assets_path(
        data_class: str, 
        scenefolder: str = None, 
        datafolder: str = None) -> Path:
    """Returns the absolute path of the assets directory for the given data on the current machine.

    Args:
        data_class (str): Short code for the data class ('f' for feature track data, 's' for segmented video input, 'v' for video input).
        scenefolder (str, optional): Scene folder within the data class directory.
        datafolder (str, optional): Specific data folder within the scene folder.

    Returns:
        Path: The absolute path to the assets directory.
    """
    assets_dir_path = get_unity_dataclass(data_class)

    if scenefolder is not None:
        assets_dir_path /= scenefolder

    if datafolder is not None:
        assets_dir_path /= datafolder

    if not assets_dir_path.exists():
        assets_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Initialized empty directory for shared bucket data at {assets_dir_path}.")

    return assets_dir_path

def find_unity_data_folder_path(
        data_name: str, 
        data_class: str= 'f', 
        scene_folder: str = None) -> Path:
    
    data_class_dir = data_class_dir_map(data_class)
    unity_path = get_unity()
    assets_dir_path = unity_path / data_class_dir

    # If scene_folder is specified, look only in that folder
    if scene_folder:
        specific_scene_path = assets_dir_path / scene_folder
        if specific_scene_path.exists() and specific_scene_path.is_dir():
            for folder in specific_scene_path.iterdir():
                if folder.is_dir() and folder.name == data_name:
                    return folder
    else:
        # Search for the data name folder inside all scene folders
        for scenefolder in assets_dir_path.iterdir():
            if scenefolder.is_dir():
                for folder in scenefolder.iterdir():
                    if folder.is_dir() and folder.name == data_name:
                        return folder

    raise FileNotFoundError(f"Data name '{data_name}' not found in any scenefolder under data class '{data_class}'.")

def find_unity_data_file_path(
        data_name: str, 
        data_class: str = 'f', 
        resolution: int = 200, 
        background: bool = True, 
        scene_folder: str = None, 
        light: bool = True) -> str:
    
    # Look for the data folder in shared_data_bucket' / 'input_data' / 'unity' / (data_class_folder / scene_folder)
    folder_path = find_unity_data_folder_path(data_name, data_class, scene_folder)

    # Construct the file name based on the specified settings
    light_setting = 'lit' if (light == True) else 'unlit'
    background_setting = 'bg' if (background == True) else 'nobg'
    file_name = f"{light_setting}_{background_setting}_{resolution}p.input.npz"
    file_path = folder_path / file_name
    
    # Check if the file exists
    if file_path.exists():
        return str(file_path)
    else:
        print(f"'{data_name}' only exists as {data_class_dir_map(data_class)} with the following settings:")
        for existing_file in folder_path.glob('*.npz'):
            if existing_file.is_file():
                print(data_settings(existing_file.name))
        raise FileNotFoundError(f"Specified file '{file_name}' not found in '{folder_path}'.")

def data_settings(filename: str):
    parts = filename.split('/')[-1].split('_')
    
    if len(parts) != 3:
        raise ValueError("Filename does not have the expected format.")

    light_setting = True if (parts[0] == 'lit') else False
    background_setting = True if (parts[1] == 'bg') else False
    resolution = parts[2].split('p.')[0]

    return f"resolution: {resolution}, background: {background_setting}, lighting: {light_setting}"