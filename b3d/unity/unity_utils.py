from pathlib import Path
import os
import b3d


def get_unity() -> Path:
    """Return the absolute path of the Unity directory on the current machine."""
    unity_dir_path = b3d.get_shared_large() / "unity"

    if not os.path.exists(unity_dir_path):
        os.makedirs(unity_dir_path)
        print(f"Initialized empty directory for unity data at {unity_dir_path}.")

    return unity_dir_path


def map_data_class(data_class: str) -> str:
    """Map a short name code to the full data class name."""
    data_class_map = {
        "f": "feature_track_data",
        "feature_track_data": "feature_track_data",
        "s": "segmented_video_input",
        "segmented_video_input": "segmented_video_input",
        "v": "video_input",
        "video_input": "video_input",
        "a": "additional_data",
        "additional_data": "additional_data",
    }
    if data_class not in data_class_map:
        raise ValueError(
            f"Invalid data class '{data_class}'. Expected one of {list(data_class_map.keys())}."
        )
    return data_class_map[data_class]


def get_existing_data_folder_path(
    data_name: str, data_class: str = "f", scene_folder: str = None
) -> Path:
    unity_folder = get_unity()
    data_class_dir = map_data_class(data_class)

    if scene_folder:
        data_path = unity_folder / scene_folder / data_name / data_class_dir
        if data_path.exists() and data_path.is_dir():
            return data_path
        raise FileNotFoundError(
            f"Data '{data_name}' folder not found in specified scene folder '{scene_folder}' with data class '{data_class_dir}'."
        )
    else:
        # Walk through all the directories and subdirectories in the 'unity' folder
        for root, dirs, files in os.walk(unity_folder):
            for dir_name in dirs:
                if dir_name == data_name:
                    data_path = Path(root) / dir_name / data_class_dir
                    if data_path.exists() and data_path.is_dir():
                        return data_path
        raise FileNotFoundError(
            f"Data '{data_name}' folder not found in any scene folder with data class '{data_class_dir}'."
        )


def get_data_path(
    data_name: str,
    data_class: str = "f",
    resolution: int = 200,
    background: bool = True,
    light: bool = True,
    scene_folder: str = None,
) -> str:
    """Return the file path of the Unity data file based on specified settings."""
    # Look for the data folder in 'large_data_bucket' / 'unity' / 'scene_folder' / 'data_name' / 'data_class'
    folder_path = get_existing_data_folder_path(data_name, data_class, scene_folder)
    light_setting = "lit" if light else "unlit"
    background_setting = "bg" if background else "nobg"
    file_name = f"{light_setting}_{background_setting}_{resolution}p.input.npz"
    file_path = folder_path / file_name
    if file_path.exists():
        return str(file_path)
    else:
        print(
            f"'{data_name}' only exists as {map_data_class(data_class)} with the following settings:"
        )
        for existing_file in folder_path.glob("*.npz"):
            if existing_file.is_file():
                print_data_settings(existing_file.name)
        raise FileNotFoundError(
            f"Specified file '{file_name}' not found in '{folder_path}'."
        )


def print_data_settings(filename: str):
    """Print data settings based on the filename."""
    settings = filename.split("/")[-1].split("_")
    if len(settings) != 3:
        raise ValueError("Filename does not have the expected format.")
    light_setting = settings[0] == "lit"
    background_setting = settings[1] == "bg"
    resolution = settings[2].split("p.")[0]
    print(
        f"resolution: {resolution}, background: {background_setting}, lighting: {light_setting}"
    )


def extract_file_info(file_path: str) -> dict:
    """Extract and return file information from the given file path."""
    folder_parts = file_path.split("/")
    scene_folder = folder_parts[-4]
    base_name = folder_parts[-3]
    data_class = folder_parts[-2]
    settings = folder_parts[-1].split("_")
    light_setting = settings[0] == "lit"
    background_setting = settings[1] == "bg"
    resolution = settings[2].split("p.")[0]
    return {
        "scene_folder": scene_folder,
        "data_name": base_name,
        "light_setting": light_setting,
        "background_setting": background_setting,
        "resolution": resolution,
        "data_class": data_class,
    }
