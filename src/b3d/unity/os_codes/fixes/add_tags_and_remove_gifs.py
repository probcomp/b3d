import os
from pathlib import Path
import b3d
from b3d.io.feature_track_data import FeatureTrackData
from b3d.unity.generate_visualization import create_video, create_rgb_image
import json

def get_tags(data_name):
    tags_dict = {
        1: 'dynamic camera',
        2: 'dynamic objects',
        3: 'panning',
        4: 'self-occlusion',
        5: 'occlusion',
        6: 'deformable objects',
    }
    
    print(f"Available tags for {data_name}:")
    for key, value in tags_dict.items():
        print(f"{key}: {value}")
    
    selected_tags = input(f"Enter the numbers corresponding to the tags for {data_name}, separated by commas: ").strip()
    if (selected_tags == ""):
        return []
    
    tags = [tags_dict[int(num)] for num in selected_tags.split(",") if int(num) in tags_dict]
    
    return tags

def process(root_dir):
    root_path = Path(root_dir)
    for scene_folder in root_path.iterdir():
        for data_folder in scene_folder.iterdir():
            print(f"Processing {data_folder.name}")
            for data_class_content in data_folder.iterdir():
                if (os.path.isdir(data_class_content)):
                    data_class_folder = data_class_content
                    for files in data_class_folder.iterdir():
                        if files.name.endswith('.gif'):
                            print(f"removing {files}")
                            os.remove(files)
                if (data_class_content.name.endswith('.json')):
                    print(f"editing {data_class_content}")
                    metadata_file = data_class_content
                    combined_metadata = {"tags": []}
                    with metadata_file.open('r') as f:
                        metadata = json.load(f)
                        combined_metadata["tags"].extend(metadata.get("tags", []))
                        print(f"Current tags: {combined_metadata}")

                        additional_tags = get_tags(data_folder.name)
                        combined_metadata["tags"].extend(additional_tags)
                        # Remove duplicates from combined_metadata
                        combined_metadata["tags"] = list(set(combined_metadata["tags"]))

                    # Save the combined metadata at the data_folder level
                    with metadata_file.open('w') as f:
                        json.dump(combined_metadata, f, indent=4)
                        print(f"Edited tags: {combined_metadata}")
        
if __name__ == "__main__":
    root_dir = str(b3d.get_shared_large() / 'unity').strip()
    process(root_dir)