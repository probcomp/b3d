import os
import json
from pathlib import Path
import shutil
import b3d

def get_tags(data_name):
    tags_dict = {
        1: 'dynamic camera',
        2: 'dynamic objects',
        3: 'panning',
        4: 'deformable objects',
        5: 'self-occlusion',
        6: 'occlusion',
    }
    
    print(f"Available tags for {data_name}:")
    for key, value in tags_dict.items():
        print(f"{key}: {value}")
    
    selected_tags = input(f"Enter the numbers corresponding to the tags for {data_name}, separated by commas: ").strip()
    tags = [tags_dict[int(num)] for num in selected_tags.split(",") if int(num) in tags_dict]
    
    return tags

def get_additional_data(data_name):
    options = {
        1: 'None',
        2: 'mesh'
    }
    
    print(f"Available additional data for {data_name}:")
    for key, value in options.items():
        print(f"{key}: {value}")
    
    selected_data = input(f"Enter the number corresponding to the additional data for {data_name}: ").strip()
    additional_data = options.get(int(selected_data), 'None')
    
    return [] if additional_data == 'None' else [additional_data]

def consolidate_metadata(root):
    root_path = Path(root)
    for scene_folder in root_path.iterdir():
        if scene_folder.is_dir():
            for data_folder in scene_folder.iterdir():
                if data_folder.is_dir():
                    combined_metadata = {"tags": [], "additional_data": []}
                    
                    for data_class_folder in data_folder.iterdir():
                        if data_class_folder.is_dir():
                            metadata_file = data_class_folder / "metadata.json"
                            if metadata_file.exists():
                                with metadata_file.open('r') as f:
                                    metadata = json.load(f)
                                    combined_metadata["tags"].extend(metadata.get("tags", []))
                                    combined_metadata["additional_data"].extend(metadata.get("additional_data", []))
                                
                                # Remove the metadata file from the data_class folder
                                metadata_file.unlink()
                    
                    # Request user input for tags and additional data
                    data_name = data_folder.name
                    tags = get_tags(data_name)
                    additional_data = get_additional_data(data_name)

                    # Combine user input with existing metadata
                    combined_metadata["tags"].extend(tags)
                    combined_metadata["additional_data"].extend(additional_data)

                    # Remove duplicates from combined_metadata
                    combined_metadata["tags"] = list(set(combined_metadata["tags"]))
                    combined_metadata["additional_data"] = list(set(combined_metadata["additional_data"]))

                    # Save the combined metadata at the data_folder level
                    new_metadata_file = data_folder / "metadata.json"
                    with new_metadata_file.open('w') as f:
                        json.dump(combined_metadata, f, indent=4)

                    print(f"Consolidated metadata for {data_folder.name}")

if __name__ == "__main__":
    root_dir = str(b3d.get_shared_large() / 'unity').strip()
    consolidate_metadata(root_dir)
    print("Metadata consolidation complete.")
