import os
import json
import shutil
from pathlib import Path
import b3d

# Default tags
default_tags = [
    "dynamic camera",
    "dynamic objects",
    "panning",
    "deformable objects"
]

# Default additional data options
default_additional_data = ["mesh"]

def get_tags():
    print("\nAvailable tags:")
    for i, tag in enumerate(default_tags, start=1):
        print(f"{i}. {tag}")
    
    selected_tags = []
    while True:
        choice = input("Select a tag by number (or type 'new' to add a new tag, 'done' to finish): ").strip()
        if choice.lower() == 'done':
            break
        elif choice.lower() == 'new':
            new_tag = input("Enter the new tag: ").strip()
            if new_tag and new_tag not in default_tags:
                default_tags.append(new_tag)
                selected_tags.append(new_tag)
        elif choice.isdigit() and 1 <= int(choice) <= len(default_tags):
            selected_tags.append(default_tags[int(choice) - 1])
        else:
            print("Invalid choice. Please try again.")
    return selected_tags

def get_additional_data():
    print("\nAvailable additional data options:")
    for i, data in enumerate(default_additional_data, start=1):
        print(f"{i}. {data}")
    
    choice = input("Select an additional data option by number: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(default_additional_data):
        return default_additional_data[int(choice) - 1]
    else:
        print("Invalid choice. Defaulting to 'Nothing'.")
        return "Nothing"

def reorganize_folders(root):
    root_path = Path(root)
    for data_class_folder in root_path.iterdir():
        if data_class_folder.is_dir():
            for scene_folder in data_class_folder.iterdir():
                if scene_folder.is_dir():
                    for data_folder in scene_folder.iterdir():
                        if data_folder.is_dir():
                            new_scene_folder = root_path / scene_folder.name
                            new_data_class_folder = new_scene_folder / data_folder.name / data_class_folder.name
                            new_data_class_folder.mkdir(parents=True, exist_ok=True)

                            # Move all files from the original data folder to the new location
                            for item in data_folder.iterdir():
                                shutil.move(str(item), str(new_data_class_folder / item.name))
                            
                            # Remove the original (now empty) data folder
                            data_folder.rmdir()

                            print(f"Moved {data_folder.name}")
                            
                            # Add metadata.json
                            metadata = {
                                "tags": get_tags(),
                                "additional_data": get_additional_data()
                            }
                            metadata_file = new_data_class_folder / "metadata.json"
                            with metadata_file.open('w') as f:
                                json.dump(metadata, f, indent=4)
                                
if __name__ == "__main__":
    root_dir = str(b3d.get_shared_large() / 'unity').strip()
    reorganize_folders(root_dir)
    print("Reorganization complete.")
