import json

tags_dict = {
    1: "dynamic camera",
    2: "dynamic objects",
    3: "panning",
    4: "self-occlusion",
    5: "occlusion",
    6: "deformable objects",
}


def print_tags():
    print("Available tags:")
    for key, value in tags_dict.items():
        print(f"{key}: {value}")


def request_tags(data_name):
    print_tags()
    selected_tags = input(
        f"Enter the numbers corresponding to the tags for {data_name}, separated by commas: "
    ).strip()
    if selected_tags == "":
        return []
    tags = get_tags(selected_tags)
    return tags


def get_tags(selected_tags):
    tags = [
        tags_dict[int(num)] for num in selected_tags.split(",") if int(num) in tags_dict
    ]
    return tags


def add_tags(metadata_file, data_name: str = None):
    combined_metadata = {"tags": []}
    with metadata_file.open("r") as f:
        metadata = json.load(f)
        combined_metadata["tags"].extend(metadata.get("tags", []))
        print(f"Current tags: {combined_metadata}")

        additional_tags = request_tags(data_name)
        combined_metadata["tags"].extend(additional_tags)
        # Remove duplicates from combined_metadata
        combined_metadata["tags"] = list(set(combined_metadata["tags"]))

    # Save the combined metadata at the data_folder level
    with metadata_file.open("w") as f:
        json.dump(combined_metadata, f, indent=4)
        print(f"Edited tags: {combined_metadata}")


def init_metadata(metadata_file, tags):
    metadata = {"tags": get_tags(tags)}

    # Save the combined metadata at the data_folder level
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=4)
        print(f"Init tags: {metadata}")
