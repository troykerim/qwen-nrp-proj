import random
import shutil
from pathlib import Path

# Input folders
INPUT_FOLDERS = [
    "/home/troy/Downloads/Mahfuz-set.v3i.voc-augmentation/train",
    "/home/troy/Downloads/JamCausingMaterial.v3i.voc-augmentation/train",
    "/home/troy/Downloads/JamCausingMaterial.v2i.voc/train",
    "/home/troy/Downloads/JamCausingMaterial.v2i.voc/test",
    "/home/troy/Downloads/JamCausingMaterial.v2i.voc/valid",
]

# Temp folders
TEMP_IMAGES = "/home/troy/jam-causing-voc-aug-temp/images"
TEMP_LABELS = "/home/troy/jam-causing-voc-aug-temp/labels"

# Output folders
OUTPUT_PATHS = {
    "train_images": "/home/troy/jam-causing-voc-aug-nrp/train/images",
    "train_labels": "/home/troy/jam-causing-voc-aug-nrp/train/labels",
    "valid_images": "/home/troy/jam-causing-voc-aug-nrp/valid/images",
    "valid_labels": "/home/troy/jam-causing-voc-aug-nrp/valid/labels",
    "test_images": "/home/troy/jam-causing-voc-aug-nrp/test/images",
    "test_labels": "/home/troy/jam-causing-voc-aug-nrp/test/labels",
}

RANDOM_SEED = 42


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clear_folder(path):
    p = Path(path)
    if not p.exists():
        return

    for item in p.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def is_blank_xml(label_path):
    try:
        content = label_path.read_text(encoding="utf-8", errors="ignore")
        return content.strip() == ""
    except Exception as e:
        print(f"Could not read XML file {label_path}: {e}")
        return False


def find_recursive_pairs(input_folder):
    """
    Search recursively for .jpg and .xml files and match them by filename stem.
    Example:
        image123.jpg
        image123.xml
    """
    root = Path(input_folder)

    if not root.exists():
        print(f"Warning: input folder does not exist: {input_folder}")
        return []

    image_files = list(root.rglob("*.jpg"))
    xml_files = list(root.rglob("*.xml"))

    xml_map = {}
    for xml_path in xml_files:
        stem = xml_path.stem
        if stem not in xml_map:
            xml_map[stem] = xml_path
        else:
            print(f"Warning: duplicate XML stem found, keeping first: {stem}")

    pairs = []
    missing_labels = 0

    for img_path in image_files:
        stem = img_path.stem
        if stem in xml_map:
            pairs.append((img_path, xml_map[stem]))
        else:
            print(f"Missing XML label for image: {img_path}")
            missing_labels += 1

    print(f"\nScanned: {input_folder}")
    print(f"  Images found: {len(image_files)}")
    print(f"  XML labels found: {len(xml_files)}")
    print(f"  Paired: {len(pairs)}")
    print(f"  Missing labels: {missing_labels}")

    return pairs


def copy_to_temp():
    ensure_dir(TEMP_IMAGES)
    ensure_dir(TEMP_LABELS)

    clear_folder(TEMP_IMAGES)
    clear_folder(TEMP_LABELS)

    total_copied = 0

    for input_folder in INPUT_FOLDERS:
        pairs = find_recursive_pairs(input_folder)

        folder_tag = Path(input_folder).name.replace(" ", "_")
        parent_tag = Path(input_folder).parent.name.replace(" ", "_")
        dataset_tag = f"{parent_tag}_{folder_tag}"

        for img_path, xml_path in pairs:
            base_name = img_path.stem

            temp_img = Path(TEMP_IMAGES) / f"{base_name}.jpg"
            temp_xml = Path(TEMP_LABELS) / f"{base_name}.xml"

            # Handle duplicate filenames across datasets
            if temp_img.exists() or temp_xml.exists():
                new_base = f"{dataset_tag}_{base_name}"
                temp_img = Path(TEMP_IMAGES) / f"{new_base}.jpg"
                temp_xml = Path(TEMP_LABELS) / f"{new_base}.xml"

                counter = 1
                while temp_img.exists() or temp_xml.exists():
                    new_base = f"{dataset_tag}_{base_name}_{counter}"
                    temp_img = Path(TEMP_IMAGES) / f"{new_base}.jpg"
                    temp_xml = Path(TEMP_LABELS) / f"{new_base}.xml"
                    counter += 1

            shutil.copy2(img_path, temp_img)
            shutil.copy2(xml_path, temp_xml)
            total_copied += 1

    print(f"\nFinished copying to temp.")
    print(f"Total pairs copied: {total_copied}")


def remove_blank_labels_and_images():
    removed_count = 0

    for label_path in sorted(Path(TEMP_LABELS).glob("*.xml")):
        if is_blank_xml(label_path):
            image_path = Path(TEMP_IMAGES) / f"{label_path.stem}.jpg"

            print(f"Removing blank XML label: {label_path.name}")
            label_path.unlink()

            if image_path.exists():
                print(f"Removing associated image: {image_path.name}")
                image_path.unlink()

            removed_count += 1

    print(f"\nFinished removing blank XML labels.")
    print(f"Total blank XML/image pairs removed: {removed_count}")


def get_clean_pairs():
    pairs = []

    for image_path in sorted(Path(TEMP_IMAGES).glob("*.jpg")):
        label_path = Path(TEMP_LABELS) / f"{image_path.stem}.xml"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            print(f"Warning: image without XML label in temp: {image_path.name}")

    return pairs


def clear_output_folders():
    for path in OUTPUT_PATHS.values():
        ensure_dir(path)
        clear_folder(path)


def split_pairs(pairs):
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    total = len(pairs)
    train_count = int(total * 0.80)
    valid_count = int(total * 0.10)
    test_count = int(total * 0.10)

    assigned = train_count + valid_count + test_count
    remainder = total - assigned

    # Put any remainder into train
    train_count += remainder

    train_pairs = pairs[:train_count]
    valid_pairs = pairs[train_count:train_count + valid_count]
    test_pairs = pairs[train_count + valid_count:train_count + valid_count + test_count]

    return train_pairs, valid_pairs, test_pairs


def copy_pairs_to_output(pairs, image_dir, label_dir):
    for image_path, label_path in pairs:
        shutil.copy2(image_path, Path(image_dir) / image_path.name)
        shutil.copy2(label_path, Path(label_dir) / label_path.name)


def main():
    print("Step 1: Copying all images and XML labels into temp folder...")
    copy_to_temp()

    print("\nStep 2: Removing blank XML labels and associated images from temp...")
    remove_blank_labels_and_images()

    print("\nStep 3: Collecting cleaned pairs...")
    pairs = get_clean_pairs()
    print(f"Total cleaned pairs available for split: {len(pairs)}")

    if len(pairs) == 0:
        print("\nNo valid image/XML pairs were found.")
        return

    print("\nStep 4: Clearing output folders...")
    clear_output_folders()

    print("\nStep 5: Splitting into train/valid/test...")
    train_pairs, valid_pairs, test_pairs = split_pairs(pairs)

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Valid pairs: {len(valid_pairs)}")
    print(f"Test pairs:  {len(test_pairs)}")

    print("\nStep 6: Copying split data to output folders...")
    copy_pairs_to_output(train_pairs, OUTPUT_PATHS["train_images"], OUTPUT_PATHS["train_labels"])
    copy_pairs_to_output(valid_pairs, OUTPUT_PATHS["valid_images"], OUTPUT_PATHS["valid_labels"])
    copy_pairs_to_output(test_pairs, OUTPUT_PATHS["test_images"], OUTPUT_PATHS["test_labels"])

    print("\nDone.")
    print("Final dataset has been created successfully.")


if __name__ == "__main__":
    main()