import os
import shutil
import random
from collections import defaultdict

def organize_breast_cancer_data(
    root_dir,
    output_dir="./data",
    train_ratio=0.8,
    seed=42,
    total_sample_size=20000
):
    """
    Combines nested patient folders (e.g. Breast_Cancer/8863/0/, Breast_Cancer/8863/1/)
    into unified class folders 'non_idc/' and 'idc/', but only takes a random subset
    (~20k total images, evenly balanced between classes) distributed across patients.

    Args:
        root_dir (str): Path to the main Breast_Cancer directory.
        output_dir (str): Where to save Training and Testing sets.
        train_ratio (float): Fraction of data for training.
        seed (int): Random seed.
        total_sample_size (int): Total number of images to include in final dataset.
    """

    random.seed(seed)
    class_map = {"0": "non_idc", "1": "idc"}
    combined_dir = os.path.join(output_dir, "Combined")
    os.makedirs(combined_dir, exist_ok=True)

    # Create destination dirs
    for class_name in class_map.values():
        os.makedirs(os.path.join(combined_dir, class_name), exist_ok=True)

    print("🔄 Gathering patient image paths...")
    patient_images = defaultdict(lambda: {"0": [], "1": []})

    # Collect all image paths grouped by patient and label
    for patient_folder in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        for label in class_map.keys():
            src_dir = os.path.join(patient_path, label)
            if not os.path.isdir(src_dir):
                continue

            for img_file in os.listdir(src_dir):
                img_path = os.path.join(src_dir, img_file)
                if os.path.isfile(img_path):
                    patient_images[patient_folder][label].append(img_path)

    # Calculate per-class target counts
    per_class_target = total_sample_size // 2  # e.g., 10k idc, 10k non-idc
    sampled_images = {"0": [], "1": []}

    print(f"🎯 Sampling {per_class_target} images per class, balanced across patients...")

    for label in ["0", "1"]:
        # Flatten all patients’ image lists but keep distribution even
        all_images = []
        for patient, images in patient_images.items():
            if images[label]:
                all_images.append((patient, images[label]))

        # Calculate fair per-patient allocation
        per_patient_quota = max(1, per_class_target // len(all_images))

        for patient, imgs in all_images:
            random.shuffle(imgs)
            sampled_images[label].extend(imgs[:per_patient_quota])

        # If still short, randomly fill up to target
        if len(sampled_images[label]) < per_class_target:
            remaining_needed = per_class_target - len(sampled_images[label])
            extra_pool = [img for _, imgs in all_images for img in imgs]
            extra_choices = random.sample(extra_pool, min(remaining_needed, len(extra_pool)))
            sampled_images[label].extend(extra_choices)

        # Trim excess in case of rounding
        sampled_images[label] = random.sample(sampled_images[label], per_class_target)

    # Copy sampled images into Combined folder
    print("📦 Copying sampled images into Combined folder...")
    for label, class_name in class_map.items():
        dest_dir = os.path.join(combined_dir, class_name)
        for src_path in sampled_images[label]:
            patient_id = os.path.basename(os.path.dirname(os.path.dirname(src_path)))
            img_file = os.path.basename(src_path)
            new_name = f"{patient_id}_{label}_{img_file}"
            shutil.copy2(src_path, os.path.join(dest_dir, new_name))

    # Split into Training and Testing
    print("\n🔍 Splitting into Training and Testing sets...")
    train_dir = os.path.join(output_dir, "Training")
    test_dir = os.path.join(output_dir, "Testing")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in class_map.values():
        src_dir = os.path.join(combined_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        all_imgs = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        random.shuffle(all_imgs)

        split_idx = int(len(all_imgs) * train_ratio)
        train_imgs = all_imgs[:split_idx]
        test_imgs = all_imgs[split_idx:]

        for img in train_imgs:
            shutil.copy2(os.path.join(src_dir, img), os.path.join(train_class_dir, img))
        for img in test_imgs:
            shutil.copy2(os.path.join(src_dir, img), os.path.join(test_class_dir, img))

        print(f"{class_name}: {len(train_imgs)} train, {len(test_imgs)} test")

    print(f"\n✅ Done. Sampled dataset (~{total_sample_size} images) created at: {output_dir}")

# Example usage
organize_breast_cancer_data(
    "./data/Breast_Cancer_1",
    output_dir="./data/Breast_Cancer",
    train_ratio=0.8,
    total_sample_size=20000
)