import os
import shutil
import random

def split_dataset(root_dir, output_dir="./data", train_ratio=0.8, seed=42):
    """
    Splits dataset in `root_dir` (organized by class folders) into 
    Training and Testing sets inside `output_dir/Training` and `output_dir/Testing`.
    
    Args:
        root_dir (str): Path to dataset folder with subfolders as class names.
        output_dir (str): Base directory for new Training/Testing folders.
        train_ratio (float): Fraction of images to include in training set.
        seed (int): Random seed for reproducibility.
    """

    random.seed(seed)
    train_dir = os.path.join(output_dir, "Training")
    test_dir = os.path.join(output_dir, "Testing")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Loop through each class folder
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # skip non-folder items

        # Create matching subfolders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # List all images in class
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)

        # Split
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        # Copy to destination
        for img in train_imgs:
            shutil.copy2(os.path.join(class_path, img),
                         os.path.join(train_dir, class_name, img))
        for img in test_imgs:
            shutil.copy2(os.path.join(class_path, img),
                         os.path.join(test_dir, class_name, img))

        print(f"{class_name}: {len(train_imgs)} train, {len(test_imgs)} test")

    print(f"\n✅ Dataset split complete. Training -> {train_dir}, Testing -> {test_dir}")

# Example usage
split_dataset("./data/Blood_Cancer_1", output_dir="./data/Blood_Cancer", train_ratio=0.8)