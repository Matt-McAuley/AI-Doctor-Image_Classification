import os
import shutil
from tqdm import tqdm

# ======================
# CONFIGURATION
# ======================
base_dir = './data'  # parent folder containing domain subdirectories
domains = ['Blood_Cancer', 'Bone_Fracture', 'Brain_MRI', 'Breast_Cancer', 'Chest_Xray']

# where to store merged dataset
output_dir = './data/All_Domains'
train_out = os.path.join(output_dir, 'Training')
test_out = os.path.join(output_dir, 'Testing')

# create output dirs
for d in [train_out, test_out]:
    os.makedirs(d, exist_ok=True)

# ======================
# MERGE LOOP
# ======================
for domain in domains:
    print(f'Processing domain: {domain}')

    domain_train = os.path.join(base_dir, domain, 'Training')
    domain_test = os.path.join(base_dir, domain, 'Testing')

    # Target folders for this domain in merged dataset
    merged_train_class = os.path.join(train_out, domain)
    merged_test_class = os.path.join(test_out, domain)

    os.makedirs(merged_train_class, exist_ok=True)
    os.makedirs(merged_test_class, exist_ok=True)

    # Copy all train images
    for root, _, files in os.walk(domain_train):
        for f in tqdm(files, desc=f'Train: {domain}', leave=False):
            src = os.path.join(root, f)
            dst = os.path.join(merged_train_class, f)
            shutil.copy2(src, dst)

    # Copy all test images
    for root, _, files in os.walk(domain_test):
        for f in tqdm(files, desc=f'Test: {domain}', leave=False):
            src = os.path.join(root, f)
            dst = os.path.join(merged_test_class, f)
            shutil.copy2(src, dst)

print('\nMerge complete.')
print(f'Merged dataset created at: {output_dir}')