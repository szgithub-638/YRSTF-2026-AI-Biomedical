import os
import pandas as pd
import shutil

# Updated paths for your folder structure
csv_path = "../data/labels/ISIC_2019_Training_GroundTruth.csv"
image_folder = "../data/images"

df = pd.read_csv(csv_path)

benign_folder = os.path.join(image_folder, "benign")
malignant_folder = os.path.join(image_folder, "malignant")

os.makedirs(benign_folder, exist_ok=True)
os.makedirs(malignant_folder, exist_ok=True)

for index, row in df.iterrows():
    image_id = row["image"]
    melanoma = row["MEL"]

    src = os.path.join(image_folder, image_id + ".jpg")

    if melanoma == 1:
        dst = os.path.join(malignant_folder, image_id + ".jpg")
    else:
        dst = os.path.join(benign_folder, image_id + ".jpg")

    if os.path.exists(src):
        shutil.move(src, dst)

print("Dataset organized successfully!")