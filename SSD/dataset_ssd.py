import os
import shutil
import pandas as pd

base_path = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob\dataset_yolo"

image_folders = [
    "train",
    "val",
    "test"
]

classes = [
    "dolphin",
    "penguin",
    "rabbit"
]

ssd_path = "ssd_dataset"
ssd_images = os.path.join(ssd_path, "images")

os.makedirs(ssd_images, exist_ok=True)

rows = []

for split in image_folders:

    images_dir = os.path.join(base_path, "images", split)
    labels_dir = os.path.join(base_path, "labels", split)

    for label_file in os.listdir(labels_dir):

        image_id = label_file.replace(".txt", "")
        txt_path = os.path.join(labels_dir, label_file)
        image_extensions = [".jpg"]
        image_path = None

        for ext in image_extensions:
            candidate = os.path.join(images_dir, image_id + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        shutil.copy(image_path, ssd_images)

        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            cls, x_center, y_center, width, height = map(float,line.strip().split())

            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2

            rows.append({
                "ImageID": image_id,
                "LabelName": classes[int(cls)],
                "XMin": xmin,
                "YMin": ymin,
                "XMax": xmax,
                "YMax": ymax
            })


df = pd.DataFrame(rows)

os.makedirs(ssd_path, exist_ok=True)

csv_path = os.path.join(ssd_path, "df.csv")

df.to_csv(csv_path, index=False)

print()
print("SSD dataset terminado")