import os
import random

project_path = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob"
dataset_path = os.path.join(project_path,"Proyecto","ssd_dataset")
images_path = os.path.join(dataset_path, "images")
split_path = os.path.join(project_path,"Proyecto","ssd_splits")

os.makedirs(split_path, exist_ok=True)

images = [
    x for x in os.listdir(images_path)
    if x.endswith(".jpg")
]

videos = {}

for img in images:
    name = img.replace(".jpg", "")

    video_id = "_".join(name.split("_")[:-1])

    if video_id not in videos:
        videos[video_id] = []

    videos[video_id].append(name)

train_ids = []
val_ids = []
test_ids = []

random.seed(42)

classes = ["dolphin", "penguin", "rabbit"]

for cls in classes:

    cls_videos = [
        v for v in videos.keys()
        if v.startswith(cls)
    ]
    cls_videos = sorted(cls_videos)
    random.shuffle(cls_videos)

    train_videos = cls_videos[:2]
    val_videos = cls_videos[2:4]
    test_videos = cls_videos[4:]

    for v in train_videos:
        train_ids.extend(videos[v])

    for v in val_videos:
        val_ids.extend(videos[v])

    for v in test_videos:
        test_ids.extend(videos[v])

with open(os.path.join(split_path, "train_ids.txt"),"w") as f:
    for x in train_ids:
        f.write(x + "\n")

with open(os.path.join(split_path, "val_ids.txt"),"w") as f:
    for x in val_ids:
        f.write(x + "\n")

with open(os.path.join(split_path, "test_ids.txt"),"w") as f:
    for x in test_ids:
        f.write(x + "\n")

print()

print("Splits created")
