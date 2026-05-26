import os
import cv2

root = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob\dataset"
output = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob\dataset_yolo"

splits = ["train", "val", "test"]

for s in splits:
    os.makedirs(f"{output}/images/{s}", exist_ok=True)
    os.makedirs(f"{output}/labels/{s}", exist_ok=True)

classes = sorted(os.listdir(root))

for class_id, class_name in enumerate(classes):

    class_path = os.path.join(root, class_name)

    video_folder = os.path.join(class_path, "videos")
    label_folder = os.path.join(class_path, "gt")

    videos = sorted([v for v in os.listdir(video_folder) if v.endswith(".mp4")])

    split_dict = {
        "train": videos[:2],
        "val": videos[2:4],
        "test": videos[4:]
    }

    for split in split_dict:

        for video in split_dict[split]:

            video_path = os.path.join(video_folder, video)
            label_path = os.path.join(label_folder, video.replace(".mp4", "_gt.txt"))

            cap = cv2.VideoCapture(video_path)

            annotations = {}

            with open(label_path) as f:
                for line in f:

                    frame_id, target, x, y, w, h, conf, cls, vis = line.strip().split(",")

                    frame_id = int(frame_id)

                    if frame_id not in annotations:
                        annotations[frame_id] = []

                    annotations[frame_id].append((float(x), float(y), float(w), float(h)))

            frame_number = 1

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                img_h, img_w = frame.shape[:2]

                name = f"{class_name}_{video[:-4]}_{frame_number}"

                img_out = f"{output}/images/{split}/{name}.jpg"
                label_out = f"{output}/labels/{split}/{name}.txt"

                cv2.imwrite(img_out, frame)

                if frame_number in annotations:

                    with open(label_out, "w") as f:

                        for x, y, w, h in annotations[frame_number]:

                            x_center = (x + w/2) / img_w
                            y_center = (y + h/2) / img_h
                            w = w / img_w
                            h = h / img_h

                            f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

                frame_number += 1

            cap.release()

print("Dataset creado correctamente")