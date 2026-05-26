import os
import re
import cv2
import numpy as np
import pandas as pd
import motmetrics as mm

from collections import defaultdict
from ultralytics import YOLO

model_pat = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob\runs\detect\train9\weights\best.pt"

test_images = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob\dataset_yolo\images\test"

test_labels = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob\dataset_yolo\labels\test"

output_dir = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob\tracking_results2"

os.makedirs(output_dir, exist_ok=True)

# TRACKER_CONFIG = "botsort.yaml"
TRACKER_CONFIG = "bytetrack.yaml"

TRACKER_NAME = TRACKER_CONFIG.replace(".yaml", "")

CLASS_NAMES = {
    0: "dolphin",
    1: "penguin",
    2: "rabbit"
}

model = YOLO(model_pat)

all_images = [
    f for f in os.listdir(test_images)
    if f.endswith((".jpg"))
]

video_groups = defaultdict(list)

for img_name in all_images:

    match = re.match(r"(.+_\d+)_(\d+)\.(jpg|png|jpeg)",img_name)

    if match:
        video_name = match.group(1)
        frame_number = int(match.group(2))
        video_groups[video_name].append((frame_number, img_name))

for video_name in video_groups:
    video_groups[video_name] = sorted(video_groups[video_name],key=lambda x: x[0])

acc = mm.MOTAccumulator(auto_id=True)

tracking_columns = [
    'frame',
    'id',
    'bb_left',
    'bb_top',
    'bb_width',
    'bb_height',
    'conf',
    'x',
    'y',
    'z'
]

tracking_df = pd.DataFrame([],columns=tracking_columns)

np.random.seed(42)

colors = {
    i: (int(np.random.randint(0,255)),int(np.random.randint(0,255)),int(np.random.randint(0,255)))
    for i in range(1000)
}

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = ((boxA[2]-boxA[0]) *(boxA[3]-boxA[1]))
    boxBArea = ((boxB[2]-boxB[0]) *(boxB[3]-boxB[1]))
    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0
    return interArea / union

global_frame = 0

for video_name, frames in video_groups.items():

    print(f"Processing video: {video_name}")
    print(f"Frames: {len(frames)}")

    model.predictor = None

    first_img = cv2.imread(os.path.join(test_images,frames[0][1]))

    height, width, _ = first_img.shape

    video_path = os.path.join(output_dir,f"{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter( video_path,fourcc,20,(width, height))

    for frame_idx, (frame_number, img_name) in enumerate(frames):

        global_frame += 1
        img_path = os.path.join(test_images,img_name)
        image = cv2.imread(img_path)
        h, w, _ = image.shape

        results = model.track(
            source=img_path,
            persist=True,
            tracker=TRACKER_CONFIG,
            verbose=False
        )[0]

        pred_boxes = []
        pred_ids = []

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for box, track_id, conf, cls in zip(boxes,ids,confs,classes):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                pred_boxes.append([x1,y1,x2,y2])

                pred_ids.append(track_id)
                color = colors[track_id % 1000]

                class_name = CLASS_NAMES.get(int(cls),"obj")

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = (
                    f"{class_name} "
                    f"ID:{track_id} "
                    f"{conf:.2f}"
                )

                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                tracking_row = [
                    global_frame,
                    track_id,
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    conf,
                    -1,
                    -1,
                    -1
                ]

                tracking_df.loc[
                    len(tracking_df)
                ] = tracking_row

        label_path = os.path.join(test_labels,os.path.splitext(img_name)[0] + ".txt")

        gt_boxes = []
        gt_ids = []

        if os.path.exists(label_path):

            with open(label_path, "r") as f:
                lines = f.readlines()
                for gt_id, line in enumerate(lines):

                    values = line.strip().split()
                    if len(values) != 5:
                        continue

                    cls, xc, yc, bw, bh = map(float,values)
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)

                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)

                    gt_boxes.append([x1,y1,x2,y2])
                    gt_ids.append(gt_id)

        distance_matrix = []

        for gt_box in gt_boxes:
            row = []
            for pred_box in pred_boxes:

                iou = compute_iou(gt_box,pred_box)
                if iou < 0.5:
                    row.append(np.nan)

                else:
                    row.append(1 - iou)

            distance_matrix.append(row)


        if len(gt_ids) > 0 and len(pred_ids) > 0:

            acc.update(gt_ids,pred_ids,distance_matrix)

        elif len(gt_ids) > 0:
            acc.update(gt_ids,[],np.empty((len(gt_ids), 0)))

        elif len(pred_ids) > 0:

            acc.update([],pred_ids,np.empty((0, len(pred_ids))))

        writer.write(image)

        print(f"{video_name} | "f"{frame_idx+1}/{len(frames)}")

    writer.release()

tracking_txt_path = os.path.join(output_dir,f"{TRACKER_NAME}_tracking.txt")

tracking_df.to_csv(tracking_txt_path,header=False,index=False)

mh = mm.metrics.create()

summary = mh.compute(
    acc,
    metrics=[
        'num_frames',
        'idf1',
        'idp',
        'idr',
        'recall',
        'precision',
        'num_objects',
        'mostly_tracked',
        'partially_tracked',
        'mostly_lost',
        'num_false_positives',
        'num_misses',
        'num_switches',
        'num_fragmentations',
        'mota',
        'motp'
    ],
    name=TRACKER_NAME
)

print(summary)

metrics_csv_path = os.path.join(output_dir,f"{TRACKER_NAME}_metrics.csv")
summary.to_csv(metrics_csv_path)