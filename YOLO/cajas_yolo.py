import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO

model_path = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob\runs\detect\train9\weights\best.pt"

images_dir = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob\dataset_yolo\images\test"

labels_dir = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob\dataset_yolo\labels\test"

results_dir = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob\yolo_test_results"

os.makedirs(results_dir, exist_ok=True)

classes = [
    "dolphin",
    "penguin",
    "rabbit"
]

model = YOLO(model_path)

print("Modelo cargado")


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = ((box1[2] - box1[0]) *(box1[3] - box1[1]))
    area2 = ((box2[2] - box2[0]) *(box2[3] - box2[1]))
    union = area1 + area2 - inter

    if union == 0:
        return 0

    return inter / union

TP = 0
FP = 0
FN = 0

y_true = []
y_pred = []


images = [
    x for x in os.listdir(images_dir)
    if x.endswith(".jpg")
]

print(f"Imágenes encontradas: {len(images)}")

for idx, image_name in enumerate(images):

    print(f"[{idx+1}/{len(images)}] {image_name}")

    image_path = os.path.join(images_dir,image_name)
    label_path = os.path.join(labels_dir,image_name.replace(".jpg", ".txt"))
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    gt_boxes = []
    gt_labels = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        cls, x_center, y_center, bw, bh = map(float,line.strip().split())

        xmin = (x_center - bw / 2) * w
        ymin = (y_center - bh / 2) * h
        xmax = (x_center + bw / 2) * w
        ymax = (y_center + bh / 2) * h

        gt_boxes.append([xmin,ymin,xmax,ymax])
        gt_labels.append(classes[int(cls)])

    results = model(image_path)

    pred_boxes = []
    pred_labels = []

    for r in results:

        for box in r.boxes:

            coords = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = coords
            cls = int(box.cls[0])
            pred_boxes.append([xmin,ymin,xmax,ymax])
            pred_labels.append(classes[cls])


    matched_gt = set()

    for pred_box, pred_label in zip(pred_boxes,pred_labels):

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):

            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(pred_box,gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if (best_iou >= 0.5 and best_gt_idx != -1 and pred_label == gt_labels[best_gt_idx]):

            TP += 1
            matched_gt.add(best_gt_idx)
            y_true.append(gt_labels[best_gt_idx])
            y_pred.append(pred_label)

        else:

            FP += 1
            y_true.append("background")
            y_pred.append(pred_label)


    for gt_idx, gt_label in enumerate(gt_labels):

        if gt_idx not in matched_gt:
            FN += 1
            y_true.append(gt_label)
            y_pred.append("background")

    fig, ax = plt.subplots(figsize=(12,12))

    ax.imshow(image_rgb)

    for box, label in zip(gt_boxes,gt_labels):

        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="green",
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            xmin,
            ymin - 10,
            f"GT: {label}",
            color="white",
            fontsize=10,
            bbox=dict(
                facecolor="green",
                alpha=0.7
            )
        )

    for box, label in zip(pred_boxes,pred_labels):

        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="red",
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            xmin,
            ymax + 10,
            f"PRED: {label}",
            color="white",
            fontsize=10,
            bbox=dict(
                facecolor="red",
                alpha=0.7
            )
        )
    plt.axis("off")
    plt.title(image_name)
    plt.savefig(os.path.join(results_dir,image_name),bbox_inches="tight")
    plt.close()

precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)

f1 = (2 * precision * recall) / (precision + recall + 1e-6)

print()
print("========= RESULTS =========")
print()

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


metrics_df = pd.DataFrame({
    "Precision": [precision],
    "Recall": [recall],
    "F1-score": [f1],

    "TP": [TP],
    "FP": [FP],
    "FN": [FN]
})

metrics_df.to_csv(
    os.path.join(
        results_dir,
        "metrics.csv"
    ),
    index=False
)
