import os
import sys
import time

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

from torchvision import transforms

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    average_precision_score
)

base_path = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob"
os.chdir(base_path)

utils_path = os.path.join(base_path, "Proyecto", "ssd-utils")
sys.path.append(utils_path)

from model import SSD300

data_path = os.path.join(base_path, "Proyecto", "ssd_dataset")
image_path = os.path.join(data_path, "images")

results_path = os.path.join(base_path,"Proyecto","ssd_test_results")
os.makedirs(results_path, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

label2target = {
    "dolphin": 1,
    "penguin": 2,
    "rabbit": 3,
    "background": 0
}

target2label = {
    1: "dolphin",
    2: "penguin",
    3: "rabbit"
}

valid_classes = [
    "dolphin",
    "penguin",
    "rabbit"
]

df = pd.read_csv(os.path.join(data_path, "df.csv"))
split_dir = os.path.join(base_path,"Proyecto","ssd_splits")
with open(os.path.join(split_dir, "test_ids.txt")) as f:
    test_ids = [
        x.strip()
        for x in f.readlines()
    ]

test_df = df[df["ImageID"].isin(test_ids)]

model = SSD300(4, device)
model.load_state_dict(torch.load(os.path.join(base_path,"Proyecto","best_ssd_model.pth"),map_location=device))
model = model.to(device)
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = ((box1[2] - box1[0]) *(box1[3] - box1[1]))
    box2_area = ((box2[2] - box2[0]) *(box2[3] - box2[1]))
    union_area = (box1_area +box2_area -inter_area)

    if union_area == 0:
        return 0

    return inter_area / union_area

TP = 0
FP = 0
FN = 0

y_true = []
y_pred = []

class_tp = {c: 0 for c in valid_classes}
class_fp = {c: 0 for c in valid_classes}
class_fn = {c: 0 for c in valid_classes}

all_scores = []
all_binary_true = []

print()
print("Start evaluation")

unique_images = test_df.ImageID.unique()

for idx, image_id in enumerate(unique_images):
    img_path = os.path.join(image_path,image_id + ".jpg")
    original_img = Image.open(img_path).convert("RGB")

    img = np.array(original_img.resize((300, 300))) / 255.
    img = torch.tensor(img).permute(2, 0, 1)
    img = normalize(img)
    img = img.float().unsqueeze(0).to(device)

    with torch.no_grad():

        predicted_locs, predicted_scores = model(img)

        det_boxes, det_labels, det_scores = model.detect_objects(
            predicted_locs,
            predicted_scores,
            min_score=0.2,
            max_overlap=0.5,
            top_k=10
        )


    det_boxes = det_boxes[0].cpu().numpy()
    det_labels = det_labels[0].cpu().numpy()
    det_scores = det_scores[0].cpu().numpy()

    gt = test_df[test_df["ImageID"] == image_id]

    gt_boxes = []
    gt_labels = []

    for _, row in gt.iterrows():

        xmin = row["XMin"]
        ymin = row["YMin"]
        xmax = row["XMax"]
        ymax = row["YMax"]

        gt_boxes.append([xmin,ymin,xmax,ymax])

        gt_labels.append(row["LabelName"])

    matched_gt = set()

    for pred_box, pred_label, pred_score in zip(det_boxes,det_labels,det_scores):

        pred_label = int(pred_label)
        if pred_label == 0:
            continue

        pred_class = target2label[pred_label]

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):

            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(pred_box,gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if (best_iou >= 0.5 and best_gt_idx != -1 and gt_labels[best_gt_idx] == pred_class):
            TP += 1
            class_tp[pred_class] += 1

            matched_gt.add(best_gt_idx)
            y_true.append(gt_labels[best_gt_idx])
            y_pred.append(pred_class)

            all_binary_true.append(1)
            all_scores.append(pred_score)

        else:

            FP += 1
            class_fp[pred_class] += 1

            y_true.append("background")
            y_pred.append(pred_class)

            all_binary_true.append(0)
            all_scores.append(pred_score)

    for gt_idx, gt_label in enumerate(gt_labels):

        if gt_idx not in matched_gt:

            FN += 1
            class_fn[gt_label] += 1
            y_true.append(gt_label)
            y_pred.append("background")

    fig, ax = plt.subplots(figsize=(10,10))

    ax.imshow(original_img)

    w_img, h_img = original_img.size

    for box, label, score in zip(det_boxes,det_labels,det_scores):
        label = int(label)
        if label == 0:
            continue
        xmin, ymin, xmax, ymax = box
        xmin *= w_img
        xmax *= w_img
        ymin *= h_img
        ymax *= h_img

        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            linewidth=2,
            edgecolor="red"
        )

        ax.add_patch(rect)

        class_name = target2label.get(label,"unknown")

        ax.text(
            xmin,
            ymin - 5,
            f"PRED {class_name}: {score:.2f}",
            fontsize=10,
            color="white",
            bbox=dict(
                facecolor="red",
                alpha=0.7
            )
        )

    for gt_box, gt_label in zip(gt_boxes,gt_labels):

        xmin, ymin, xmax, ymax = gt_box

        xmin *= w_img
        xmax *= w_img
        ymin *= h_img
        ymax *= h_img

        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            linewidth=2,
            edgecolor="green"
        )

        ax.add_patch(rect)
        ax.text(
            xmin,
            ymax + 10,
            f"GT {gt_label}",
            fontsize=10,
            color="white",
            bbox=dict(
                facecolor="green",
                alpha=0.7
            )
        )

    ax.axis("off")
    plt.title(image_id)
    plt.savefig(
        os.path.join(
            results_path,
            f"{image_id}.png"
        ),
        bbox_inches="tight"
    )
    plt.close()
    print(
        f"[{idx+1}/{len(unique_images)}] Saved: {image_id}"
    )

precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = (2 * precision * recall) / (precision + recall + 1e-6)

print()
print("========= GLOBAL RESULTS =========")
print()

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


report = classification_report(
    y_true,
    y_pred,
    labels=valid_classes,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(results_path,"classification_report.csv"))

print()
print("========= PER CLASS RESULTS =========")
print()

print(report_df)

cm_labels = valid_classes + ["background"]
cm = confusion_matrix(y_true,y_pred,labels=cm_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=cm_labels)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
plt.title("SSD Confusion Matrix")
plt.savefig(os.path.join(results_path,"confusion_matrix.png"))
plt.close()

precision_curve, recall_curve, _ = precision_recall_curve(all_binary_true,all_scores)

ap = average_precision_score(all_binary_true,all_scores)

plt.figure(figsize=(8,6))
plt.plot(recall_curve,precision_curve,linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve | AP = {ap:.4f}")
plt.grid()
plt.savefig(os.path.join(results_path,"pr_curve.png"))
plt.close()

metrics_df = pd.DataFrame({
    "TP": [TP],
    "FP": [FP],
    "FN": [FN],

    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
})

metrics_df.to_csv(os.path.join(results_path,"metrics.csv"),index=False)

print()
print("Evaluation finished")