import os
import sys

import pandas as pd
import numpy as np

import torch

import matplotlib.pyplot as plt

from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms

base_path = r"C:\Users\aadri\OneDrive\Escritorio\rec_ob"
os.chdir(base_path)
utils_path = os.path.join(base_path,"Proyecto","ssd-utils")
sys.path.append(utils_path)

from torch_snippets import *
from model import SSD300, MultiBoxLoss

data_path = os.path.join(base_path,"Proyecto","ssd_dataset")
image_path = os.path.join(data_path,"images")
df = pd.read_csv(os.path.join(data_path, "df.csv"))
split_path = os.path.join(base_path,"Proyecto","ssd_splits")

with open(os.path.join(split_path, "train_ids.txt")) as f:
    train_ids = [
        x.strip()
        for x in f.readlines()
    ]

with open(os.path.join(split_path, "val_ids.txt")) as f:
    val_ids = [
        x.strip()
        for x in f.readlines()
    ]

train_df = df[df["ImageID"].isin(train_ids)]
val_df = df[df["ImageID"].isin(val_ids)]

label2target = {
    "dolphin": 1,
    "penguin": 2,
    "rabbit": 3,
    "background": 0
}

target2label = {t:l for l,t in label2target.items()}
num_classes = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)
    return img.float()

class OpenDataset(torch.utils.data.Dataset):
    w = 300
    h = 300

    def __init__(self, df, image_dir=image_path):

        self.df = df
        self.image_dir = image_dir
        self.image_infos = df.ImageID.unique()
        print(f"{len(self)} images loaded")

    def __getitem__(self, ix):

        image_id = self.image_infos[ix]
        img_path = os.path.join(self.image_dir,image_id + ".jpg")

        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((self.w, self.h),resample=Image.BILINEAR)) / 255.

        data = self.df[self.df["ImageID"] == image_id]

        labels = data["LabelName"].values.tolist()
        boxes = data[["XMin","YMin","XMax","YMax"]].values.astype(np.float32)

        boxes[:, [0,2]] *= self.w
        boxes[:, [1,3]] *= self.h

        clean_boxes = []
        clean_labels = []

        for box, label in zip(boxes, labels):

            xmin, ymin, xmax, ymax = box
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(self.w - 1, xmax)
            ymax = min(self.h - 1, ymax)

            if xmax <= xmin:
                continue
            if ymax <= ymin:
                continue
            if np.isnan([xmin, ymin, xmax, ymax]).any():
                continue

            clean_boxes.append([xmin,ymin,xmax,ymax])
            clean_labels.append(label)

        boxes = clean_boxes
        labels = clean_labels

        if len(boxes) == 0:
            return self.__getitem__((ix + 1) % len(self))

        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax = boxes[i]
                xmin_new = self.w - xmax
                xmax_new = self.w - xmin
                boxes[i] = [xmin_new,ymin,xmax_new,ymax]

        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.9, 1.1)
            img = np.clip(img * factor,0,1)

        return img, boxes, labels

    def collate_fn(self, batch):
        images = []
        boxes = []
        labels = []

        for item in batch:
            img, image_boxes, image_labels = item
            img = preprocess_image(img)[None]

            images.append(img)
            boxes.append(torch.tensor(image_boxes).float().to(device) / 300.)
            labels.append(torch.tensor([label2target[c] for c in image_labels]).long().to(device))

        images = torch.cat(images).to(device)
        return images, boxes, labels

    def __len__(self):
        return len(self.image_infos)

train_ds = OpenDataset(train_df)
val_ds = OpenDataset(val_df)


train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    collate_fn=train_ds.collate_fn,
    drop_last=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    collate_fn=val_ds.collate_fn,
    drop_last=True
)

def train_batch(inputs,model,criterion,optimizer):

    model.train()
    images, boxes, labels = inputs
    predicted_locs, predicted_scores = model(images)

    loss = criterion(predicted_locs,predicted_scores,boxes,labels)
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def validate_batch(inputs,model,criterion):

    model.eval()
    images, boxes, labels = inputs
    predicted_locs, predicted_scores = model(images)
    loss = criterion(predicted_locs,predicted_scores,boxes,labels)

    return loss.item()

n_epochs = 8

model = SSD300(num_classes,device)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr=5e-6,weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1
)

criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy,device=device)

train_history = []
val_history = []
lr_history = []

best_val_loss = float("inf")

patience = 3
counter = 0

print()
print("Start training")

for epoch in range(n_epochs):

    train_losses = []

    for inputs in train_loader:
        loss = train_batch(inputs,model,criterion,optimizer)
        train_losses.append(loss)

    avg_train_loss = np.mean(train_losses)

    val_losses = []

    for inputs in val_loader:
        loss = validate_batch(inputs,model,criterion)
        val_losses.append(loss)

    avg_val_loss = np.mean(val_losses)

    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    train_history.append(avg_train_loss)
    val_history.append(avg_val_loss)
    lr_history.append(current_lr)

    print(
        f"Epoch {epoch+1}/{n_epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
    )


    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(),os.path.join(base_path,"Proyecto","best_ssd_model.pth"))
        print("Best model saved")
    else:
        counter += 1

    if counter >= patience:
        print()
        print("Early stopping")
        break

torch.save(
    model.state_dict(),
    os.path.join(
        base_path,
        "Proyecto",
        "ssd_model.pth"
    )
)

results_path = os.path.join(base_path,"Proyecto","ssd_results")
os.makedirs(results_path, exist_ok=True)

plt.figure(figsize=(10,5))
plt.plot(train_history,label="Train Loss")
plt.plot(val_history,label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SSD Loss Curves")
plt.legend()
plt.grid()
plt.savefig(os.path.join(results_path,"loss_curves.png"))
plt.close()

plt.figure(figsize=(10,5))
plt.plot(lr_history,label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid()
plt.savefig(os.path.join(results_path,"lr_curve.png"))
plt.close()

metrics_df = pd.DataFrame({
    "epoch": range(1,len(train_history) + 1),
    "train_loss": train_history,
    "val_loss": val_history,
    "learning_rate": lr_history})

metrics_df.to_csv(os.path.join(results_path,"metrics.csv"),index=False)
print()
print("Training finished")