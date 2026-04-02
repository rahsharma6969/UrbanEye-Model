# =============================================
#  LEVIR-CD Training with U-Net (Deep Learning)
#  Recommended for best accuracy
# =============================================

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

# ================= CONFIG =================
DATA_ROOT = r"C:\Users\Rahul\CollegeProject\UrbanModel\data\LEVIR_CD"  # ← your path

IMG_SIZE    = 256
BATCH_SIZE  = 8
EPOCHS      = 60
LR          = 0.0005   # lower LR for stability
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_PATH = "best_unet_levir_cd.pth"

# ================= DATASET =================
class LEVIRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.t1_dir = os.path.join(root_dir, "A")
        self.t2_dir = os.path.join(root_dir, "B")
        self.label_dir = os.path.join(root_dir, "label")
        self.file_list = sorted([f for f in os.listdir(self.t1_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        t1 = cv2.cvtColor(cv2.imread(os.path.join(self.t1_dir, fname)), cv2.COLOR_BGR2RGB)
        t2 = cv2.cvtColor(cv2.imread(os.path.join(self.t2_dir, fname)), cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.label_dir, fname), cv2.IMREAD_GRAYSCALE)
        image = np.concatenate([t1, t2], axis=2)  # 6 channels
        label = (label > 0).astype(np.float32)
        if self.transform:
            aug = self.transform(image=image, mask=label)
            image, label = aug['image'], aug['mask']
        return image, label.unsqueeze(0)

# ================= TRANSFORMS =================
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=[0.485]*6, std=[0.229]*6),  # 6 channels
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485]*6, std=[0.229]*6),
    ToTensorV2()
])

# ================= LOADERS =================
train_ds = LEVIRDataset(os.path.join(DATA_ROOT, "train"), train_transform)
val_ds   = LEVIRDataset(os.path.join(DATA_ROOT, "val"),   val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ================= MODEL =================
model = smp.Unet(
    encoder_name="efficientnet-b4",      # strong & efficient backbone
    encoder_weights="imagenet",
    in_channels=6,                       # T1 RGB + T2 RGB
    classes=1,
    activation=None
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ================= TRAINING LOOP =================
best_f1 = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    preds, gts = [], []
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
            pred = (torch.sigmoid(outputs) > 0.5).float()
            preds.append(pred.cpu().numpy().flatten())
            gts.append(masks.cpu().numpy().flatten())
    val_loss /= len(val_loader)
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    f1 = f1_score(gts, preds)

    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {f1:.4f}")

    scheduler.step()
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  → Saved best model (F1: {f1:.4f})")

print(f"Done! Best F1: {best_f1:.4f}")



 