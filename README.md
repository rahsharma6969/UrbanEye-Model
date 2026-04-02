# 🏙️ Urban Change Detection using U-Net + EfficientNet-B4

## 📌 Overview

This project focuses on **detecting urban changes (e.g., building construction or demolition)** using deep learning.
We use a **U-Net architecture with an EfficientNet-B4 encoder** to perform **pixel-level change detection** on satellite images from different time periods.

The model is trained on the **LEVIR-CD dataset**, which contains bi-temporal images and corresponding change masks.

---

## 🚀 Features

* 🔍 Detects changes between two satellite images (T1 & T2)
* 🧠 Uses **EfficientNet-B4** for powerful feature extraction
* 🏗️ **U-Net decoder** for precise segmentation
* 📊 Evaluated using **F1-score**
* ⚡ Optimized with **Adam + Cosine Annealing LR Scheduler**
* 🖼️ Supports **6-channel input (T1 RGB + T2 RGB)**

---

## 🧠 Model Architecture

```
Input (T1 + T2 images → 6 channels)
        ↓
EfficientNet-B4 (Encoder)
        ↓
U-Net Decoder (Upsampling + Skip Connections)
        ↓
Output (Binary Change Mask)
```

---

## 📂 Dataset

We use the **LEVIR-CD Dataset**, which includes:

* `A/` → Image at time T1
* `B/` → Image at time T2
* `label/` → Ground truth change mask

Dataset Structure:

```
LEVIR_CD/
 ├── train/
 │   ├── A/
 │   ├── B/
 │   └── label/
 └── val/
     ├── A/
     ├── B/
     └── label/
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/urban-change-detection.git
cd urban-change-detection
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### Required Libraries

* torch
* torchvision
* segmentation-models-pytorch
* albumentations
* opencv-python
* numpy
* scikit-learn
* tqdm

---

## ▶️ Training

Update dataset path in the script:

```python
DATA_ROOT = "path_to_LEVIR_CD"
```

Run training:

```bash
python train.py
```

---

## 📊 Training Details

| Parameter     | Value             |
| ------------- | ----------------- |
| Image Size    | 256 × 256         |
| Batch Size    | 8                 |
| Epochs        | 60                |
| Learning Rate | 0.0005            |
| Loss Function | BCEWithLogitsLoss |
| Optimizer     | Adam              |
| Scheduler     | CosineAnnealingLR |

---

## 📈 Evaluation Metric

We use **F1 Score** for evaluation:

* Handles class imbalance
* Balances precision and recall
* Standard metric for change detection

---

## 💾 Model Saving

The best model (based on validation F1-score) is saved as:

```
best_unet_levir_cd.pth
```

---

## 🔬 How It Works

1. Combine T1 and T2 images → 6-channel input
2. Pass through EfficientNet encoder
3. Extract multi-scale features
4. Decode using U-Net
5. Generate pixel-wise change map

---

## 🎯 Results

* Accurate detection of:

  * 🏢 New buildings
  * 🏚️ Demolished structures
  * 🌆 Urban expansion

---

## 📌 Future Improvements

* Add attention mechanisms (Attention U-Net)
* Try Transformer-based models
* Use Dice/Focal Loss for imbalance handling
* Deploy as a web application

---

## 👨‍💻 Author

**Rahul Sharma**

---

## 📜 License

This project is for educational and research purposes.
