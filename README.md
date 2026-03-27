# Corrosion Detection — Instance Segmentation Pipeline
**XIS AI Department — Technical Assessment Submission**

---

## Overview

An end-to-end instance segmentation pipeline that detects and delineates corroded regions in images using **Mask R-CNN** (ResNet-50-FPN backbone) fine-tuned on a public corrosion dataset in COCO format.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | Corrosion Detection v1i |
| **Source** | [Roboflow Universe](https://universe.roboflow.com/) |
| **Format** | COCO Segmentation |
| **Train images** | 17,198 |
| **Validation images** | 1,844 |
| **Test images** | 922 (annotated) |
| **Original categories** | 21 (remapped to binary: background / corrosion) |

---

## Model

| Property | Value |
|---|---|
| **Architecture** | Mask R-CNN |
| **Backbone** | ResNet-50 + FPN |
| **Pre-trained weights** | COCO (torchvision `DEFAULT`) |
| **Fine-tuned classes** | 2 (background + corrosion) |
| **Framework** | PyTorch / torchvision |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 12 (4 frozen backbone + 8 full fine-tune) |
| Batch size | 4 |
| Learning rate | 0.005 → 0.0005 (StepLR ×0.5 every 4 epochs) |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Train samples used | 3,000 (subsampled for Colab speed) |
| Augmentations | Horizontal flip, vertical flip, brightness/contrast jitter |

---

## Results (Test Set)

| Metric | Score |
|---|---|
| **mAP@0.5** | **0.5797** |
| **mAP@0.5:0.95** | **0.3748** |
| **Mean IoU** | **0.8295** |
| **Precision** | **0.5933** |
| **Recall** | **0.6419** |
| **F1-Score** | **0.6166** |

Best validation loss: **0.5448** (epoch 12)

---

## Repository Structure

```
├── project.ipynb          # Full pipeline notebook (dataset → train → eval → inference)
├── XIS.pdf                # Original assessment brief
└── README.md
```

---

## How to Run

1. Upload `project.ipynb` to **Google Colab**
2. Place the dataset zip at `MyDrive/Dataset/Corrosion Detection.v1i.coco-segmentation.zip`
3. Run cells sequentially — each cell is labelled by phase:
   - **Cell 1–2**: Mount Drive & unzip dataset
   - **Cell 3–4**: Dataset inspection & visualisation
   - **Cell 5**: Model training (saves `best_model.pth`)
   - **Cell 6**: Evaluation — computes all metrics
   - **Cell 7–8**: Inference on new images

To run inference on a single new image:
```python
run_inference("path/to/image.jpg", score_threshold=0.5)
```

---

## Dependencies

```
torch, torchvision, pycocotools, Pillow, matplotlib, numpy, tqdm
```

Install with:
```bash
pip install pycocotools
```
All other dependencies are pre-installed in Google Colab.
