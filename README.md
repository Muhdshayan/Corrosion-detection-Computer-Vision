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
| Hardware | Google Colab (NVIDIA GPU) |

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
├── run_inference.py       # Standalone local inference script
├── best_model.pth         # Trained model weights (tracked via Git LFS)
├── requirements.txt       # Python dependencies
├── Documentation.pdf      # Full technical report
└── README.md
```

---

## Local Inference — Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Muhdshayan/Corrosion-detection-Computer-Vision.git
cd Corrosion-detection-Computer-Vision
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run inference

**On a single image:**
```bash
python run_inference.py path/to/your/image.jpg
```

**On a whole folder of images:**
```bash
python run_inference.py                  # defaults to test/ folder
python run_inference.py my_images/       # or specify any folder
```

Results are saved automatically to the `outputs/` folder as side-by-side comparison images (original | predicted mask overlay).

### Example output

```
  Device : CPU
  Loading model: best_model.pth
  Model ready.

  Processing 50 images from 'test/'
  Results will be saved to 'outputs/'

  Image                                           Detections
  ----------------------------------------------------------
  Corrosion2317...jpg            →  4 instance(s) detected
  Corrosion2335...jpg            →  2 instance(s) detected
  GEN_103...jpg                  →  2 instance(s) detected
  ...

  Done. All results saved → outputs/
```

---

## Google Colab — Full Pipeline

1. Upload `project.ipynb` to **Google Colab**
2. Place the dataset zip at `MyDrive/Dataset/Corrosion Detection.v1i.coco-segmentation.zip`
3. Run cells sequentially:
   - **Cell 1–2**: Mount Drive & unzip dataset
   - **Cell 3–4**: Dataset inspection & visualisation
   - **Cell 5**: Model training (saves `best_model.pth`)
   - **Cell 6**: Evaluation — computes all 6 metrics
   - **Cell 7–8**: Inference on new images

---

## Dependencies

```
torch>=2.1.0
torchvision>=0.16.0
Pillow>=9.5.0
numpy>=1.24.0
pycocotools>=2.0.7
matplotlib>=3.7.0
tqdm>=4.66.0
```

Install all with:
```bash
pip install -r requirements.txt
```
