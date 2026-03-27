import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as TF


# ── CONFIG ────────────────────────────────────────────────────
MODEL_PATH      = "best_model.pth"   # path to saved weights
IMAGE_PATH      = None               # single image  (used if FOLDER is None)
FOLDER          = "test"             # run on all images in this folder
SCORE_THRESHOLD = 0.5                # confidence cutoff
SAVE_OUTPUT     = True               # saves each result to  outputs/
# ─────────────────────────────────────────────────────────────


def build_model(num_classes=2):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features_box  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found → {model_path}")
        sys.exit(1)
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def run_inference(image_path, model, device, score_threshold=0.5, save_output=True):
    if not os.path.exists(image_path):
        print(f"  SKIP: file not found → {image_path}")
        return

    img    = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    img_t  = TF.to_tensor(img).to(device)

    with torch.no_grad():
        output = model([img_t])[0]

    scores = output["scores"].cpu().numpy()
    masks  = output["masks"].cpu().numpy()
    boxes  = output["boxes"].cpu().numpy()

    keep   = scores >= score_threshold
    scores = scores[keep]
    masks  = masks[keep]
    boxes  = boxes[keep]

    print(f"  {os.path.basename(image_path):<55} → {len(scores)} instance(s) detected")

    # ── Build overlay ─────────────────────────────────────────
    MASK_COLOR = np.array([0.2, 0.85, 0.2])
    overlay    = img_np.copy().astype(float)
    for mask in masks:
        binary = mask[0] > 0.5
        overlay[binary] = overlay[binary] * 0.35 + MASK_COLOR * 255 * 0.65
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # ── Plot ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.imshow(img_np)
    ax1.set_title("Input Image", fontsize=12)
    ax1.axis("off")

    ax2.imshow(overlay)
    ax2.set_title(f"Corrosion Detected: {len(scores)} instance(s)", fontsize=12)
    ax2.axis("off")

    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = b
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax2.add_patch(rect)
        ax2.text(x1, max(y1 - 4, 0), f"{s:.2f}",
                 color="lime", fontsize=9, fontweight="bold")

    legend = [mpatches.Patch(color=[0.2, 0.85, 0.2], label="Predicted Corrosion Mask")]
    fig.legend(handles=legend, loc="lower center", fontsize=11)
    plt.suptitle(
        f"Mask R-CNN — {os.path.basename(image_path)}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if save_output:
        os.makedirs("outputs", exist_ok=True)
        out_name = "outputs/" + os.path.splitext(os.path.basename(image_path))[0] + "_result.jpg"
        plt.savefig(out_name, dpi=120, bbox_inches="tight")

    plt.show()
    plt.close()


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    # Command-line overrides:
    #   python run_inference.py                        → runs on test/ folder
    #   python run_inference.py myimage.jpg            → single image
    #   python run_inference.py test/                  → whole folder

    arg = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"\n  Device : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Loading model: {MODEL_PATH}")
    model, device = load_model(MODEL_PATH)
    print("  Model ready.\n")

    if arg and os.path.isfile(arg):
        # Single image passed as argument
        run_inference(arg, model, device, SCORE_THRESHOLD, SAVE_OUTPUT)

    else:
        # Folder mode — process every .jpg / .png in the folder
        folder = arg if (arg and os.path.isdir(arg)) else FOLDER
        exts   = (".jpg", ".jpeg", ".png")
        images = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

        if not images:
            print(f"  No images found in: {folder}")
            sys.exit(1)

        print(f"  Processing {len(images)} images from '{folder}/'")
        print(f"  Results will be saved to 'outputs/'\n")
        print(f"  {'Image':<55}   Detections")
        print(f"  {'-'*65}")

        for fname in images:
            run_inference(
                os.path.join(folder, fname),
                model, device, SCORE_THRESHOLD, SAVE_OUTPUT,
            )

        print(f"\n  Done. All results saved → outputs/")
