# CNN Object Detector in Python (PyTorch)

A modular, end-to-end Convolutional Neural Network (CNN) built from scratch in PyTorch designed specifically for **Object Detection** (simultaneous object classification and bounding box localization).

---

## 🏗️ How CNN Object Detection Works

Standard classification CNNs output only class probabilities for the entire image. An **Object Detection CNN** extends this by simultaneously predicting:
1. **What the object is** (Classification Head $\to$ Softmax probabilities)
2. **Where the object is located** (Bounding Box Regression Head $\to$ $[x_1, y_1, x_2, y_2]$ coordinates)

```
[ Input Image (3, 64, 64) ]
             │
             ▼
   ┌───────────────────┐
   │   CNN Backbone    │  (Conv2D + BatchNorm + ReLU + MaxPool)
   └───────────────────┘
             │ Feature Maps (128, 4, 4)
             ▼
   ┌───────────────────┐
   │ Adaptive Pooling  │  (Compresses spatial dimension to 2x2)
   └───────────────────┘
             │ Flattened Features (512-dim vector)
             ▼
   ┌───────────────────┐
   │ Shared FC Layer   │  (Linear 512 -> 128 + ReLU + Dropout)
   └───────────────────┘
             ├───┬────────────────────────────────┐
             │   │                                │
             ▼   ▼                                ▼
┌───────────────────────────┐      ┌─────────────────────────────┐
│    Classification Head    │      │ Bounding Box Regressor Head │
│ Linear(128, 64) -> C=4    │      │ Linear(128, 64) -> 4 dims   │
│ Cross-Entropy Loss        │      │ Sigmoid [0, 1] + Smooth L1  │
└───────────────────────────┘      └─────────────────────────────┘
```

---

## 📁 Project Structure

```
runaway/
├── model.py              # CNN Architecture (Backbone + Dual Detection Heads)
├── dataset.py            # Synthetic dataset generator & memory caching
├── train.py              # Multi-task training pipeline (CrossEntropy + Smooth L1)
├── predict.py            # CLI inference script & bounding box visualizer
├── utils.py              # IoU computation & Pillow bounding box drawing routines
├── main.py               # End-to-end demo (trains, evaluates & creates visual gallery)
├── outputs/              # Saved visual detection images & composite grids
│   ├── detection_grid.png
│   └── sample_prediction.png
└── detector_model.pth    # Saved PyTorch model checkpoint
```

---

## 🚀 Quickstart Guide

### 1. Run Complete End-to-End Demo
Trains the CNN detector for 12 epochs on CPU, computes validation accuracy & Mean IoU, and outputs sample detections with bounding boxes:
```bash
python main.py
```

### 2. Train Custom Epochs
```bash
python train.py
```

### 3. Run Inference on Any Image
```bash
# Run on demo sample
python predict.py

# Run on your own image file
python predict.py --image "path/to/your/image.jpg" --output "outputs/my_result.png"
```

---

## 📐 Loss Formulation & Metrics

### Multi-Task Loss Function
The network is trained end-to-end minimizing a composite multi-task objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{classification}} + \lambda_{\text{box}} \cdot \mathcal{L}_{\text{box}}$$

- **Classification Loss ($\mathcal{L}_{\text{cls}}$)**: Categorical Cross-Entropy across classes (Background, Rectangle, Circle, Triangle).
- **Bounding Box Loss ($\mathcal{L}_{\text{box}}$)**: Smooth $L_1$ loss over normalized $[x_1, y_1, x_2, y_2]$ coordinates, masked so that background samples without objects do not incur coordinate penalties.
- **$\lambda_{\text{box}}$**: Balance weight (default: $4.0$) prioritizing coordinate precision.

### Intersection over Union (IoU)
$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{B_{\text{pred}} \cap B_{\text{gt}}}{B_{\text{pred}} \cup B_{\text{gt}}}$$
An $\text{IoU} \ge 0.5$ is standard criteria for a successful localized detection.

---

## 🔍 Scaling to Multi-Object Detection (YOLO / SSD)
This implementation localizes the primary object in the frame. To detect **multiple objects simultaneously**, modern architectures use:
- **Grid-based Feature Slicing (YOLO)**: Divides the final feature map into an $S \times S$ grid. Each cell is responsible for detecting objects centered within it.
- **Anchor Boxes (SSD / Faster R-CNN)**: Places predefined anchor boxes with varying aspect ratios across multi-scale feature maps.
- **Non-Maximum Suppression (NMS)**: Eliminates redundant overlapping bounding box predictions based on IoU thresholding.

