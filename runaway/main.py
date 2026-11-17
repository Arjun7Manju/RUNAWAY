import os
import torch
import numpy as np
from PIL import Image

from train import train_detector
from predict import load_trained_model, detect_objects
from dataset import SyntheticShapesDataset, CLASS_NAMES
from utils import calculate_iou

def create_detection_gallery(model, dataset, num_samples=6, output_path="outputs/detection_grid.png"):
    """
    Runs detection on multiple test images and stitches them into a comparison grid.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    images_annotated = []
    
    print("\n" + "="*60, flush=True)
    print("   RUNNING CNN OBJECT DETECTION ON TEST IMAGES", flush=True)
    print("="*60, flush=True)

    for i in range(num_samples):
        img_tensor, label, gt_box = dataset[i]
        raw_img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        gt_class = CLASS_NAMES[label.item()]

        res = detect_objects(
            model=model,
            image_input=raw_img,
            gt_box=gt_box.tolist(),
            gt_class=gt_class
        )
        
        # Calculate IoU for this prediction
        pred_box_tensor = torch.tensor(res["box"]).unsqueeze(0)
        gt_box_tensor = gt_box.unsqueeze(0)
        iou = calculate_iou(pred_box_tensor, gt_box_tensor).item() if label.item() > 0 else 1.0

        print(
            f"Sample #{i+1:02d} | Ground Truth: {gt_class:<10} | "
            f"Detected: {res['class']:<10} | Conf: {res['confidence']*100:.1f}% | IoU: {iou:.3f}",
            flush=True
        )
        images_annotated.append(res["annotated_image"])

    # Stitch into a 2x3 grid
    cols = 3
    rows = (num_samples + cols - 1) // cols
    w, h = images_annotated[0].size
    grid = Image.new("RGB", (cols * w, rows * h), color=(20, 20, 20))

    for idx, img in enumerate(images_annotated):
        c = idx % cols
        r = idx // cols
        grid.paste(img, (c * w, r * h))

    grid.save(output_path)
    print(f"\n[+] Saved test gallery to: {output_path}", flush=True)

def main():
    print("==================================================", flush=True)
    print("   CNN OBJECT DETECTOR (PyTorch Implementation)   ", flush=True)
    print("==================================================", flush=True)
    
    # 1. Train the CNN Detector
    checkpoint_file = "detector_model.pth"
    model = train_detector(num_epochs=12, batch_size=32, lr=2e-3, save_path=checkpoint_file)

    # 2. Test Set Evaluation & Visualization
    test_dataset = SyntheticShapesDataset(num_samples=20, img_size=64, seed=777)
    create_detection_gallery(model, test_dataset, num_samples=6, output_path="outputs/detection_grid.png")

    print("\n==================================================", flush=True)
    print("   Demo Complete! Model ready for custom usage.   ", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    main()
