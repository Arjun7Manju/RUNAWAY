import torch
import numpy as np
from PIL import Image
import os
import argparse

from model import ObjectDetectorCNN
from dataset import CLASS_NAMES
from utils import draw_detection

def load_trained_model(model_path: str = "detector_model.pth", device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ObjectDetectorCNN(in_channels=3, num_classes=len(CLASS_NAMES))
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def detect_objects(model, image_input, device=None, output_path: str = None, gt_box=None, gt_class=None):
    """
    Runs CNN object detection on an image.
    
    Args:
        model: Trained ObjectDetectorCNN
        image_input: File path (str) or PIL Image
        device: torch device
        output_path: Optional path to save visual result
        gt_box: Optional ground truth bbox [x1, y1, x2, y2]
        gt_class: Optional ground truth class name
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(image_input, str):
        raw_img = Image.open(image_input).convert("RGB")
    else:
        raw_img = image_input.convert("RGB")

    # Resize to model input resolution (64x64)
    resized_img = raw_img.resize((64, 64))
    img_np = np.array(resized_img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, box_coords = model(tensor)
        probabilities = torch.softmax(class_logits, dim=1).squeeze(0)
        box = box_coords.squeeze(0).cpu().tolist()

    pred_idx = torch.argmax(probabilities).item()
    pred_class = CLASS_NAMES[pred_idx]
    pred_score = probabilities[pred_idx].item()

    # Draw result on original image
    annotated = draw_detection(
        image=raw_img,
        pred_box=box,
        pred_class=pred_class,
        pred_score=pred_score,
        gt_box=gt_box,
        gt_class=gt_class
    )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        annotated.save(output_path)
        print(f"[+] Saved visual detection to: {output_path}")

    return {
        "class": pred_class,
        "confidence": pred_score,
        "box": box,
        "annotated_image": annotated
    }

def main():
    parser = argparse.ArgumentParser(description="Run CNN Object Detector on an image")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--model", type=str, default="detector_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="outputs/prediction.png", help="Path to save annotated image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model):
        print(f"[-] Model checkpoint '{args.model}' not found. Run train.py or main.py first.")
        return

    model = load_trained_model(args.model, device)

    if args.image:
        result = detect_objects(model, args.image, device, output_path=args.output)
        print(f"\nDetection Results for '{args.image}':")
        print(f"  Class      : {result['class']}")
        print(f"  Confidence : {result['confidence'] * 100:.2f}%")
        print(f"  Coordinates: [x1={result['box'][0]:.3f}, y1={result['box'][1]:.3f}, x2={result['box'][2]:.3f}, y2={result['box'][3]:.3f}]")
    else:
        # Run demo on a synthetic sample
        from dataset import SyntheticShapesDataset
        val_dataset = SyntheticShapesDataset(num_samples=10, img_size=64, seed=999)
        img_tensor, label, gt_box = val_dataset[2]
        raw_img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        gt_class = CLASS_NAMES[label.item()]

        result = detect_objects(
            model=model,
            image_input=raw_img,
            device=device,
            output_path=args.output,
            gt_box=gt_box.tolist(),
            gt_class=gt_class
        )
        print(f"\n[Demo Run on Synthetic Sample]:")
        print(f"  Ground Truth: {gt_class} {gt_box.tolist()}")
        print(f"  Detected    : {result['class']} ({result['confidence'] * 100:.2f}% confidence)")
        print(f"  Pred Box    : {result['box']}")

if __name__ == "__main__":
    main()
