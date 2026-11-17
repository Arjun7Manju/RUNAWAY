import torch
from PIL import Image, ImageDraw, ImageFont

def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between two sets of boxes.
    Boxes are in format [x1, y1, x2, y2] normalized to [0, 1].
    
    Args:
        box1: Tensor of shape (..., 4)
        box2: Tensor of shape (..., 4)
    Returns:
        Tensor of IoU values.
    """
    # Box coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Intersection coordinates
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    # Union Area
    area1 = torch.clamp(b1_x2 - b1_x1, min=0.0) * torch.clamp(b1_y2 - b1_y1, min=0.0)
    area2 = torch.clamp(b2_x2 - b2_x1, min=0.0) * torch.clamp(b2_y2 - b2_y1, min=0.0)
    union_area = area1 + area2 - inter_area + 1e-7

    return inter_area / union_area

def draw_detection(
    image: Image.Image,
    pred_box: list,
    pred_class: str,
    pred_score: float = None,
    gt_box: list = None,
    gt_class: str = None
) -> Image.Image:
    """
    Visualizes predicted and ground-truth bounding boxes on a PIL Image.
    
    Args:
        image: PIL Image
        pred_box: [x1, y1, x2, y2] normalized (0 to 1)
        pred_class: Name of predicted class
        pred_score: Prediction confidence score (0 to 1)
        gt_box: Optional ground truth [x1, y1, x2, y2]
        gt_class: Optional ground truth class name
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Draw Ground Truth (if provided) in Blue / Cyan
    if gt_box is not None and gt_class is not None and gt_class != "background":
        gx1, gy1, gx2, gy2 = gt_box[0] * w, gt_box[1] * h, gt_box[2] * w, gt_box[3] * h
        draw.rectangle([gx1, gy1, gx2, gy2], outline="#00d2ff", width=2)
        draw.text((gx1 + 2, gy1 + 2), f"GT: {gt_class}", fill="#00d2ff")

    # Draw Prediction in Lime Green / Coral Red
    if pred_box is not None and pred_class != "background":
        px1, py1, px2, py2 = pred_box[0] * w, pred_box[1] * h, pred_box[2] * w, pred_box[3] * h
        draw.rectangle([px1, py1, px2, py2], outline="#00ff66", width=2)
        
        label = f"Pred: {pred_class}"
        if pred_score is not None:
            label += f" ({pred_score:.2f})"
        
        # Draw label background for readability
        draw.text((px1 + 2, max(0, py1 - 12)), label, fill="#00ff66")

    return img

