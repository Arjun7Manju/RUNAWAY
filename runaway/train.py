import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SyntheticShapesDataset, CLASS_NAMES
from model import ObjectDetectorCNN
from utils import calculate_iou
import os
import time

def train_epoch(model, dataloader, optimizer, criterion_cls, criterion_box, device, lambda_box=4.0):
    model.train()
    total_loss = 0.0
    correct_cls = 0
    total_samples = 0

    for images, labels, boxes in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        boxes = boxes.to(device)

        optimizer.zero_grad()
        pred_cls, pred_boxes = model(images)

        # 1. Classification Loss (CrossEntropy)
        loss_cls = criterion_cls(pred_cls, labels)

        # 2. Bounding Box Loss (Smooth L1 only on object classes, ignoring background)
        has_object = (labels > 0).float()
        raw_box_loss = criterion_box(pred_boxes, boxes).mean(dim=1)
        if has_object.sum() > 0:
            loss_box = (raw_box_loss * has_object).sum() / (has_object.sum() + 1e-7)
        else:
            loss_box = torch.tensor(0.0, device=device)

        # Multi-task loss
        loss = loss_cls + (lambda_box * loss_box)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(pred_cls, dim=1)
        correct_cls += (preds == labels).sum().item()
        total_samples += images.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = (correct_cls / total_samples) * 100.0
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion_cls, criterion_box, device, lambda_box=4.0):
    model.eval()
    total_loss = 0.0
    correct_cls = 0
    total_samples = 0
    total_iou = 0.0
    object_count = 0

    with torch.no_grad():
        for images, labels, boxes in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            boxes = boxes.to(device)

            pred_cls, pred_boxes = model(images)

            loss_cls = criterion_cls(pred_cls, labels)
            has_object = (labels > 0).float()
            raw_box_loss = criterion_box(pred_boxes, boxes).mean(dim=1)
            loss_box = (raw_box_loss * has_object).sum() / (has_object.sum() + 1e-7)

            loss = loss_cls + (lambda_box * loss_box)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(pred_cls, dim=1)
            correct_cls += (preds == labels).sum().item()
            total_samples += images.size(0)

            # IoU evaluation for detected objects
            ious = calculate_iou(pred_boxes, boxes)
            obj_mask = labels > 0
            if obj_mask.sum() > 0:
                total_iou += ious[obj_mask].sum().item()
                object_count += obj_mask.sum().item()

    val_loss = total_loss / total_samples
    val_acc = (correct_cls / total_samples) * 100.0
    mean_iou = (total_iou / max(1, object_count))
    return val_loss, val_acc, mean_iou

def train_detector(
    num_epochs: int = 12,
    batch_size: int = 32,
    lr: float = 2e-3,
    save_path: str = "detector_model.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}", flush=True)

    print("[*] Generating synthetic datasets...", flush=True)
    t0 = time.time()
    train_dataset = SyntheticShapesDataset(num_samples=500, img_size=64, seed=42)
    val_dataset = SyntheticShapesDataset(num_samples=120, img_size=64, seed=123)
    print(f"[+] Datasets ready in {time.time() - t0:.2f}s", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ObjectDetectorCNN(in_channels=3, num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_box = nn.SmoothL1Loss(reduction="none")

    print(f"[*] Starting CNN training for {num_epochs} epochs...", flush=True)
    start_time = time.time()

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_box, device
        )
        val_loss, val_acc, mean_iou = evaluate(
            model, val_loader, criterion_cls, criterion_box, device
        )
        scheduler.step()

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%, Mean IoU: {mean_iou:.3f}",
            flush=True
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "mean_iou": mean_iou
            }, save_path)

    total_time = time.time() - start_time
    print(f"\n[+] Training completed in {total_time:.1f} seconds. Model saved to '{save_path}'", flush=True)
    return model

if __name__ == "__main__":
    train_detector(num_epochs=12)
