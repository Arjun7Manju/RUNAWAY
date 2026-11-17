import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import random
import numpy as np

CLASS_NAMES = ["background", "rectangle", "circle", "triangle"]

class SyntheticShapesDataset(Dataset):
    """
    Synthetic Object Detection Dataset.
    Generates images containing geometrical shapes (rectangle, circle, triangle)
    or background, along with their ground-truth bounding box coordinates.
    Pre-generates samples in memory for ultra-fast training iterations.
    """
    def __init__(self, num_samples: int = 500, img_size: int = 64, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        
        self.images = torch.zeros((num_samples, 3, img_size, img_size), dtype=torch.float32)
        self.labels = torch.zeros(num_samples, dtype=torch.long)
        self.boxes = torch.zeros((num_samples, 4), dtype=torch.float32)

        rng = random.Random(seed)

        colors = [
            (255, 69, 0),    # Red-Orange
            (30, 144, 255),  # Dodger Blue
            (50, 205, 50),   # Lime Green
            (255, 215, 0),   # Gold
            (238, 130, 238), # Violet
            (0, 255, 255),   # Cyan
            (255, 105, 180), # Hot Pink
        ]

        for idx in range(num_samples):
            # 1. Base dark background
            base_col = rng.randint(20, 50)
            img = Image.new("RGB", (img_size, img_size), (base_col, base_col, base_col))
            draw = ImageDraw.Draw(img)

            # Class selection: 0: background (10%), 1: rect (30%), 2: circle (30%), 3: triangle (30%)
            class_id = rng.choices([0, 1, 2, 3], weights=[0.1, 0.3, 0.3, 0.3])[0]

            if class_id == 0:
                bbox = [0.0, 0.0, 0.0, 0.0]
            else:
                min_dim = int(img_size * 0.25)
                max_dim = int(img_size * 0.55)
                w = rng.randint(min_dim, max_dim)
                h = rng.randint(min_dim, max_dim)

                x1 = rng.randint(2, img_size - w - 2)
                y1 = rng.randint(2, img_size - h - 2)
                x2 = x1 + w
                y2 = y1 + h

                fill_color = rng.choice(colors)

                if class_id == 1:
                    # Rectangle
                    draw.rectangle([x1, y1, x2, y2], fill=fill_color)
                elif class_id == 2:
                    # Circle
                    draw.ellipse([x1, y1, x2, y2], fill=fill_color)
                elif class_id == 3:
                    # Triangle
                    p1 = ((x1 + x2) // 2, y1)
                    p2 = (x1, y2)
                    p3 = (x2, y2)
                    draw.polygon([p1, p2, p3], fill=fill_color)

                bbox = [x1 / img_size, y1 / img_size, x2 / img_size, y2 / img_size]

            # Store in float tensor normalized [0, 1]
            img_np = np.array(img, dtype=np.float32) / 255.0
            self.images[idx] = torch.from_numpy(img_np).permute(2, 0, 1)
            self.labels[idx] = class_id
            self.boxes[idx] = torch.tensor(bbox, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.boxes[idx]
