import math
import os
import xml.etree.ElementTree as ET
from glob import glob
from typing import Generator

import torch
from binary_classifier import BinaryClassifier
from PIL import Image, ImageDraw
from torchvision import transforms

FOLDER = "data/VOC2012/VOC2012_test/JPEGImages"
MODEL_PATH = "checkpoints/best_model.pth"
THRESHOLD = 0.9
OUTPUT_PATH = "results/collage.jpg"
WINDOW_SIZES = [256, 512]
STRIDE_RATIO = 0.5
NMS_IOU_THRESHOLD = 0.2
ANNOTATION_DIR = "data/VOC2012/VOC2012_test/Annotations"
TARGET_CLASSES = {"person", "cat"}


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def sliding_window(
    image_width: int, image_height: int, window_size: int, stride: int
) -> Generator[tuple[int, int, int, int], None, None]:
    """Generate sliding window coordinates."""
    for y in range(0, image_height - window_size + 1, stride):
        for x in range(0, image_width - window_size + 1, stride):
            yield (x, y, x + window_size, y + window_size)

    # Handle right edge
    if (image_width - window_size) % stride != 0:
        for y in range(0, image_height - window_size + 1, stride):
            x = image_width - window_size
            yield (x, y, x + window_size, y + window_size)

    # Handle bottom edge
    if (image_height - window_size) % stride != 0:
        for x in range(0, image_width - window_size + 1, stride):
            y = image_height - window_size
            yield (x, y, x + window_size, y + window_size)

    # Handle bottom-right corner
    if (image_width - window_size) % stride != 0 and (
        image_height - window_size
    ) % stride != 0:
        yield (
            image_width - window_size,
            image_height - window_size,
            image_width,
            image_height,
        )


def compute_iou(
    box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def non_max_suppression(
    detections: list[tuple[tuple[int, int, int, int], float]], iou_threshold: float
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Apply NMS to detections. Each detection is (box, confidence)."""
    if not detections:
        return []

    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x[1], reverse=True)

    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)

        detections = [
            d for d in detections if compute_iou(best[0], d[0]) < iou_threshold
        ]

    return keep


def detect_in_image(
    image: Image.Image,
    model: BinaryClassifier,
    transform: transforms.Compose,
    device: torch.device,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Run sliding window detection on a single image."""
    width, height = image.size
    all_detections = []

    for window_size in WINDOW_SIZES:
        if width < window_size or height < window_size:
            continue

        stride = int(window_size * STRIDE_RATIO)

        # Collect all windows for this scale
        windows = list(sliding_window(width, height, window_size, stride))
        crops = []
        coords = []

        for x1, y1, x2, y2 in windows:
            crop = image.crop((x1, y1, x2, y2))
            crops.append(transform(crop))
            coords.append((x1, y1, x2, y2))

        # Batch inference
        BATCH_SIZE = 64
        for batch_start in range(0, len(crops), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(crops))
            batch_tensor = torch.stack(crops[batch_start:batch_end]).to(device)

            with torch.no_grad():
                outputs = model(batch_tensor)
                confidences = torch.sigmoid(outputs).squeeze(-1)

            for j, conf in enumerate(confidences):
                if conf.item() >= THRESHOLD:
                    all_detections.append((coords[batch_start + j], conf.item()))

    detections = non_max_suppression(all_detections, NMS_IOU_THRESHOLD)
    return detections


def get_valid_image_ids() -> set[str]:
    """Get image IDs that contain target classes."""
    valid_ids = set()

    for xml_file in os.listdir(ANNOTATION_DIR):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(ANNOTATION_DIR, xml_file))
        root = tree.getroot()

        classes_in_image = set()
        for obj in root.findall("object"):
            name = obj.find("name")
            if name is not None and name.text:
                classes_in_image.add(name.text)

        if classes_in_image & TARGET_CLASSES:
            image_id = xml_file.replace(".xml", "")
            valid_ids.add(image_id)

    return valid_ids


def run_inference() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = BinaryClassifier(device=str(device))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = get_transform()

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(FOLDER, ext)))
        image_paths.extend(glob(os.path.join(FOLDER, ext.upper())))

    print(f"Found {len(image_paths)} images")

    # Filter to only images with target classes
    valid_ids = get_valid_image_ids()
    image_paths = [
        p for p in image_paths if os.path.splitext(os.path.basename(p))[0] in valid_ids
    ]
    print(f"Filtered to {len(image_paths)} images containing {TARGET_CLASSES}")

    results = []  # (path, image_with_boxes, num_detections, max_confidence)

    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        detections = detect_in_image(img, model, transform, device)

        if detections:
            # Draw bounding boxes on image
            img_with_boxes = img.copy()
            draw = ImageDraw.Draw(img_with_boxes)

            max_conf = 0.0
            for box, conf in detections:
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1] - 15), f"{conf:.2f}", fill="red")
                max_conf = max(max_conf, conf)

            results.append((path, img_with_boxes, len(detections), max_conf))
            print(
                f"✓ {os.path.basename(path)}: {len(detections)} detections (max conf: {max_conf:.3f})"
            )

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(image_paths)} images...")

    print(f"\n{len(results)} images with detections")

    if not results:
        print("No detections found.")
        return

    # Sort by max confidence
    results.sort(key=lambda x: x[3], reverse=True)

    # Create collage with bounding boxes
    GRID_SIZE = 5
    THUMB_SIZE = 200
    PAGE_SIZE = GRID_SIZE * THUMB_SIZE

    num_pages = math.ceil(len(results) / 25)

    for page_idx in range(num_pages):
        collage = Image.new("RGB", (PAGE_SIZE, PAGE_SIZE), (255, 255, 255))
        start = page_idx * 25
        end = min(start + 25, len(results))

        for i, (path, img_with_boxes, num_det, max_conf) in enumerate(
            results[start:end]
        ):
            thumb = img_with_boxes.copy()
            thumb.thumbnail((THUMB_SIZE, THUMB_SIZE - 20))

            cell = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (255, 255, 255))
            x_offset = (THUMB_SIZE - thumb.width) // 2
            cell.paste(thumb, (x_offset, 0))

            # Add text showing detections and confidence
            draw = ImageDraw.Draw(cell)
            text = f"{num_det} det | {max_conf:.2f}"
            bbox = draw.textbbox((0, 0), text)
            text_x = (THUMB_SIZE - (bbox[2] - bbox[0])) // 2
            draw.text((text_x, THUMB_SIZE - 16), text, fill=(255, 0, 0))

            row, col = divmod(i, GRID_SIZE)
            collage.paste(cell, (col * THUMB_SIZE, row * THUMB_SIZE))

        if num_pages == 1:
            save_path = OUTPUT_PATH
        else:
            base, ext = os.path.splitext(OUTPUT_PATH)
            save_path = f"{base}_page{page_idx + 1}{ext}"

        collage.save(save_path)
        print(f"Saved collage: {save_path}")


if __name__ == "__main__":
    run_inference()
