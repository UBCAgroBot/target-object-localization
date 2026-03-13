import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class PascalVOCDataset(Dataset[Tuple[torch.Tensor, Dict[str, Any]]]):
    """
    Pascal VOC Dataset for object localization.
    Each sample represents one bounding box (object instance).
    """

    def __init__(
        self,
        root_dir: str,
        year: str = "2012",
        split: str = "train",
        transform: transforms.Compose | None = None,
        include_difficult: bool = False,
        target_size: int = 64,
    ):
        """
        Args:
            root_dir: Root directory containing VOCdevkit/
            year: Dataset year ('2007' or '2012')
            split: One of 'train', 'val', or 'trainval'
            transform: Optional transform to apply to images
            include_difficult: Whether to include objects marked as difficult
        """
        self.root_dir = root_dir
        self.year = year
        self.split = split
        self.transform = transform
        self.include_difficult = include_difficult
        self.target_size = target_size

        self.voc_root = os.path.join(root_dir, f"VOC{year}_train_val")
        self.image_dir = os.path.join(self.voc_root, "JPEGImages")
        self.annotation_dir = os.path.join(self.voc_root, "Annotations")

        # Pascal VOC classes
        self.classes = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load image IDs from split file
        split_file = os.path.join(self.voc_root, "ImageSets", "Main", f"{split}.txt")
        with open(split_file, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        # Parse all annotations and create list of (image_id, bbox, label)
        self.samples: List[Dict[str, Any]] = []
        self._parse_annotations()

        print(
            f"Loaded {len(self.samples)} object instances from {len(self.image_ids)} images"
        )

    def _parse_annotations(self) -> None:
        """Parse all XML annotations and create sample list."""
        for image_id in self.image_ids:
            annotation_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            # Get image dimensions
            size = root.find("size")
            if size is None:
                raise ValueError(f"Missing size element in annotation for {image_id}")

            width_elem = size.find("width")
            if width_elem is None or width_elem.text is None:
                raise ValueError(f"Missing width in annotation for {image_id}")
            img_width = int(width_elem.text)

            height_elem = size.find("height")
            if height_elem is None or height_elem.text is None:
                raise ValueError(f"Missing height in annotation for {image_id}")
            img_height = int(height_elem.text)

            # Parse all objects in the image
            for obj in root.findall("object"):
                difficult_elem = obj.find("difficult")
                if difficult_elem is None or difficult_elem.text is None:
                    raise ValueError(
                        f"Missing difficult field in annotation for {image_id}"
                    )
                difficult = int(difficult_elem.text)

                # Skip difficult objects if specified
                if difficult and not self.include_difficult:
                    continue

                # Get class label
                name_elem = obj.find("name")
                if name_elem is None or name_elem.text is None:
                    raise ValueError(f"Missing name field in annotation for {image_id}")
                class_name = name_elem.text
                if class_name not in self.class_to_idx:
                    continue

                label = self.class_to_idx[class_name]

                # Get bounding box
                bbox = obj.find("bndbox")
                if bbox is None:
                    raise ValueError(f"Missing bndbox in annotation for {image_id}")

                xmin_elem = bbox.find("xmin")
                if xmin_elem is None or xmin_elem.text is None:
                    raise ValueError(f"Missing xmin in bounding box for {image_id}")
                xmin = int(xmin_elem.text)

                ymin_elem = bbox.find("ymin")
                if ymin_elem is None or ymin_elem.text is None:
                    raise ValueError(f"Missing ymin in bounding box for {image_id}")
                ymin = int(ymin_elem.text)

                xmax_elem = bbox.find("xmax")
                if xmax_elem is None or xmax_elem.text is None:
                    raise ValueError(f"Missing xmax in bounding box for {image_id}")
                xmax = int(xmax_elem.text)

                ymax_elem = bbox.find("ymax")
                if ymax_elem is None or ymax_elem.text is None:
                    raise ValueError(f"Missing ymax in bounding box for {image_id}")
                ymax = int(ymax_elem.text)

                # Store sample
                self.samples.append(
                    {
                        "image_id": image_id,
                        "bbox": [xmin, ymin, xmax, ymax],
                        "label": label,
                        "class_name": class_name,
                        "img_width": img_width,
                        "img_height": img_height,
                    }
                )

    def __len__(self) -> int:
        """Return total number of object instances."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            image: Cropped image tensor of shape (C, H, W)
            target: Dictionary with 'bbox', 'label', 'class_name', etc.
        """
        sample = self.samples[idx]

        # Load image
        image_path = os.path.join(self.image_dir, f"{sample['image_id']}.jpg")
        image = Image.open(image_path).convert("RGB")

        # Convert to tensor
        image_tensor = transforms.ToTensor()(image)
        cropped_image = self.crop_to_bbox(image_tensor, sample["bbox"])

        # Return full image and annotation
        target = {
            "bbox": torch.tensor(sample["bbox"], dtype=torch.float32),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "class_name": sample["class_name"],
            "image_id": sample["image_id"],
        }

        return cropped_image, target

    def crop_to_bbox(self, image: torch.Tensor, bbox: List[int]) -> torch.Tensor:
        """
        Crop image tensor to bounding box, convert to square, then resize to target size.

        Args:
            image: Image tensor of shape (C, H, W)
            bbox: Bounding box as [xmin, ymin, xmax, ymax]

        Returns:
            Cropped image tensor of shape (C, target_size, target_size)
        """
        xmin, ymin, xmax, ymax = bbox

        # Calculate center and size
        xcenter = xmin + (xmax - xmin) // 2
        ycenter = ymin + (ymax - ymin) // 2
        w = xmax - xmin
        h = ymax - ymin
        size = max(w, h)

        # Calculate square crop bounds
        crop_xmin = xcenter - size // 2
        crop_ymin = ycenter - size // 2
        crop_xmax = crop_xmin + size
        crop_ymax = crop_ymin + size

        # Handle edge cases - adjust if crop goes out of bounds
        img_h, img_w = image.shape[1], image.shape[2]

        if crop_xmin < 0:
            crop_xmax = crop_xmax - crop_xmin
            crop_xmin = 0
        if crop_ymin < 0:
            crop_ymax = crop_ymax - crop_ymin
            crop_ymin = 0
        if crop_xmax > img_w:
            crop_xmin = crop_xmin - (crop_xmax - img_w)
            crop_xmax = img_w
        if crop_ymax > img_h:
            crop_ymin = crop_ymin - (crop_ymax - img_h)
            crop_ymax = img_h

        # Ensure coordinates are within bounds
        crop_xmin = max(0, crop_xmin)
        crop_ymin = max(0, crop_ymin)
        crop_xmax = min(img_w, crop_xmax)
        crop_ymax = min(img_h, crop_ymax)

        # Crop using tensor slicing
        cropped = image[:, crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        # Resize to target size
        resize = transforms.Resize((self.target_size, self.target_size))
        cropped = resize(cropped)

        return cropped  # type: ignore[no-any-return]

    def get_cropped_object(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Convenience method to get already-cropped object region.

        Returns:
            cropped_image: Cropped image tensor
            target: Dictionary with label and metadata
        """
        image, target = self.__getitem__(idx)
        bbox = target["bbox"].int().tolist()
        cropped_image = self.crop_to_bbox(image, bbox)

        return cropped_image, target


# ---------------------------------------------------------------------
# Example usage (uncomment to visualize)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = PascalVOCDataset(root_dir="data/VOC2012", year="2012", split="trainval")

    print(f"Total objects: {len(dataset)}")

    # Example: show one cropped object
    image, target = dataset.get_cropped_object(123)
    plt.imshow(image.permute(1, 2, 0))
    plt.title(target["class_name"])
    plt.axis("off")
    plt.show()
