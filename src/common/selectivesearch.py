import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from utils import crop_and_resize, xywh_to_xyxy

# needs revision


class SelectiveSearch:
    """
    Wraps OpenCV's Selective Search algorithm to generate candidate bounding
    boxes (region proposals) from an image.

    This is a region proposal algorithm, NOT a learned network.
    """

    def __init__(
        self, mode: str = "fast", max_proposals: int = 2000, min_size: int = 20
    ):
        self.mode = mode
        self.max_proposals = max_proposals
        self.min_size = min_size
        self._ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def propose(self, pil_image: Image.Image) -> list[tuple[int, int, int, int]]:
        """
        Run Selective Search on a PIL image and return filtered bounding boxes.
        Returns boxes in Pascal VOC format (xmin, ymin, xmax, ymax).
        """
        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        self._ss.setBaseImage(img_bgr)

        if self.mode == "fast":
            self._ss.switchToSelectiveSearchFast()
        else:
            self._ss.switchToSelectiveSearchQuality()

        rects = self._ss.process()

        boxes = []
        for x, y, w, h in rects:
            if w >= self.min_size and h >= self.min_size:
                boxes.append(xywh_to_xyxy((x, y, w, h)))
            if len(boxes) >= self.max_proposals:
                break

        return boxes

    def propose_crops(
        self, pil_image: Image.Image, size: int = 224
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
        """
        Propose regions and return cropped + resized tensors ready for CNN.

        Returns:
            crops:  Tensor of shape (N, 3, size, size)
            bboxes: Corresponding boxes in xyxy format
        """
        img_tensor = transforms.ToTensor()(pil_image)
        bboxes = self.propose(pil_image)
        if bboxes is not None and len(bboxes) > 0:
            print(f"detected {len(bboxes)} boxes")
        else:
            print("No bounding boxes found")

        crops = [crop_and_resize(img_tensor, bb, size) for bb in bboxes]

        return torch.stack(crops), bboxes
