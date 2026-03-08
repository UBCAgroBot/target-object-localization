import torch
import torch.nn as nn
from PIL import Image
from selective_search import SelectiveSearch  # updated import


class ObjectLocalizer(nn.Module):
    """
    Two-pass object localizer:
      Pass 1 — Selective Search generates region proposals
      Pass 2 — CNN binary classifier scores each region
    """

    def __init__(
        self,
        classifier: nn.Module,
        threshold: float = 0.5,
        ss_mode: str = "fast",  # renamed from rpn_mode
        max_proposals: int = 2000,
        crop_size: int = 224,
        device: str = "cpu",
    ):
        super().__init__()

        self.classifier = classifier
        self.threshold = threshold
        self.crop_size = crop_size
        self.device = device

        self.selective_search = SelectiveSearch(  # renamed from self.rpn
            mode=ss_mode, max_proposals=max_proposals
        )

        self.classifier.to(device)
        self.classifier.eval()

    @torch.no_grad()
    def forward(self, pil_image: Image.Image) -> list[dict]:
        """
        Run full two-pass detection on a single image.

        Returns:
            List of detections sorted by confidence (highest first):
            [{"bbox": (xmin, ymin, xmax, ymax), "score": float}, ...]
        """
        # Pass 1: region proposals via Selective Search
        crops, bboxes = self.selective_search.propose_crops(
            pil_image, size=self.crop_size
        )
        crops = crops.to(self.device)

        # Pass 2: CNN scores each proposed region
        logits = self.classifier(crops)
        scores = torch.sigmoid(logits).squeeze(1)

        # Filter by threshold and sort by confidence
        detections = [
            {"bbox": bbox, "score": score.item()}
            for bbox, score in zip(bboxes, scores)
            if score.item() >= self.threshold
        ]

        return sorted(detections, key=lambda d: d["score"], reverse=True)


### How they connect end-to-end (notes for personal understanding)
# ```
# PIL Image
#    │
#    v
# rpn.propose()          # OpenCV Selective Search → ~2000 (x,y,w,h) boxes
#    │
#    v
# xywh_to_xyxy()         # utils.py — convert box format for compatibility
#    │
#    v
# crop_and_resize()      # utils.py — crop each region, resize to 224×224
#    │
#    v
# torch.stack(crops)     # single (N, 3, 224, 224) batch
#    │
#    v
# classifier(crops)      # your CNN — one score per region
#    │
#    v
# sigmoid + threshold    # filter out background regions
#    │
#    v
# sorted detections      # [{"bbox": ..., "score": ...}, ...]
