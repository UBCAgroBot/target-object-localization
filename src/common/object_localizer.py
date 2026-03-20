import torch
import torch.nn as nn
from PIL import Image
from selective_search import SelectiveSearch


class ObjectLocalizer(nn.Module):  # type: ignore[misc]
    """
    Two-pass object localizer:
      Pass 1 — Selective Search generates region proposals
      Pass 2 — CNN binary classifier scores each region
    """

    def __init__(
        self,
        classifier: nn.Module,
        threshold: float = 0.5,
        ss_mode: str = "fast",
        max_proposals: int = 2000,
        crop_size: int = 224,
        device: str = "cpu",
    ):
        super().__init__()

        self.classifier = classifier
        self.threshold = threshold
        self.crop_size = crop_size
        self.device = device

        self.selective_search = SelectiveSearch(
            mode=ss_mode, max_proposals=max_proposals
        )

        self.classifier.to(device)
        self.classifier.eval()

    def forward(self, pil_image: Image.Image) -> list[dict[str, object]]:
        with torch.no_grad():
            # Pass 1: region proposals via Selective Search
            crops, bboxes = self.selective_search.propose_crops(
                pil_image, size=self.crop_size
            )
            crops = crops.to(self.device)

            # Pass 2: CNN scores each proposed region
            logits = self.classifier(crops)
            probs = torch.softmax(logits, dim=1)
            scores, predicted_classes = probs.max(dim=1)

            # Filter by threshold and sort by confidence
            detections = [
                {"bbox": bbox, "score": score.item(), "class": pred_class.item()}
                for bbox, score, pred_class in zip(bboxes, scores, predicted_classes)
                if score.item() >= self.threshold
            ]

            return sorted(detections, key=lambda d: d["score"], reverse=True)
