import torch
import torch.nn.functional as F


# needs revision
def crop_and_resize(
    image: torch.Tensor, bbox: tuple[int, int, int, int], size: int = 224
) -> torch.Tensor:
    """
    This is extracted from PascalVOCDataset.crop_to_bbox, but with resizing added.
    The original only cropped — we also resize so every crop is the same
    shape, which the CNN requires as fixed-size input.
    """

    xmin, ymin, xmax, ymax = bbox

    # Clamp coordinates so we never slice outside the image dimensions.
    # image.shape is (C, H, W), so shape[1]=height, shape[2]=width
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image.shape[2], xmax)  # can't exceed image width
    ymax = min(image.shape[1], ymax)  # can't exceed image height

    if xmax <= xmin or ymax <= ymin:
        print(f"invalid x: {xmin}, {xmax}. invalid y: {ymin}, {ymax}.")
        return None 
    else:
        cropped = image[:, ymin:ymax, xmin:xmax]

    # Crop via tensor slicing — same logic as the original crop_to_bbox.
    # [:] keeps all channels, ymin:ymax slices rows, xmin:xmax slices columns  # shape: (C, H_crop, W_crop)

    # F.interpolate requires a batch dimension (N, C, H, W), so we add one,
    # resize, then remove it so the output is back to (C, size, size)
    cropped = cropped.unsqueeze(0)  # (1, C, H_crop, W_crop)
    resized = F.interpolate(
        cropped, size=(size, size), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0)  # (C, size, size)


def xywh_to_xyxy(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    OpenCV Selective Search returns boxes as (x, y, w, h):
      x, y  = top-left corner
      w, h  = width and height

    But crop_and_resize (and the original crop_to_bbox) expect Pascal VOC
    format: (xmin, ymin, xmax, ymax) — two corner points.

    This converts between the two so the rest of the code stays consistent.
    """
    x, y, w, h = bbox
    return x, y, x + w, y + h  # xmax = x + w, ymax = y + h
