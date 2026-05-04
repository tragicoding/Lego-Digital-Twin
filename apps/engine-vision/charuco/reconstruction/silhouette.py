"""
LEGO silhouette mask extraction.

Strategy
--------
1. Remove background using HSV-based colour segmentation or GrabCut.
2. Optionally refine with morphological ops.
3. Return binary mask (255 = LEGO region).

For the TOP view, also returns a colour-annotated version used by color_estimator.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class SilhouetteResult:
    mask: np.ndarray          # uint8 binary (H×W)
    masked_image: np.ndarray  # original image with background zeroed
    view: str


def extract_by_background_subtraction(image: np.ndarray,
                                       background: np.ndarray,
                                       threshold: int = 30) -> np.ndarray:
    """
    Diff-based mask when a clean background frame is available.
    Best accuracy; requires a reference shot before placing LEGO.
    """
    diff = cv2.absdiff(image, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = _clean_mask(mask)
    return mask


def extract_by_color_range(image: np.ndarray,
                            lower_hsv: tuple = (0, 40, 40),
                            upper_hsv: tuple = (180, 255, 255),
                            background_color_bgr: tuple = (200, 200, 200)) -> np.ndarray:
    """
    HSV-range mask.  Tune lower/upper to reject the background board colour.
    For a white/grey baseplate, invert: keep pixels that are NOT near background.
    """
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    mask = _clean_mask(mask)
    return mask


def extract_by_grabcut(image: np.ndarray,
                        rect: tuple | None = None) -> np.ndarray:
    """
    GrabCut when no background reference is available.
    rect = (x, y, w, h) ROI hint; if None, uses full-image centre crop.
    """
    h, w = image.shape[:2]
    if rect is None:
        margin = 20
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mask_gc   = np.zeros(image.shape[:2], np.uint8)

    cv2.grabCut(image, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask = _clean_mask(mask)
    return mask


def _clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask


def build_silhouette(image: np.ndarray, view: str,
                     background: np.ndarray | None = None,
                     method: str = "background") -> SilhouetteResult:
    """
    Unified entry point.

    method: "background" | "color" | "grabcut"
    """
    if method == "background" and background is not None:
        mask = extract_by_background_subtraction(image, background)
    elif method == "color":
        mask = extract_by_color_range(image)
    else:
        mask = extract_by_grabcut(image)

    masked = cv2.bitwise_and(image, image, mask=mask)
    return SilhouetteResult(mask=mask, masked_image=masked, view=view)
