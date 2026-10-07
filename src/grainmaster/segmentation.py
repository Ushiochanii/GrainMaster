"""One-class YOLO26 inference and explicitly unvalidated classical bootstrap."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .contracts import SeedInstance


MASK_CLEANUP_VERSION = "largest-component-v2"


def _clean_mask_fragments(mask):
    """Keep only the largest 8-connected component of one seed instance.

    A physical seed should form one connected foreground body. Any detached
    component inside the same model instance is treated as segmentation noise
    and excluded before bbox, polygon, QC, and phenotype measurement.
    """
    binary = np.asarray(mask, dtype=bool).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    metadata = {
        "mask_cleanup_version": MASK_CLEANUP_VERSION,
        "component_count_before": max(0, int(count) - 1),
        "component_count_after": 0,
        "fragment_removed_pixels": 0,
        "fragment_removed_components": 0,
    }
    if count <= 1:
        return binary.astype(bool), metadata

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    clean = labels == largest
    metadata["component_count_after"] = 1
    metadata["main_component_pixels"] = int(stats[largest, cv2.CC_STAT_AREA])
    metadata["fragment_removed_components"] = max(0, int(count) - 2)
    metadata["fragment_removed_pixels"] = int(binary.sum() - clean.sum())
    return clean, metadata


DUPLICATE_MASK_IOU = .80
DUPLICATE_MASK_CONTAINMENT = .95


def _mask_overlap_metrics(first, second):
    """Return mask IoU and smaller-mask containment using bbox-gated overlap."""
    ax, ay, aw, ah = first.bbox
    bx, by, bw, bh = second.bbox
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0., 0.

    a = first.mask[y0:y1, x0:x1]
    b = second.mask[y0:y1, x0:x1]
    intersection = int(np.count_nonzero(a & b))
    if not intersection:
        return 0., 0.
    area_a = int(np.count_nonzero(first.mask))
    area_b = int(np.count_nonzero(second.mask))
    union = area_a + area_b - intersection
    return intersection / union, intersection / min(area_a, area_b)


def _suppress_duplicate_instances(instances, iou_threshold=DUPLICATE_MASK_IOU,
                                  containment_threshold=DUPLICATE_MASK_CONTAINMENT):
    """Suppress near-identical masks after connected-component cleanup."""
    kept = []
    for candidate in sorted(instances, key=lambda seed: seed.confidence, reverse=True):
        duplicate_of = None
        duplicate_metrics = None
        for accepted in kept:
            iou, containment = _mask_overlap_metrics(candidate, accepted)
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate_of = accepted
                duplicate_metrics = (iou, containment)
                break
        if duplicate_of is None:
            candidate.metadata["duplicate_suppressed_count"] = 0
            kept.append(candidate)
        else:
            duplicate_of.metadata["duplicate_suppressed_count"] = (
                duplicate_of.metadata.get("duplicate_suppressed_count", 0) + 1)
            duplicate_of.metadata.setdefault("suppressed_duplicates", []).append({
                "confidence": float(candidate.confidence),
                "iou": float(duplicate_metrics[0]),
                "containment": float(duplicate_metrics[1]),
            })
    return kept


def _instances(masks, confidences, backend):
    found = []
    for mask, confidence in zip(masks, confidences):
        mask, cleanup = _clean_mask_fragments(mask)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        found.append(SeedInstance(0, mask.astype(bool), float(confidence),
                                 cv2.boundingRect(cv2.findNonZero(mask.astype(np.uint8))),
                                 contour[:, 0, :],
                                 {**cleanup, "backend": backend, "label_source":
                                  "pseudo-label" if backend == "classical" else "prediction",
                                  "confidence_kind": "heuristic" if backend == "classical"
                                  else "model"}))
    found = _suppress_duplicate_instances(found)
    found.sort(key=lambda seed: (seed.bbox[1], seed.bbox[0]))
    for index, seed in enumerate(found, 1):
        seed.seed_id = index
    return found


def _classical(crop):
    height, width = crop.shape[:2]
    size = min(height, width)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # The media is blue; yellow/brown seed pixels occupy warm hues. White
    # reflections and bright blue speckles are deliberately not foreground.
    foreground = ((hsv[..., 0] < 45) & (hsv[..., 1] > 45) & (hsv[..., 2] > 55))
    interior = np.zeros((height, width), np.uint8)
    cv2.ellipse(interior, (width // 2, height // 2),
                (round(width * .445), round(height * .445)), 0, 0, 360, 1, -1)
    foreground = (foreground & interior.astype(bool)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    min_area = max(25, round(height * width * .0012))
    count, components, stats, _ = cv2.connectedComponentsWithStats(foreground)
    clean = np.zeros_like(foreground, dtype=bool)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean |= components == label
    if not clean.any():
        return []
    clean = ndi.binary_fill_holes(clean)
    components, count = ndi.label(clean)
    areas = np.bincount(components.ravel())[1:]
    typical_area = float(np.median(np.sort(areas)[:max(1, (len(areas) + 1) // 2)]))
    distance = ndi.gaussian_filter(ndi.distance_transform_edt(clean), max(.6, size * .003))
    # Isolated grains remain whole. Watershed is applied only to oversized
    # connected groups, preventing their internal grooves from becoming seeds.
    markers = np.zeros_like(components)
    next_marker = 1
    for component in range(1, count + 1):
        mask = components == component
        if areas[component - 1] < typical_area * 1.4:
            markers[mask] = next_marker
            next_marker += 1
        else:
            positions = peak_local_max(distance, min_distance=max(3, round(size * .034)),
                                       labels=mask.astype(np.uint8), exclude_border=False)
            peaks = np.zeros_like(components)
            peak_count = len(positions)
            for peak_id, (py, px) in enumerate(positions, 1):
                peaks[py, px] = peak_id
            if peak_count == 0:
                peaks[mask] = 1
                peak_count = 1
            markers[peaks > 0] = peaks[peaks > 0] + next_marker - 1
            next_marker += peak_count
    labels = watershed(-distance, markers, mask=clean)
    masks = []
    for label in range(1, int(labels.max()) + 1):
        mask = labels == label
        area = int(mask.sum())
        if min_area <= area <= height * width * .04:
            masks.append(mask)
    return _instances(masks, [.5] * len(masks), "classical")


@lru_cache(maxsize=2)
def _load_model(weights):
    from ultralytics import YOLO
    model = YOLO(weights)
    names = model.names
    if model.task != "segment" or list(names.values()) != ["seed"]:
        raise ValueError("YOLO backend requires a trained segmentation checkpoint with only seed")
    architecture = model.model.yaml
    yaml_name = Path(str(architecture.get("yaml_file", ""))).stem
    scale = architecture.get("scale")
    if "yolo26" not in yaml_name or "seg" not in yaml_name or scale != "s":
        raise ValueError("Checkpoint architecture must be YOLO26s-seg")
    trained_size = (model.ckpt or {}).get("train_args", {}).get("imgsz")
    if trained_size != 1024:
        raise ValueError("Checkpoint must have been trained with imgsz=1024")
    return model


def _enable_rocm_cpu_nms_fallback():
    """Use CPU NMS on ROCm when torchvision's GPU NMS kernel is unavailable.

    Some Windows ROCm torchvision builds can run ordinary torch GPU kernels on
    RDNA3 but fail specifically in torchvision.ops.nms with
    hipErrorInvalidDeviceFunction. Keep the model forward pass on GPU and move
    only the small NMS inputs to CPU.
    """
    import torch
    import torchvision

    if not getattr(torch.version, "hip", None):
        return
    current = torchvision.ops.nms
    if getattr(current, "_grainmaster_rocm_cpu_fallback", False):
        return

    original = current

    def rocm_safe_nms(boxes, scores, iou_threshold):
        if boxes.is_cuda:
            keep = original(boxes.detach().cpu(), scores.detach().cpu(), iou_threshold)
            return keep.to(boxes.device)
        return original(boxes, scores, iou_threshold)

    rocm_safe_nms._grainmaster_rocm_cpu_fallback = True
    torchvision.ops.nms = rocm_safe_nms


def segment_seeds(crop_bgr, backend="yolo26", weights=None, imgsz=1024, confidence=.25, device="cpu"):
    """Return boolean instance masks in original dish-crop coordinates.

    Classical masks/confidence are bootstrap candidates, never human ground truth.
    COCO pretrained weights are rejected by the production seed backend.
    """
    if crop_bgr is None or crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR dish crop")
    if backend == "classical":
        return _classical(crop_bgr)
    if backend != "yolo26":
        raise ValueError(f"Unknown segmentation backend: {backend}")
    if imgsz != 1024:
        raise ValueError("YOLO26s-seg seed inference requires imgsz=1024")
    if not weights or not Path(weights).is_file():
        raise FileNotFoundError("Supply a trained YOLO26s-seg seed checkpoint; use classical for P0")
    if str(device).lower() not in {"cpu", "none"}:
        _enable_rocm_cpu_nms_fallback()
    result = _load_model(str(Path(weights).resolve())).predict(
        crop_bgr, imgsz=1024, conf=confidence, retina_masks=True, verbose=False, device=device)[0]
    if result.masks is None:
        return []
    height, width = crop_bgr.shape[:2]
    masks = [cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > .5
             for mask in result.masks.data.cpu().numpy()]
    return _instances(masks, result.boxes.conf.cpu().numpy(), "yolo26")





def segment_seed_batch(crops, backend="yolo26", weights=None, imgsz=1024,
                       confidence=.25, device="cpu"):
    """Segment multiple dish crops in one model call while preserving crop coordinates."""
    crops = list(crops)
    if not crops:
        return []
    if backend == "classical":
        return [_classical(crop) for crop in crops]
    if backend != "yolo26":
        raise ValueError(f"Unknown segmentation backend: {backend}")
    if imgsz != 1024:
        raise ValueError("YOLO26s-seg seed inference requires imgsz=1024")
    if not weights or not Path(weights).is_file():
        raise FileNotFoundError("Supply a trained YOLO26s-seg seed checkpoint; use classical for P0")
    for crop in crops:
        if crop is None or crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError("Expected three-channel BGR dish crops")
    if str(device).lower() not in {"cpu", "none"}:
        _enable_rocm_cpu_nms_fallback()
    results = _load_model(str(Path(weights).resolve())).predict(
        crops, imgsz=1024, conf=confidence, retina_masks=True, verbose=False,
        device=device, batch=len(crops)
    )
    batches = []
    for crop, result in zip(crops, results):
        if result.masks is None:
            batches.append([])
            continue
        height, width = crop.shape[:2]
        masks = [
            cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > .5
            for mask in result.masks.data.cpu().numpy()
        ]
        batches.append(_instances(masks, result.boxes.conf.cpu().numpy(), "yolo26"))
    return batches
