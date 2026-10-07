"""Read immutable pipeline outputs; persist only human review decisions."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import tempfile

import cv2
import colour
import numpy as np
from PIL import Image

STATES = {"pending", "confirmed", "discarded"}
TRAITS = ("length_mm", "width_mm", "area_mm2", "L", "a", "b")
SHAPE_TRAITS = ("aspect_ratio", "circularity", "solidity")
COLOR_TRAITS = ("chroma", "hue_deg")
COLOR_METHOD = "Per-seed color is the median of corrected pixels inside the mask; dish color is the equal-weight mean of seed Lab values (D65)."


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _reviews(path):
    return _read(path) if Path(path).exists() else {"revision": 0, "states": {}}


def _hex(L, a, b):
    if L is None:
        return None
    lab = np.array([[[L, a, b]]], dtype=np.float32)
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)[0, 0]
    values = np.clip(np.rint(rgb * 255), 0, 255).astype(int)
    return "#" + "".join(f"{v:02x}" for v in values)


def _color_descriptors(a, b):
    if a is None:
        return None, None
    chroma = float(np.hypot(a, b))
    return chroma, float(np.degrees(np.arctan2(b, a)) % 360) if chroma > 1e-6 else None


def _shape_descriptors(mask, length, width):
    """Shape ratios use external contour polygon area, not raster pixel area.

    This keeps area/perimeter/convex-hull ratios geometrically consistent.
    The existing area_mm2 remains the full raster mask area.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = sum(cv2.contourArea(c) for c in contours)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    hull = cv2.convexHull(np.concatenate(contours)) if contours else None
    hull_area = cv2.contourArea(hull) if hull is not None else 0
    return {
        "aspect_ratio": length / width if width > 0 else None,
        "circularity": float(4 * np.pi * area / perimeter**2) if perimeter > 0 else None,
        "solidity": float(area / hull_area) if hull_area > 0 else None,
    }


def _color_distances(seeds):
    """Within-dish distances to componentwise median of INCLUDED seed Lab.

    Relative consistency only; no biological or measurement-error threshold.
    A singleton has no meaningful between-seed distribution.
    """
    if len(seeds) < 2:
        return None, {}
    labs = np.array([[s["L"], s["a"], s["b"]] for s in seeds])
    reference = np.median(labs, axis=0)
    distances = colour.delta_E(labs, reference, method="CIE 2000")
    return reference.tolist(), {s["key"]: float(v) for s, v in zip(seeds, distances, strict=True)}


def _means(seeds):
    result = {"seed_count": len(seeds)}
    for key in TRAITS + SHAPE_TRAITS:
        values = [s[key] for s in seeds if s.get(key) is not None]
        result["mean_" + key] = float(np.mean(values)) if values else None
    result["color_hex"] = _hex(result["mean_L"], result["mean_a"], result["mean_b"])
    result["color_chroma"], result["color_hue_deg"] = _color_descriptors(result["mean_a"], result["mean_b"])
    reference, distances = _color_distances(seeds)
    result["delta_e00_reference_lab"] = reference
    result["delta_e00_mean"] = float(np.mean(list(distances.values()))) if distances else None
    result["delta_e00_p90"] = float(np.percentile(list(distances.values()), 90)) if distances else None
    result["delta_e00_values"] = [{"seed_key": k, "delta_e00": v} for k, v in distances.items()]
    return result


def _counts(seeds):
    counts = {state + "_count": sum(s["review_state"] == state for s in seeds)
              for state in STATES}
    counts["candidate_count"] = len(seeds) - counts["discarded_count"]
    return counts


def _display_crop_with_label(image_bgr: np.ndarray, bbox) -> tuple[list[int], list[int] | None]:
    """Expand a dish-only crop to include the nearby paper sample label.

    Presentation only. The original dish bbox remains unchanged for segmentation,
    masks, and measurements.
    """
    image_h, image_w = image_bgr.shape[:2]
    x, y, w, h = map(int, bbox)
    search_x = max(0, int(x - 0.15 * w))
    search_w = min(image_w - search_x, int(1.30 * w))
    search_y = max(0, int(y + 0.97 * h))
    search_bottom = min(image_h, int(y + 1.45 * h))
    roi = image_bgr[search_y:search_bottom, search_x:search_x + search_w]
    if roi.size == 0:
        return [x, y, w, h], None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    cutoff = max(110, int(np.percentile(value, 85)))
    mask = ((value >= cutoff) & (saturation <= 115)).astype(np.uint8) * 255
    kernel_x = max(5, int(w * 0.025))
    kernel_y = max(3, int(h * 0.010))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_x, kernel_y)),
        iterations=2,
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dish_cx = x + w / 2
    dish_bottom = y + h
    best = None
    for contour in contours:
        rx, ry, rw, rh = cv2.boundingRect(contour)
        gx, gy = rx + search_x, ry + search_y
        area = cv2.contourArea(contour)
        aspect = rw / max(rh, 1)
        if not (0.22 * w <= rw <= 0.95 * w and
                0.05 * h <= rh <= 0.38 * h and
                1.3 <= aspect <= 7.5):
            continue
        if area < rw * rh * 0.28:
            continue
        label_cx = gx + rw / 2
        gap = max(0, gy - dish_bottom)
        if abs(label_cx - dish_cx) > 0.38 * w or gap > 0.45 * h:
            continue
        score = abs(label_cx - dish_cx) / w + gap / h * 0.8 + abs(rw / w - 0.48) * 0.3
        candidate = (score, [gx, gy, rw, rh])
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        return [x, y, w, h], None

    label_bbox = best[1]
    lx, ly, lw, lh = label_bbox
    margin = max(12, int(round(w * 0.025)))
    left = max(0, min(x, lx) - margin)
    top = max(0, y - margin)
    right = min(image_w, max(x + w, lx + lw) + margin)
    bottom = min(image_h, max(y + h, ly + lh) + margin)
    return [left, top, right - left, bottom - top], label_bbox


def load_result(output_dir: Path, *, image_id: str, filename: str, review_path: Path,
                image_url: str, corrected_url: str, original_path: Path) -> dict:
    """Use original pixel coordinates, never regenerate measurements on review."""
    out = Path(output_dir)
    calibration = _read(out / "calibration.json")
    instances = _read(out / "instances.json")
    backend = instances["backend"]
    records = {(r["dish_id"], r["seed_id"]): r for r in instances["instances"]}
    review = _reviews(review_path)
    paper_file = out / 'sample_labels.json'
    paper_labels = {r['dish_id']: r for r in (_read(paper_file).get('labels', []) if paper_file.exists() else [])}
    with (out / "seed_instances.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with Image.open(original_path) as image:
        width, height = image.size
    display_image = cv2.imread(str(original_path))
    if display_image is None:
        raise ValueError("Missing original image for review display")
    warnings = ["The long ruler provides the millimeter scale; absolute measurement accuracy has not yet been independently validated.",
                "Color-chart fit quality does not guarantee full-image color accuracy; chart-version matching has not yet been independently verified."]
    if backend == "classical":
        warnings.append("The current result uses classical pre-segmentation; 0.5 is a heuristic score, not a seed-detection probability.")
    correction = calibration.get("spatial_correction")
    corrected_spatial = bool(correction and correction.get("applied_to_measurements"))
    if corrected_spatial:
        warnings.append("Planar spatial correction using ruler and shape constraints has been applied; this is not equivalent to an independent camera calibration.")
        if not correction.get("accepted"):
            warnings.append("The spatial-correction internal shape check shows residual disagreement; inspect the corrected image.")
    if not corrected_spatial and any("spatial_reference_disagreement" in r["qc_flag"] for r in rows):
        warnings.append("The ruler and chart-scale estimates disagree; the ruler remains the scale reference and no full-image distortion correction was applied.")
    spatial, color = dict(calibration["spatial"]), calibration["color"]
    if corrected_spatial:
        spatial["pixels_per_mm"] = correction["after"]["pixels_per_mm"]
        spatial["mm_per_pixel"] = 1 / spatial["pixels_per_mm"]
    result = dict(image_id=image_id, filename=filename, width=width, height=height,
                  backend=backend, revision=review.get("revision", 0), image_url=image_url,
                  corrected_url=corrected_url + "?v=" + str((out / "calibration.json").stat().st_mtime_ns), color_method=COLOR_METHOD, dishes=[],
                  calibration=dict(pixels_per_mm=spatial["pixels_per_mm"],
                                   mm_per_pixel=spatial["mm_per_pixel"], scale_source="long_ruler",
                                   spatial_correction_applied=corrected_spatial,
                                   color_space="CIELAB D65",
                                   color_model=color.get("metadata", {}).get("model", "affine_baseline"),
                                   delta_e_before=color["mean_delta_e00_before"],
                                   delta_e_after=color["mean_delta_e00_after"], warnings=warnings))
    spatial_file = out / "spatial_preview.json"
    result["spatial_preview"] = None
    if spatial_file.exists():
        result["spatial_preview"] = _read(spatial_file)
        result["spatial_preview"]["url"] = f"/api/images/{image_id}/asset/spatial?v={spatial_file.stat().st_mtime_ns}"
        result["spatial_preview"]["color_url"] = f"/api/images/{image_id}/asset/spatial_color?v={spatial_file.stat().st_mtime_ns}"
    for dish in _read(out / "dishes.json"):
        dish_id = dish["dish_id"]
        x, y, w, h = dish["bbox"]
        labels = cv2.imread(str(out / "masks" / f"dish_{dish_id:02d}.png"), cv2.IMREAD_UNCHANGED)
        if labels is None or labels.shape != (h, w):
            raise ValueError(f"Missing or mismatched mask for dish {dish_id}")
        dish_rows = [r for r in rows if int(r["dish_id"]) == dish_id]
        median_area = float(np.median([float(r["area_mm2"]) for r in dish_rows])) if dish_rows else 0
        seeds = []
        for row in dish_rows:
            seed_id = int(row["seed_id"])
            independent = out / "masks" / f"dish_{dish_id:02d}_seed_{seed_id:03d}.png"
            mask = cv2.imread(str(independent), cv2.IMREAD_GRAYSCALE) if independent.exists() else (labels == seed_id).astype(np.uint8)
            if mask is None or mask.shape != (h, w):
                raise ValueError(f"Missing independent mask for {dish_id}:{seed_id}")
            mask = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise ValueError(f"Mask missing for seed {dish_id}:{seed_id}")
            contour = max(contours, key=cv2.contourArea)
            # Retain disjoint outlines too, so reviewed overlays display the entire mask.
            polygons = [(cv2.approxPolyDP(c, 0.65, True).reshape(-1, 2) + [x, y]).tolist()
                        for c in contours]
            bx, by, bw, bh = cv2.boundingRect(cv2.findNonZero(mask))
            seed = {k: float(row[k]) for k in TRAITS + ("seg_confidence",)}
            shape = records.get((dish_id, seed_id), {}).get("shape_descriptors")
            seed.update(shape or _shape_descriptors(mask, seed["length_mm"], seed["width_mm"]))
            seed["chroma"], seed["hue_deg"] = _color_descriptors(seed["a"], seed["b"])
            key = f"{dish_id}:{seed_id}"
            seed.update(dish_id=dish_id, seed_id=seed_id, key=key, label=f"S{seed_id:02d}",
                        bbox=[x + bx, y + by, bw, bh],
                        polygon=(cv2.approxPolyDP(contour, 0.65, True).reshape(-1, 2) + [x, y]).tolist(),
                        polygons=polygons, qc_flag=row["qc_flag"],
                        review_state=review.get("states", {}).get(key, "pending"),
                        color_hex=_hex(seed["L"], seed["a"], seed["b"]),
                        provenance=records.get((dish_id, seed_id), {}).get("metadata", {}))
            reasons = []
            if median_area and seed["area_mm2"] > median_area * 1.65:
                reasons.append("Area is substantially above the dish median; the instance may contain touching or merged seeds.")
            if median_area and seed["area_mm2"] < median_area * 0.5:
                reasons.append("Area is substantially below the dish median; the instance may be a fragment or an incomplete segmentation.")
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area and cv2.contourArea(contour) / hull_area < 0.85:
                reasons.append("The contour is strongly concave, suggesting touching seeds or incomplete segmentation.")
            if sum(cv2.contourArea(c) > max(5, cv2.contourArea(contour) * 0.05) for c in contours) > 1:
                reasons.append("The instance contains multiple disconnected regions.")
            if bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1:
                reasons.append("The instance touches the crop boundary and may be truncated.")
            if backend != "classical" and seed["seg_confidence"] < 0.5:
                reasons.append("Model confidence is low; manual review is recommended.")
            seed["ambiguity_reasons"] = reasons
            if seed["review_state"] not in STATES:
                raise ValueError(f"Invalid saved review state for {key}")
            seeds.append(seed)
        summary = _counts(seeds)
        summary["draft"] = _means([s for s in seeds if s["review_state"] != "discarded"])
        summary["confirmed"] = _means([s for s in seeds if s["review_state"] == "confirmed"])
        display_bbox, label_bbox = _display_crop_with_label(display_image, dish["bbox"])
        paper = paper_labels.get(dish_id, {})
        manual = review.get('sample_labels', {}).get(str(dish_id))
        sample_id = manual['sample_id'] if manual else paper.get('sample_id') if paper.get('status') == 'proposed' else None
        label_status = 'confirmed' if manual else paper.get('status', 'not_read')
        result["dishes"].append(dict(sample_id=sample_id, sample_id_candidate=paper.get('sample_id'), sample_label_status=label_status, paper_label=paper, dish_id=dish_id, label=f"D{10 + dish_id}", bbox=dish["bbox"],
                                       display_bbox=display_bbox, label_bbox=label_bbox,
                                       center=dish["center"], seeds=seeds, summary=summary,
                                       thumbnail_url=f"/api/images/{image_id}/dish/{dish_id}"))
    result["summary"] = _counts([s for d in result["dishes"] for s in d["seeds"]])
    return result


def save_review(review_path: Path, result: dict, dish_id: int, seed_id: int, state: str) -> None:
    """Atomic replace. Caller serializes concurrent writes; masks and CSV remain immutable."""
    if state not in STATES:
        raise ValueError("state must be pending, confirmed, or discarded")
    if not any(d["dish_id"] == dish_id and any(s["seed_id"] == seed_id for s in d["seeds"])
               for d in result["dishes"]):
        raise ValueError("Unknown dish or seed")
    path = Path(review_path)
    current = _reviews(path)
    if current.get("revision", 0) != result.get("revision", 0):
        raise ValueError("Review revision changed; reload before saving")
    current.setdefault("states", {})[f"{dish_id}:{seed_id}"] = state
    current.update(revision=current.get("revision", 0) + 1, image_id=result["image_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="review_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(current, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_csv(result: dict, kind: str = "seeds", mode: str = "confirmed", precision: int = 2) -> str:
    """Export a compact analysis-ready CSV.

    The CSV intentionally omits repeated implementation/provenance prose.
    Detailed calibration and method metadata remain available in the result JSON/UI.
    """
    if kind not in {"seeds", "dishes"} or mode not in {"confirmed", "draft"}:
        raise ValueError("Invalid export kind or mode")
    if precision not in {2, 3, 4}:
        raise ValueError("Invalid export precision")

    rows = []
    if kind == "seeds":
        columns = [
            "image_id", "dish_id", "dish_label", "sample_id", "seed_id",
            "length_mm", "width_mm", "area_mm2",
            "aspect_ratio", "circularity", "solidity",
            "L", "a", "b", "chroma", "hue_deg",
            "seg_confidence", "qc_flag", "review_state",
        ]
        for dish in result["dishes"]:
            for seed in dish["seeds"]:
                if seed["review_state"] == "discarded":
                    continue
                if mode == "confirmed" and seed["review_state"] != "confirmed":
                    continue
                row = {key: seed.get(key) for key in columns}
                row["review_state"] = {"pending": "Auto", "confirmed": "Checked", "discarded": "Excluded"}.get(seed.get("review_state"), seed.get("review_state"))
                row.update(
                    image_id=result["image_id"],
                    dish_id=dish["dish_id"],
                    dish_label=dish["label"],
                    sample_id=dish.get("sample_id") or dish.get("sample_id_candidate") or "",
                )
                rows.append(row)
    else:
        columns = [
            "image_id", "dish_id", "dish_label", "sample_id", "seed_count",
            "mean_length_mm", "mean_width_mm", "mean_area_mm2",
            "mean_aspect_ratio", "mean_circularity", "mean_solidity",
            "mean_L", "mean_a", "mean_b", "color_chroma", "color_hue_deg",
            "delta_e00_mean", "delta_e00_p90",
            "confirmed_count", "pending_count", "discarded_count",
        ]
        for dish in result["dishes"]:
            included = [
                seed for seed in dish["seeds"]
                if seed["review_state"] == "confirmed"
                or (mode == "draft" and seed["review_state"] == "pending")
            ]
            summary = _means(included)
            counts = _counts(dish["seeds"])
            row = {
                "image_id": result["image_id"],
                "dish_id": dish["dish_id"],
                "dish_label": dish["label"],
                "sample_id": dish.get("sample_id") or dish.get("sample_id_candidate") or "",
                **summary,
                "confirmed_count": counts["confirmed_count"],
                "pending_count": counts["pending_count"],
                "discarded_count": counts["discarded_count"],
            }
            rows.append(row)

    for row in rows:
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = f"{value:.{precision}f}"

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + stream.getvalue()


def save_sample_id(review_path, result, dish_id, sample_id):
    if not any(d['dish_id'] == dish_id for d in result['dishes']):
        raise ValueError('Unknown dish')
    sample_id = sample_id.strip()
    if not sample_id or len(sample_id) > 128 or any(ord(c) < 32 for c in sample_id):
        raise ValueError('Enter a sample identifier of 1–128 characters.')
    path = Path(review_path)
    current = _reviews(path)
    current.setdefault('sample_labels', {})[str(dish_id)] = dict(sample_id=sample_id, source='human', status='confirmed')
    current['revision'] = current.get('revision', 0) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='sample_id_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(current, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
