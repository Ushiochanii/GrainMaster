"""Convert human full-photo LabelMe polygons, or export separate pseudo candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grainmaster.dishes import detect_dishes, crop_dish
from grainmaster.segmentation import segment_seeds


def prepare(raw, annotations, output, pseudo=False):
    raw, annotations, output = map(Path, (raw, annotations, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to avoid mixing label provenance")
    records = []
    if pseudo:
        for photo in sorted(raw.glob("*.jpg")):
            records.append((photo, None))
    else:
        for annotation in sorted(annotations.glob("*.json")):
            record = json.loads(annotation.read_text(encoding="utf-8"))
            if record.get("label_source") == "pseudo-label" or record.get("pseudo_label"):
                raise ValueError(f"Pseudo-label cannot be imported as human GT: {annotation}")
            photo = raw / Path(record.get("imagePath", annotation.with_suffix(".jpg").name)).name
            shapes = record.get("shapes", [])
            if any(s.get("label") != "seed" or s.get("shape_type", "polygon") != "polygon"
                   for s in shapes):
                raise ValueError(f"Only seed polygons are supported: {annotation}")
            records.append((photo, shapes))
    if not records:
        raise ValueError("No human annotations found; annotate seed polygons or use --pseudo")
    ordered = sorted(records, key=lambda pair: hashlib.sha256(pair[0].stem.encode()).hexdigest())
    validation = {photo.stem for photo, _ in ordered[:max(1, round(len(records) * .2))]}
    if not pseudo and len(records) < 2:
        raise ValueError("At least two annotated whole photos are needed for train/val separation")
    manifest = []
    for photo, shapes in records:
        image = cv2.imread(str(photo))
        if image is None:
            raise ValueError(f"Cannot read {photo}")
        split = "candidates" if pseudo else "val" if photo.stem in validation else "train"
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        full_masks = []
        if not pseudo:
            for shape in shapes:
                mask = np.zeros(image.shape[:2], np.uint8)
                points = np.asarray(shape["points"], dtype=np.float32)
                if len(points) < 3 or not np.isfinite(points).all():
                    raise ValueError(f"Invalid polygon in {photo}")
                cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
                full_masks.append(mask)
        assigned = [0] * len(full_masks)
        for dish in detect_dishes(image):
            crop = crop_dish(image, dish)
            x, y, width, height = dish.bbox
            masks = [s.mask for s in segment_seeds(crop, backend="classical")] if pseudo else [
                mask[y:y + height, x:x + width] for mask in full_masks
                if np.count_nonzero(mask[y:y + height, x:x + width]) >= .95 * mask.sum()]
            if not pseudo:
                for index, mask in enumerate(full_masks):
                    if np.count_nonzero(mask[y:y + height, x:x + width]) >= .95 * mask.sum():
                        assigned[index] += 1
            labels = []
            for mask in masks:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                polygon = cv2.approxPolyDP(contour, .001 * cv2.arcLength(contour, True), True)
                points = polygon[:, 0, :].astype(float) / [width, height]
                if len(points) >= 3:
                    labels.append("0 " + " ".join(f"{v:.7f}" for v in points.ravel()))
            stem = f"{photo.stem}_dish{dish.dish_id:02d}"
            cv2.imwrite(str(image_dir / f"{stem}.jpg"), crop)
            (label_dir / f"{stem}.txt").write_text("\n".join(labels), encoding="utf-8")
            manifest.append(dict(image=stem, source_photo=photo.name, split=split,
                                 instances=len(labels), label_source="pseudo-label" if pseudo
                                 else "human", crop_bbox_xywh=dish.bbox))
        if not pseudo and any(count != 1 for count in assigned):
            raise ValueError(f"Every seed polygon must be assigned to exactly one dish: {photo}")
    if not manifest:
        raise ValueError("No dish crops found")
    if not pseudo and any(not any(row["split"] == split and row["instances"] > 0
                                  for row in manifest) for split in ("train", "val")):
        raise ValueError("Both train and val require human seed instances")
    metadata = dict(label_source="pseudo-label" if pseudo else "human",
                    requires_human_review=pseudo, split_unit="whole_photo", records=manifest)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not pseudo:
        (output / "dataset.yaml").write_text(yaml.safe_dump(dict(
            path=str(output.resolve()), train="images/train", val="images/val", names={0: "seed"})),
            encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--annotations", default="data/annotations")
    parser.add_argument("--output", default=None)
    parser.add_argument("--pseudo", action="store_true")
    args = parser.parse_args()
    output = args.output or ("data/processed/pseudo_candidates" if args.pseudo else
                             "data/processed/seed_yolo")
    print(json.dumps(prepare(args.raw, args.annotations, output, args.pseudo), indent=2))
