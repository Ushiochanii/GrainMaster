import cv2
import numpy as np
import pytest
import importlib.util
import json
from pathlib import Path
import yaml

from grainmaster.segmentation import segment_seeds, _instances, _clean_mask_fragments


@pytest.mark.parametrize("backend", ["classical", "yolo26"])
def test_detached_tiny_fragments_removed_before_instance_geometry(backend):
    mask = np.zeros((120, 140), bool)
    mask[20:70, 30:70] = True  # 2000 px main region; threshold 20 px.
    mask[3:5, 3:5] = True
    mask[90:93, 120:123] = True
    seed = _instances([mask], [.9], backend)[0]
    assert seed.mask.sum() == 2000
    assert seed.bbox == (30, 20, 40, 50)
    assert seed.metadata["fragment_removed_pixels"] == 13
    assert seed.metadata["fragment_removed_components"] == 2
    assert seed.metadata["mask_cleanup_version"] == "largest-component-v2"


def test_disconnected_part_removed_by_largest_component_cleanup():
    mask = np.zeros((120, 140), bool)
    mask[20:70, 30:70] = True
    mask[90:96, 120:126] = True
    seed = _instances([mask], [.9], "yolo26")[0]
    assert seed.mask.sum() == 2000
    assert seed.bbox == (30, 20, 40, 50)
    assert seed.metadata["fragment_removed_components"] == 1
    assert seed.metadata["fragment_removed_pixels"] == 36


def test_diagonal_connectivity_kept_but_detached_component_removed():
    mask = np.zeros((160, 160), bool)
    mask[20:120, 20:120] = True
    mask[19, 19] = True  # 8-connected to the main component.
    mask[140:147, 140:150] = True
    cleaned, metadata = _clean_mask_fragments(mask)
    assert cleaned.sum() == 10001
    assert cleaned[19, 19]
    assert not cleaned[140:147, 140:150].any()
    assert metadata["fragment_removed_components"] == 1
    assert metadata["fragment_removed_pixels"] == 70


def test_cleanup_empty_and_single_small_component_safe():
    mask = np.zeros((10, 10), bool)
    assert _instances([mask], [.9], "yolo26") == []
    mask[2, 3] = True
    cleaned, metadata = _clean_mask_fragments(mask)
    assert np.array_equal(cleaned, mask)
    assert metadata["fragment_removed_pixels"] == 0


def test_classical_separate_seeds_and_exclude_rim():
    crop = np.full((400, 400, 3), (160, 30, 30), np.uint8)
    cv2.circle(crop, (200, 200), 190, (70, 220, 230), 8)
    cv2.ellipse(crop, (150, 150), (22, 12), 30, 0, 360, (30, 160, 210), -1)
    cv2.ellipse(crop, (250, 230), (23, 12), -20, 0, 360, (30, 160, 210), -1)
    seeds = segment_seeds(crop, backend="classical")
    assert len(seeds) == 2
    assert all(s.mask.dtype == bool and s.mask.shape == crop.shape[:2] for s in seeds)
    assert all(s.metadata["label_source"] == "pseudo-label" for s in seeds)
    assert not (seeds[0].mask & seeds[1].mask).any()
    assert [s.seed_id for s in seeds] == [1, 2]


def test_empty_blue_dish():
    assert segment_seeds(np.full((300, 300, 3), (160, 30, 30), np.uint8),
                         backend="classical") == []


def test_touching_seeds_are_split_with_isolated_seed_intact():
    crop = np.full((400, 400, 3), (160, 30, 30), np.uint8)
    for center in ((155, 200), (190, 200), (280, 130)):
        cv2.ellipse(crop, center, (21, 13), 0, 0, 360, (30, 160, 210), -1)
    seeds = segment_seeds(crop, backend="classical")
    assert len(seeds) == 3
    assert all(s.mask.sum() > 650 for s in seeds)


def test_production_backend_requires_trained_seed_weights():
    crop = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(FileNotFoundError, match="trained"):
        segment_seeds(crop)
    with pytest.raises(ValueError, match="1024"):
        segment_seeds(crop, imgsz=640)
    with pytest.raises(ValueError, match="Unknown"):
        segment_seeds(crop, backend="made_up")


def _script(name):
    spec = importlib.util.spec_from_file_location(name,
                Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_rejects_pseudo_as_human_and_stale_output(tmp_path):
    prepare = _script("prepare_seed_annotations").prepare
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "a.json").write_text(json.dumps({"label_source": "pseudo-label"}))
    with pytest.raises(ValueError, match="Pseudo-label"):
        prepare(tmp_path, annotations, tmp_path / "out")
    output = tmp_path / "out"
    output.mkdir()
    (output / "stale.txt").write_text("stale")
    with pytest.raises(ValueError, match="empty"):
        prepare(tmp_path, annotations, output)


def test_training_rejects_leakage_pseudo_and_empty_splits(tmp_path):
    validate = _script("train_yolo26").validate_dataset
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(yaml.safe_dump({"names": {0: "seed"}}))
    metadata = {"label_source": "human", "split_unit": "whole_photo", "records": [
        {"source_photo": "a.jpg", "split": split, "instances": 2, "label_source": "human"}
        for split in ("train", "val")]}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="leakage"):
        validate(dataset)
    metadata["label_source"] = "pseudo-label"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="human"):
        validate(dataset)
    metadata.update(label_source="human", records=[])
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Both"):
        validate(dataset)
