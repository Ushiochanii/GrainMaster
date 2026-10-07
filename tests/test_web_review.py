import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from grainmaster.web_review import load_result, save_review, export_csv

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/prototype_p0/0462"


def load(path):
    return load_result(OUTPUT, image_id="0462", filename="0462.jpg", review_path=path,
                       image_url="/api/images/0462/original", corrected_url="/api/images/0462/asset/corrected")


@pytest.fixture
def result(tmp_path):
    if not OUTPUT.exists():
        pytest.skip("Real 0462 pipeline outputs required")
    return load(tmp_path / "review.json")


def test_real_adapter_coordinates_and_pending(result):
    assert result["backend"] == "classical"
    assert len(result["dishes"]) == 4
    assert [len(d["seeds"]) for d in result["dishes"]] == [12, 15, 3, 23]
    assert result["summary"] == dict(candidate_count=53, pending_count=53, confirmed_count=0, discarded_count=0)
    reasons = []
    for dish in result["dishes"]:
        x, y, w, h = dish["bbox"]
        assert dish["summary"]["confirmed"]["mean_L"] is None
        for seed in dish["seeds"]:
            p = np.array(seed["polygon"])
            assert np.all((p[:, 0] >= x) & (p[:, 0] < x + w))
            assert np.all((p[:, 1] >= y) & (p[:, 1] < y + h))
            assert seed["color_hex"].startswith("#")
            reasons.append(seed["ambiguity_reasons"])
    assert any(reasons) and any(not r for r in reasons)
    json.dumps(result, allow_nan=False)


def test_review_persistence_undo_and_color_means(tmp_path, result):
    path = tmp_path / "review.json"
    save_review(path, result, 1, 1, "confirmed")
    updated = load(path)
    assert updated["summary"]["confirmed_count"] == 1
    first = updated["dishes"][0]["seeds"][0]
    assert updated["dishes"][0]["summary"]["confirmed"]["mean_L"] == first["L"]
    original_mean = updated["dishes"][0]["summary"]["draft"]["mean_L"]
    save_review(path, updated, 1, 2, "discarded")
    updated = load(path)
    assert updated["summary"]["candidate_count"] == 52
    assert updated["dishes"][0]["summary"]["draft"]["mean_L"] != original_mean
    save_review(path, updated, 1, 1, "pending")
    updated = load(path)
    assert updated["dishes"][0]["summary"]["confirmed"]["seed_count"] == 0
    assert updated["dishes"][0]["summary"]["confirmed"]["color_hex"] is None
    save_review(path, updated, 1, 2, "pending")
    restored = load(path)
    assert restored["summary"] == result["summary"]
    assert restored["dishes"][0]["summary"]["draft"] == result["dishes"][0]["summary"]["draft"]
    assert json.loads(path.read_text())["revision"] == 4
    assert not list(tmp_path.glob("*.tmp"))


def test_review_validation_and_revision(tmp_path, result):
    path = tmp_path / "review.json"
    for args in [(1, 1, "yes"), (999, 1, "confirmed"), (1, 999, "confirmed")]:
        with pytest.raises(ValueError):
            save_review(path, result, *args)
    assert not path.exists()
    save_review(path, result, 1, 1, "confirmed")
    with pytest.raises(ValueError, match="revision"):
        save_review(path, result, 1, 2, "confirmed")


def csv_rows(text):
    assert text.startswith("\ufeff")
    return list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))


def test_exports_modes_and_equal_seed_color(tmp_path, result):
    assert len(csv_rows(export_csv(result, "seeds", "draft"))) == 53
    assert len(csv_rows(export_csv(result, "seeds", "confirmed"))) == 0
    empty_dishes = csv_rows(export_csv(result, "dishes", "confirmed"))
    assert len(empty_dishes) == 4
    assert all(r["seed_count"] == "0" and r["mean_L"] == "" for r in empty_dishes)
    path = tmp_path / "review.json"
    save_review(path, result, 1, 1, "confirmed")
    updated = load(path)
    save_review(path, updated, 1, 2, "confirmed")
    updated = load(path)
    save_review(path, updated, 1, 3, "discarded")
    updated = load(path)
    confirmed = csv_rows(export_csv(updated, "seeds", "confirmed"))
    assert len(confirmed) == 2
    assert all(r["label_source"] == "pseudo-label" and r["review_state"] == "confirmed" for r in confirmed)
    assert len(csv_rows(export_csv(updated, "seeds", "draft"))) == 52
    dish = csv_rows(export_csv(updated, "dishes", "confirmed"))[0]
    assert float(dish["mean_L"]) == pytest.approx(np.mean([s["L"] for s in updated["dishes"][0]["seeds"][:2]]))
    for args in [("foo", "draft"), ("seeds", "foo")]:
        with pytest.raises(ValueError):
            export_csv(updated, *args)


def test_shape_definitions_on_rectangle():
    from grainmaster.web_review import _shape_descriptors
    mask = np.zeros((50, 70), np.uint8)
    mask[10:31, 10:51] = 1  # External polygon is exactly 40 by 20 px.
    traits = _shape_descriptors(mask, 8, 4)
    assert traits["aspect_ratio"] == 2
    assert traits["solidity"] == 1
    assert traits["circularity"] == pytest.approx(4*np.pi*800/120**2)
    mask[10:21, 20:41] = 0
    assert _shape_descriptors(mask, 8, 4)["solidity"] < 0.9


def test_color_polar_and_included_reference(result):
    from grainmaster.web_review import _color_descriptors, _means
    assert _color_descriptors(3, 4) == pytest.approx((5, 53.130102))
    assert _color_descriptors(0, 0) == (0, None)
    assert _color_descriptors(1, -0.01)[1] > 359
    a, b = [dict(s) for s in result["dishes"][0]["seeds"][:2]]
    a.update(L=60, a=1, b=0.01)
    b.update(L=60, a=1, b=-0.01)
    summary = _means([a,b])
    assert summary["color_hue_deg"] == 0  # Avoid incorrect arithmetic mean of angles.
    assert summary["delta_e00_reference_lab"] == [60,1,0]
    assert len(summary["delta_e00_values"]) == 2
    assert summary["delta_e00_mean"] > 0
    assert _means([a])["delta_e00_mean"] is None
    assert _means([])["color_hue_deg"] is None
    identical = dict(a, key="duplicate")
    assert _means([a, identical])["delta_e00_mean"] == pytest.approx(0)


def test_new_traits_export_and_exclusion(tmp_path, result):
    rows = csv_rows(export_csv(result, "seeds", "draft"))
    assert all(float(r["aspect_ratio"]) > 0 for r in rows)
    assert all(float(r["chroma"]) >= 0 for r in rows)
    original = result["dishes"][0]["summary"]["draft"]
    path = tmp_path/"review.json"
    save_review(path, result, 1, 1, "discarded")
    updated = load(path)
    current = updated["dishes"][0]["summary"]["draft"]
    assert len(current["delta_e00_values"]) == len(original["delta_e00_values"])-1
    assert "1:1" not in [v["seed_key"] for v in current["delta_e00_values"]]
    assert current["delta_e00_reference_lab"] == pytest.approx(np.median(
        [[s["L"],s["a"],s["b"]] for s in updated["dishes"][0]["seeds"][1:]],axis=0))
