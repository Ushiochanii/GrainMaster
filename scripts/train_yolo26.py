"""Fixed YOLO26s-seg @ 1024 training; pretrained smoke is explicitly COCO only."""
import argparse
import json
import os
from pathlib import Path

import yaml


def validate_dataset(dataset):
    config = yaml.safe_load(dataset.read_text(encoding="utf-8"))
    if config.get("names") != {0: "seed"}:
        raise ValueError("Dataset must define exactly names: {0: seed}")
    metadata = json.loads((dataset.parent / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("label_source") != "human" or metadata.get("split_unit") != "whole_photo":
        raise ValueError("Training requires reviewed human labels split by whole photograph")
    records = metadata.get("records", [])
    if any(not any(record["split"] == split and record["instances"] > 0
                   for record in records) for split in ("train", "val")):
        raise ValueError("Both train and val require human seed instances")
    splits = {}
    for record in records:
        if record.get("label_source") != "human":
            raise ValueError("Every dataset record must be human reviewed")
        if record["split"] == "review":
            continue  # Quarantined crops are outside every training/evaluation split.
        if record["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown dataset split")
        splits.setdefault(record["source_photo"], set()).add(record["split"])
    if any(len(values) != 1 for values in splits.values()):
        raise ValueError("Whole-photo leakage across dataset splits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/seed_yolo/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", metavar="DISH_CROP")
    parser.add_argument("--output", default="artifacts/models")
    parser.add_argument("--check-only", action="store_true", help="Validate data without loading or training a model")
    parser.add_argument("--plan-config", type=Path, help="Optional reviewed training settings YAML")
    parser.add_argument("--name", default="seed_yolo26s_seg")
    parser.add_argument("--resume", type=Path, help="Resume a saved training checkpoint")
    parser.add_argument("--init-weights", type=Path, help="Warm-start a separate smoke run without resuming epochs")
    parser.add_argument("--memory-safe", action="store_true", help="Chunk mask loss and recompute backward intermediates")
    args = parser.parse_args()
    dataset = Path(args.data).resolve()
    if not args.smoke:
        validate_dataset(dataset)
    if args.check_only:
        print(f"Dataset validation passed: {dataset}; no model loaded or training started")
        return
    import ultralytics
    from ultralytics import YOLO
    from seed_trainer import SeedTrainer
    if args.memory_safe:
        from seed_memory_loss import enable_memory_loss
        enable_memory_loss()
        print('Memory-safe mask loss enabled: chunk=8, gradient checkpointing', flush=True)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and args.init_weights:
        raise ValueError("Choose resume or warm-start, not both")
    weights = args.init_weights.resolve(strict=True) if args.init_weights else output / "yolo26s-seg.pt"
    # Ultralytics downloads missing filenames into cwd; keep large weights in artifacts.
    previous = Path.cwd()
    if args.resume:
        checkpoint = args.resume.resolve(strict=True)
        model = YOLO(str(checkpoint))
        if model.task != "segment" or model.names != {0: "seed"}:
            raise ValueError("Resume requires a seed segmentation checkpoint")
    elif not weights.exists():
        os.chdir(output)
        try:
            model = YOLO("yolo26s-seg.pt")
        finally:
            os.chdir(previous)
    else:
        model = YOLO(str(weights))
    if args.smoke:
        results = model.predict(str(Path(args.smoke).resolve()), imgsz=1024,
                                device=args.device, verbose=False)
        results[0].save(filename=str(output / "coco_smoke.jpg"))
        report = dict(ultralytics_version=ultralytics.__version__, model="yolo26s-seg",
                      imgsz=1024, checkpoint_class_count=len(model.names),
                      architecture_yaml=model.model.yaml.get("yaml_file"),
                      architecture_scale=model.model.yaml.get("scale"),
                      checkpoint_training_imgsz=model.ckpt.get("train_args", {}).get("imgsz"),
                      task=model.task, detections=len(results[0].boxes),
                      purpose="COCO pretrained loading/inference smoke; no trained seed weights")
        (output / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    settings = yaml.safe_load(args.plan_config.read_text(encoding="utf-8")) if args.plan_config else {}
    protected = {"model", "data", "imgsz", "project", "name"}
    if protected & settings.keys():
        raise ValueError("Plan config cannot override model, dataset, resolution or output paths")
    kwargs = dict(epochs=args.epochs, batch=args.batch, device=args.device, workers=0)
    kwargs.update(settings)
    if args.resume:
        kwargs['resume'] = str(checkpoint)
    def record_runtime(trainer):
        actual = dict(amp_enabled=bool(trainer.amp), amp_mode=trainer.args.amp,
                      batch=int(trainer.batch_size), workers=int(trainer.args.workers), memory_safe=args.memory_safe)
        (trainer.save_dir / "runtime_settings.json").write_text(
            json.dumps(actual, indent=2), encoding="utf-8")
        print(f"Effective training settings: {actual}", flush=True)
        if settings.get("amp") and not trainer.amp:
            raise RuntimeError("Requested AMP was disabled; refusing silent FP32 fallback")

    def check_finite_loss(trainer):
        import torch
        items = trainer.loss_items
        values = items.values() if isinstance(items, dict) else [items]
        if not all(torch.isfinite(torch.as_tensor(value)).all().item() for value in values):
            raise RuntimeError("Non-finite training loss; stopping mixed-precision run")

    model.add_callback("on_train_start", record_runtime)
    model.add_callback("on_train_end", record_runtime)
    model.add_callback("on_train_batch_end", check_finite_loss)
    model.train(trainer=SeedTrainer, data=str(dataset), imgsz=1024, project=str(output / "training"),
                name=args.name, **kwargs)


if __name__ == "__main__":
    main()
