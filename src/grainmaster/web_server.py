"""Local web workbench over the existing P0 pipeline and persistent reviews."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Annotated
from uuid import uuid4
from time import time

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipeline import PipelineError, run_pipeline

ROOT = Path(__file__).resolve().parents[2]
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class ProcessRequest(BaseModel):
    image_id: str
    force: bool = False


class ReviewRequest(BaseModel):
    dish_id: int = Field(gt=0)
    seed_id: int = Field(gt=0)
    state: str


class SampleIDRequest(BaseModel):
    dish_id: int = Field(gt=0)
    sample_id: str = Field(min_length=1, max_length=128)


class SettingsRequest(BaseModel):
    settings: dict
    api_key: str | None = None
    clear_api_key: bool = False


class IntegrationTestRequest(BaseModel):
    provider: str = 'deepseek'
    base_url: str
    model: str
    api_key: str | None = None


def create_app(project_root=ROOT, config_path=None, state_root=None):
    project = Path(project_root).resolve()
    config = Path(config_path or project / "configs/prototype.yaml")
    workbench_config = yaml.safe_load(config.read_text(encoding="utf-8"))
    selected_backend = workbench_config.get("segmentation", {}).get("backend", "classical")
    spatial_required = workbench_config.get("spatial_correction", {}).get("enabled", False)
    state = Path(state_root or project / "artifacts/web_ui").resolve()
    state.mkdir(parents=True, exist_ok=True)
    uploads = project / "data/processed/web_uploads"
    runs = state / "runs"
    reviews = state / "reviews"
    runs.mkdir(parents=True, exist_ok=True)
    reviews.mkdir(parents=True, exist_ok=True)
    lock = RLock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grainmaster")
    registry = {}
    jobs = {}
    label_jobs = {}
    label_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='paper-labels')
    runtime_config = state / 'runtime_config.yaml'

    def current_settings():
        from .settings import load_settings
        return load_settings(state, workbench_config)

    def write_runtime_config():
        import copy
        cfg = copy.deepcopy(workbench_config)
        prefs = current_settings()
        cfg.setdefault('model', {})['device'] = 'cuda' if prefs['analysis']['device'] == 'gpu' else 'cpu'
        cfg.setdefault('model', {})['imgsz'] = prefs['analysis']['imgsz']
        runtime_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        return runtime_config

    @asynccontextmanager
    async def lifespan(app):
        if selected_backend == "yolo26":
            segmentation = workbench_config.get("segmentation", {})
            model_cfg = workbench_config.get("model", {})
            weights = segmentation.get("weights")
            if weights:
                weights_path = Path(weights)
                if not weights_path.is_absolute():
                    weights_path = (config.parent.parent / weights_path).resolve()
                from .segmentation import segment_seed_batch
                dummy = np.zeros((1024, 1024, 3), dtype=np.uint8)
                segment_seed_batch(
                    [dummy], backend="yolo26", weights=str(weights_path),
                    imgsz=int(model_cfg.get("imgsz", 1024)),
                    confidence=float(model_cfg.get("confidence", .25)),
                    device=model_cfg.get("device", "cpu"),
                )
        yield
        executor.shutdown(wait=False, cancel_futures=True)
        label_executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="GrainMaster Mozume", lifespan=lifespan)
    app.state.registry = registry
    app.state.jobs = jobs
    app.state.review_root = reviews

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and request.method not in {"GET", "HEAD", "OPTIONS"}:
            from urllib.parse import urlparse

            hostname = urlparse(origin).hostname
            trusted = hostname in {"127.0.0.1", "localhost", "::1"}
            if hostname and not trusted:
                try:
                    from ipaddress import ip_address, ip_network
                    trusted = ip_address(hostname) in ip_network("100.64.0.0/10")
                except ValueError:
                    trusted = False
            if not trusted:
                return Response("Trusted local or Tailscale access only", status_code=403)
        return await call_next(request)

    def discover():
        with lock:
            example = project / "web/assets/example-photo.jpg"
            if example.exists():
                registry.setdefault("example", {"path": example, "filename": "Example photo.jpg", "is_example": True})
            for path in sorted((project / "data/raw").glob("*")):
                if path.suffix.lower() in IMAGE_EXTENSIONS and SAFE_ID.fullmatch(path.stem):
                    registry.setdefault(path.stem, {"path": path, "filename": path.name})
            if uploads.exists():
                for path in sorted(uploads.glob("*")):
                    if path.suffix.lower() in IMAGE_EXTENSIONS and SAFE_ID.fullmatch(path.stem):
                        metadata = path.with_suffix(".json")
                        filename = (
                            json.loads(metadata.read_text(encoding="utf-8")).get(
                                "filename", path.name
                            )
                            if metadata.exists()
                            else path.name
                        )
                        registry.setdefault(path.stem, {"path": path, "filename": filename})

    def image_record(image_id):
        if not SAFE_ID.fullmatch(image_id):
            raise HTTPException(404, "Photo not found")
        discover()
        if image_id not in registry:
            raise HTTPException(404, "Photo not found")
        return registry[image_id]

    def output_dir(image_id):
        record = image_record(image_id)
        # Pipeline outputs use the source filename stem, which can differ from
        # the registry ID (the bundled example is registered as "example").
        output_id = record["path"].stem
        for folder in [runs / output_id, project / "artifacts/prototype_p0" / output_id]:
            if all(
                (folder / name).exists()
                for name in [
                    "seed_instances.csv",
                    "dishes.json",
                    "calibration.json",
                    "instances.json",
                ]
            ):
                if json.loads((folder / "instances.json").read_text())["backend"] != selected_backend:
                    continue
                if spatial_required:
                    report = folder / "spatial_preview.json"
                    if not report.exists() or not json.loads(report.read_text()).get("applied_to_measurements"):
                        continue
                status = folder / "run_status.json"
                if not status.exists() or json.loads(status.read_text())["pipeline_status"] == "ok":
                    return folder
        raise HTTPException(409, "This photo has not been analyzed yet. Start analysis first.")

    def result(image_id):
        from .web_review import load_result

        record = image_record(image_id)
        with lock:
            return load_result(
                output_dir(image_id),
                image_id=image_id,
                filename=record["filename"],
                review_path=reviews / f"{image_id}.json",
                image_url=f"/api/images/{image_id}/original",
                corrected_url=f"/api/images/{image_id}/asset/corrected",
                original_path=record["path"],
            )

    def queue(image_id, force=False):
        record = image_record(image_id)
        with lock:
            active = next(
                (
                    j
                    for j in jobs.values()
                    if j["image_id"] == image_id and j["status"] in {"queued", "running"}
                ),
                None,
            )
            if active:
                return {"job_id": active["job_id"], "image_id": image_id}
            job_id = uuid4().hex
            job = {
                "job_id": job_id,
                "image_id": image_id,
                "status": "queued",
                "message": "Queued for analysis",
                "events": [], "revision": 0,
            }
            jobs[job_id] = job

        def execute():
            with lock:
                job.update(status="running", message="Calibrating scale, locating dishes, and measuring seed candidates")
            try:
                def on_progress(event):
                    with lock:
                        revision = job['revision'] + 1
                        job['events'].append(dict(event, revision=revision, timestamp=time()))
                        job.update(revision=revision, stage=event['stage'],
                                   message=event.get('message') or f"{event['stage']}: {event['status']}")
                try:
                    if force:
                        raise HTTPException(409, 'Reanalysis requested')
                    output_dir(image_id)
                except HTTPException as exc:
                    if exc.status_code != 409:
                        raise
                    run_pipeline(record["path"], config, output_root=runs,
                                 progress_callback=on_progress)
                if current_settings()['integrations']['paper_labels']['enabled']:
                    try:
                        queue_labels(image_id)
                    except Exception:
                        with lock:
                            label_jobs[image_id] = dict(status='failed', message='Labels unavailable; enter identifiers manually.')
                with lock:
                    job.update(
                        status="done", message="Analysis complete. Ready for manual review.", result_image_id=image_id
                    )
            except Exception as exc:  # noqa: BLE001 — job boundary must report any analysis failure.
                message = (
                    exc.summary.get("error_message", str(exc))
                    if isinstance(exc, PipelineError)
                    else str(exc)
                )
                with lock:
                    job.update(status="failed", message=message)

        executor.submit(execute)
        return {"job_id": job_id, "image_id": image_id}

    def queue_labels(image_id):
        folder = output_dir(image_id)
        record = image_record(image_id)
        with lock:
            previous = label_jobs.get(image_id)
            if previous and previous['status'] in {'queued', 'running'}:
                return dict(previous)
            job = dict(status='queued', image_id=image_id, message='Reading paper labels…')
            label_jobs[image_id] = job

        def execute_labels():
            from .paper_labels import recognize_labels
            from .settings import load_api_key
            with lock:
                job['status'] = 'running'
                job['labels'] = []
            def label_completed(reading):
                with lock:
                    job['labels'] = [*job['labels'], reading]
            try:
                prefs = current_settings()['integrations']['paper_labels']
                if not prefs['enabled']:
                    raise ValueError('Paper label recognition is disabled in Settings.')
                report = recognize_labels(
                    record['path'], json.loads((folder / 'dishes.json').read_text()),
                    folder, key=load_api_key(state), model=prefs['model'],
                    base_url=prefs['base_url'], provider=prefs['provider'],
                    on_progress=label_completed)
                errors = sum(r['status'] == 'api_error' for r in report['labels'])
                with lock:
                    job.update(status='done', message=f"{len(report['labels'])} labels read" if not errors else f'{errors} label requests failed; identifiers can be entered manually.')
            except Exception:
                with lock:
                    job.update(status='failed', message='Paper label recognition failed. Check Settings → Integrations or enter identifiers manually.')
        label_executor.submit(execute_labels)
        return dict(job)

    @app.post('/api/results/{image_id}/labels')
    def read_labels(image_id: str):
        return queue_labels(image_id)

    @app.get('/api/results/{image_id}/labels')
    def label_status(image_id: str):
        image_record(image_id)
        with lock:
            return dict(label_jobs.get(image_id, {'status': 'idle', 'message': ''}))

    @app.post('/api/results/{image_id}/sample-id')
    def edit_sample_id(image_id: str, request: SampleIDRequest):
        from .web_review import save_sample_id
        with lock:
            current = result(image_id)
            try:
                save_sample_id(reviews / f'{image_id}.json', current, request.dish_id, request.sample_id)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            return result(image_id)

    @app.get('/api/images/{image_id}/paper/{dish_id}')
    def paper_crop(image_id: str, dish_id: int):
        folder = output_dir(image_id)
        if dish_id < 1 or dish_id > 100:
            raise HTTPException(404, 'Paper crop not found')
        path = folder / f'paper_label_{dish_id:02d}.jpg'
        if not path.is_file():
            raise HTTPException(404, 'Paper crop not available')
        return FileResponse(path)

    @app.get('/api/settings')
    def get_settings():
        from .settings import public_settings
        return public_settings(state, workbench_config)

    @app.post('/api/settings')
    def update_settings(request: SettingsRequest):
        from .settings import public_settings, save_api_key, save_settings
        try:
            save_settings(state, request.settings, workbench_config)
            if request.clear_api_key:
                save_api_key(state, '')
            elif request.api_key is not None and request.api_key.strip():
                save_api_key(state, request.api_key)
            return public_settings(state, workbench_config)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/settings/test-integration')
    def test_integration(request: IntegrationTestRequest):
        from .paper_labels import test_provider_connection
        from .settings import load_api_key
        key = request.api_key.strip() if request.api_key and request.api_key.strip() else load_api_key(state)
        try:
            test_provider_connection(key=key, model=request.model.strip(), base_url=request.base_url.strip().rstrip('/'))
            return {'ok': True, 'message': 'Connection successful.'}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/health")
    def health():
        return {"status": "ok", "scale_source": "long_ruler"}

    @app.get("/api/images")
    def list_images():
        discover()
        images = []
        for image_id, record in sorted(registry.items(), key=lambda item: (not item[1].get("is_example", False), item[0])):
            entry = {"image_id": image_id, "filename": record["filename"], "processed": False, "is_example": bool(record.get("is_example", False))}
            try:
                folder = output_dir(image_id)
                status = json.loads((folder / "run_status.json").read_text())
                entry.update(
                    processed=True,
                    dish_count=status["dish_count"],
                    seed_count=status["seed_count_total"],
                )
            except HTTPException:
                pass
            images.append(entry)
        return {"images": images}

    @app.post("/api/process")
    def process(request: ProcessRequest):
        return queue(request.image_id, request.force)

    @app.post("/api/upload")
    async def upload(file: Annotated[UploadFile, File()]):
        data = await file.read(30 * 1024 * 1024 + 1)
        if len(data) > 30 * 1024 * 1024:
            raise HTTPException(413, "Photo size exceeds 30 MB")
        try:
            # Check dimensions before decoding large arrays; preserve exact bytes.
            import io

            from PIL import Image

            with Image.open(io.BytesIO(data)) as decoded:
                if decoded.width * decoded.height > 50_000_000:
                    raise HTTPException(413, "Photo exceeds 50 million pixels")
                decoded.verify()
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid image")
        except HTTPException:
            raise
        except (ValueError, OSError, cv2.error) as exc:
            raise HTTPException(400, "Could not read this photo. Use a JPEG, PNG, or TIFF image.") from exc
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            suffix = ".jpg"
        image_id = "upload_" + uuid4().hex
        uploads.mkdir(parents=True, exist_ok=True)
        destination = uploads / f"{image_id}{suffix}"
        destination.write_bytes(data)
        filename = Path((file.filename or "uploaded.jpg").replace("\\", "/")).name
        destination.with_suffix(".json").write_text(
            json.dumps({"filename": filename}, ensure_ascii=False), encoding="utf-8"
        )
        with lock:
            registry[image_id] = {"path": destination, "filename": filename}
        return {"image_id": image_id, "filename": filename, "processed": False}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, after_revision: int = 0):
        with lock:
            if job_id not in jobs:
                raise HTTPException(404, "Analysis job not found")
            job = jobs[job_id]
            return {**{k: v for k, v in job.items() if k != 'events'},
                    'events': [e for e in job['events'] if e['revision'] > after_revision]}

    @app.get('/api/jobs/{job_id}/asset/{filename}')
    def job_asset(job_id: str, filename: str):
        with lock:
            job = jobs.get(job_id)
            if not job or filename not in {'_progress_color.jpg', '_progress_spatial.jpg'}:
                raise HTTPException(404, 'Progress image not found')
            path = runs / image_record(job['image_id'])["path"].stem / filename
            if not path.is_file():
                raise HTTPException(404, 'Progress image not ready')
            return FileResponse(path, headers={'Cache-Control': 'no-store'})

    @app.get("/api/results/{image_id}")
    def get_result(image_id: str):
        return result(image_id)

    @app.post("/api/results/{image_id}/review")
    def review(image_id: str, request: ReviewRequest):
        from .web_review import save_review

        with lock:
            current = result(image_id)
            try:
                save_review(
                    reviews / f"{image_id}.json",
                    current,
                    request.dish_id,
                    request.seed_id,
                    request.state,
                )
            except (ValueError, KeyError) as exc:
                raise HTTPException(400, str(exc)) from exc
            return result(image_id)

    @app.get("/api/results/{image_id}/export")
    def export(image_id: str, kind: str = "seeds", mode: str = "confirmed"):
        from .web_review import export_csv

        try:
            content = export_csv(result(image_id), kind=kind, mode=mode, precision=current_settings()['export']['decimal_precision'])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        filename = f"{image_id}_{kind}_{mode}.csv"
        return Response(
            content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/images/{image_id}/original")
    def original(image_id: str, request: Request, preview: bool = False):
        from .web_preview import display_image
        return display_image(image_record(image_id)["path"], request, state / "display_cache", preview=preview)

    @app.get("/api/images/{image_id}/asset/{name}")
    def asset(image_id: str, name: str, request: Request, preview: bool = False):
        allowed = {
            "corrected": "corrected.jpg",
            "ruler": "ruler_overlay.jpg",
            "checker": "checker_overlay.jpg",
            "preview": "preview.jpg",
            "spatial": "spatial_preview.jpg",
            "spatial_color": "spatial_color_preview.jpg",
            "spatial_rims": "spatial_rim_overlay.jpg",
        }
        if name not in allowed:
            raise HTTPException(404, "Preview not found")
        folder = output_dir(image_id)
        path = folder / allowed[name]
        if not path.exists() and name in {"corrected", "spatial_color"}:
            from .color_calibration import apply_color_calibration
            from .contracts import ColorCalibration

            calibration = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
            color = calibration["color"]
            color_calibration = ColorCalibration(
                matrix=np.asarray(color["matrix"], dtype=float),
                bbox=tuple(color["bbox"]),
                mean_delta_e00_before=float(color["mean_delta_e00_before"]),
                mean_delta_e00_after=float(color["mean_delta_e00_after"]),
                metadata=color.get("metadata", {}),
            )
            source = cv2.imread(str(image_record(image_id)["path"]))
            if source is None:
                raise HTTPException(500, "Could not load source image")
            corrected = apply_color_calibration(source, color_calibration)
            corrected_path = folder / "corrected.jpg"
            if not cv2.imwrite(str(corrected_path), corrected):
                raise HTTPException(500, "Could not generate corrected preview")
            if name == "spatial_color":
                report_path = folder / "spatial_preview.json"
                if not report_path.exists():
                    raise HTTPException(404, "Spatial preview has not been generated yet")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                matrix = np.asarray(report["matrix"], dtype=float)
                size = tuple(report["output_size"])
                spatial_color = cv2.warpPerspective(corrected, matrix, size)
                if not cv2.imwrite(str(path), spatial_color):
                    raise HTTPException(500, "Could not generate corrected spatial preview")
        if not path.exists():
            raise HTTPException(404, "Preview has not been generated yet")
        from .web_preview import display_image
        return display_image(path, request, state / "display_cache", preview=preview,
                             versioned=bool(request.query_params.get("v")))

    @app.get("/api/images/{image_id}/thumbnail")
    def thumbnail(image_id: str):
        record = image_record(image_id)
        from PIL import Image, ImageOps
        import io
        with Image.open(record["path"]) as original:
            original.draft("RGB", (240, 180))
            image = ImageOps.exif_transpose(original)
            image.thumbnail((240, 180))
            image = image.convert("RGB")
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG", quality=80)
        return Response(encoded.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/images/{image_id}/dish/{dish_id}")
    def dish(image_id: str, dish_id: int, corrected: bool = False, thumbnail: bool = False):
        folder = output_dir(image_id)
        dishes = json.loads((folder / "dishes.json").read_text())
        selected = next((d for d in dishes if d["dish_id"] == dish_id), None)
        if selected is None:
            raise HTTPException(404, "Dish not found")
        image = cv2.imread(
            str(folder / "corrected.jpg" if corrected else image_record(image_id)["path"])
        )
        x, y, w, h = selected["bbox"]
        crop = image[y : y + h, x : x + w]
        if thumbnail:
            scale = min(1., 320 / max(crop.shape[:2]))
            if scale < 1:
                crop = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)),
                                  interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(
            ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80 if thumbnail else 92]
        )
        if not ok:
            raise HTTPException(500, "Could not generate dish preview")
        return Response(encoded.tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=3600" if thumbnail else "private, max-age=0, must-revalidate"})

    @app.get("/")
    def index():
        return FileResponse(project / "web/index.html")

    @app.get("/{filename}")
    def static_alias(filename: str):
        if filename not in {"app.js", "styles.css"}:
            raise HTTPException(404, "Page not found")
        return FileResponse(project / "web" / filename)

    if (project / "web").exists():
        app.mount("/static", StaticFiles(directory=project / "web"), name="static")
    return app


app = create_app()
