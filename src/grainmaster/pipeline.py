"""P0 orchestration, CSV exports and review artifacts."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .ruler import calibrate_ruler, draw_ruler_overlay
from .color_calibration import calibrate_color, apply_color_calibration, draw_color_overlay
from .dishes import detect_dishes, crop_dish, draw_dish_overlay
from .segmentation import segment_seeds, segment_seed_batch
from .phenotyping import measure_seed, aggregate_dishes
from .card_scale import detect_card_scale, draw_card_scale_overlay
from .geometry import assess_planar_rectification, save_geometry_diagnostics
from .spatial_correction import build_spatial_correction, rectify_instance, rectify_instance_crop, shape_descriptors

SEED_COLUMNS = ['image_id', 'dish_id', 'seed_id', 'length_mm', 'width_mm', 'area_mm2',
                'L', 'a', 'b', 'seg_confidence', 'qc_flag']
DISH_COLUMNS = ['dish_id', 'seed_count', 'mean_length_mm', 'mean_width_mm', 'mean_area_mm2']
BATCH_COLUMNS = ['image_id', 'dish_count', 'seed_count_total', 'spatial_calibration_status',
                 'color_calibration_status', 'pipeline_status', 'error_message']


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False,
                                   allow_nan=False), encoding='utf-8')


def write_csv(path, rows, columns):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write image: {path}')


class PipelineError(RuntimeError):
    def __init__(self, summary):
        super().__init__(summary['error_message'])
        self.summary = summary


def run_pipeline(image_path, config_path='configs/prototype.yaml', output_root=None,
                 backend=None, progress_callback=None):
    """Run a single photograph. Failed runs retain stage status in run_status.json."""
    image_path = Path(image_path)
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    options = dict(config.get('segmentation', {}))
    if options.get('weights') and not Path(options['weights']).is_absolute():
        options['weights'] = str((Path(config_path).resolve().parent.parent / options['weights']).resolve())
    selected_backend = backend or options.get('backend', 'classical')
    out = Path(output_root or config.get('prototype', {}).get('output_dir',
                                                            'artifacts/prototype_p0'))
    out = out / image_path.stem
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(image_id=image_path.stem, dish_count=0, seed_count_total=0,
                   spatial_calibration_status='not_run', color_calibration_status='not_run',
                   pipeline_status='failed', error_message='')
    stage = 'load_image'
    def progress(step, status, **data):
        if progress_callback is not None:
            progress_callback(jsonable(dict(stage=step, status=status, **data)))

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Cannot read image: {image_path}')
        stage = 'spatial_calibration'
        progress('ruler', 'running')
        spatial = calibrate_ruler(image)
        if not np.isfinite(spatial.pixels_per_mm) or spatial.pixels_per_mm <= 0:
            raise ValueError('Invalid ruler scale')
        summary['spatial_calibration_status'] = 'ok'
        calibration = {'spatial': spatial, 'color': None}
        write_json(out / 'calibration.json', calibration)
        save_image(out / 'ruler_overlay.jpg', draw_ruler_overlay(image, spatial))
        progress('ruler', 'done', ruler=dict(bbox=spatial.bbox, polygon=spatial.metadata.get('polygon'), ticks=spatial.metadata.get('tick_points', []), pixels_per_mm=spatial.pixels_per_mm, residual=spatial.residual))
        stage = 'color_calibration'
        progress('color', 'running')
        color = calibrate_color(image, reference_name=config.get('calibration', {}).get(
            'colorchecker', {}).get('reference_dataset', 'ColorChecker 2005'),
            model=config.get('calibration', {}).get('colorchecker', {}).get('model', 'root_polynomial2'))
        progress('color', 'running', checker=dict(bbox=color.bbox, centers=color.metadata['patch_centers'], radius=color.metadata['sample_radius_px']), message='Applying global color correction')
        corrected = apply_color_calibration(image, color)
        save_image(out / 'corrected.jpg', corrected)
        summary['color_calibration_status'] = 'ok'
        calibration['color'] = color
        write_json(out / 'calibration.json', calibration)
        save_image(out / 'checker_overlay.jpg', draw_color_overlay(image, color))
        if progress_callback is not None:
            preview_scale = min(1., 1600. / max(image.shape[:2]))
            corrected_progress = cv2.resize(corrected, None, fx=preview_scale, fy=preview_scale)
            save_image(out / '_progress_color.jpg', corrected_progress)
        progress('color', 'done', image_asset='_progress_color.jpg', image_size=(image.shape[1], image.shape[0]), delta_e_before=color.mean_delta_e00_before, delta_e_after=color.mean_delta_e00_after)
        # Secondary reference is an audit. Neither averaging nor a rejected warp
        # changes physical measurements. All images retain original pixel coordinates.
        geometry_options = config.get('geometry_diagnostics', {})
        spatial_warning = False
        if geometry_options.get('enabled', True):
            try:
                card = detect_card_scale(image, color)
                difference_percent = 100 * (card['pixels_per_mm'] / spatial.pixels_per_mm - 1)
                threshold = float(geometry_options.get('scale_warning_percent', 3.0))
                spatial_warning = abs(difference_percent) > threshold
                comparison = dict(status='disagreement' if spatial_warning else 'within_qc_threshold',
                                  primary_pixels_per_mm=spatial.pixels_per_mm,
                                  secondary_pixels_per_mm=card['pixels_per_mm'],
                                  signed_difference_percent=difference_percent,
                                  warning_threshold_percent=threshold,
                                  threshold_note='Engineering QC threshold, not certified physical accuracy',
                                  measurement_scale_source='long_ruler_only',
                                  secondary_used_for_measurements=False)
                geometry = assess_planar_rectification(image, color, spatial, card)
                geometry['applied_to_measurements'] = False
                calibration.update(card_scale=card, scale_comparison=comparison,
                                   geometry_diagnostic=geometry)
                save_image(out / 'scale_crosscheck_overlay.jpg',
                           draw_card_scale_overlay(draw_ruler_overlay(image, spatial), card))
                if geometry_options.get('export_candidate_preview', True):
                    save_geometry_diagnostics(image, geometry, out / 'geometry', image_path.stem)
            except (ValueError, cv2.error) as exc:
                calibration.update(card_scale={'status': 'failed', 'error': str(exc)},
                                   geometry_diagnostic={'status': 'not_run',
                                                        'applied_to_measurements': False})
            write_json(out / 'calibration.json', calibration)
        stage = 'dish_detection'
        progress('dishes', 'running')
        dishes = detect_dishes(image)
        summary['dish_count'] = len(dishes)
        if not dishes:
            raise ValueError('No blue dishes detected')
        write_json(out / 'dishes.json', dishes)
        progress('dishes', 'done', dishes=dishes)
        correction_report = None
        measurement_spatial = spatial
        if config.get('spatial_correction', {}).get('enabled', False):
            stage = 'spatial_correction'
            progress('spatial', 'running')
            correction_report, measurement_spatial, rim_overlay = build_spatial_correction(
                image, color, spatial, dishes)
            matrix = np.asarray(correction_report['matrix'])
            size = tuple(correction_report['output_size'])
            rectified_color = cv2.warpPerspective(corrected, matrix, size)
            save_image(out / 'spatial_preview.jpg', cv2.warpPerspective(image, matrix, size))
            save_image(out / 'spatial_color_preview.jpg', rectified_color)
            save_image(out / 'spatial_rim_overlay.jpg', rim_overlay)
            write_json(out / 'spatial_preview.json', correction_report)
            calibration['spatial_correction'] = correction_report
            write_json(out / 'calibration.json', calibration)
            if progress_callback is not None:
                S = np.diag([preview_scale, preview_scale, 1.])
                preview_size = tuple(max(1, round(v * preview_scale)) for v in size)
                save_image(out / '_progress_spatial.jpg', cv2.warpPerspective(corrected_progress, S @ matrix @ np.linalg.inv(S), preview_size))
            progress('spatial', 'done', image_asset='_progress_spatial.jpg', image_size=size, matrix=matrix, pixels_per_mm=measurement_spatial.pixels_per_mm, qc_warning=not correction_report['accepted'])
        else:
            progress('spatial', 'skipped')
        preview = draw_dish_overlay(image, dishes)
        traits = []
        instance_records = []
        mask_dir = out / 'masks'
        mask_dir.mkdir(exist_ok=True)
        crop_preview_dir = out / 'dish_previews'
        crop_preview_dir.mkdir(exist_ok=True)
        stage = 'seed_segmentation_and_measurement'
        dish_crops = [crop_dish(image, dish) for dish in dishes]
        progress('seeds', 'running', active_dish_ids=[d.dish_id for d in dishes], message='Inferring seed masks')
        dish_instances = segment_seed_batch(
            dish_crops, backend=selected_backend, weights=options.get('weights'),
            imgsz=int(config['model']['imgsz']),
            confidence=float(config['model'].get('confidence', 0.25)),
            device=config['model'].get('device', 'cpu'))
        for dish, crop, instances in zip(dishes, dish_crops, dish_instances):
            progress('seeds', 'running', active_dish_ids=[dish.dish_id], active_dish_id=dish.dish_id, dish_index=dishes.index(dish)+1, dish_total=len(dishes), message='Measuring seed instances')
            corrected_crop = crop_dish(corrected, dish)
            x, y, w, h = dish.bbox
            labels = np.zeros((h, w), dtype=np.uint16)
            for instance in instances:
                if instance.mask.shape != (h, w):
                    raise ValueError('Segmentation mask shape differs from dish crop')
                measurement_mask, measurement_image = instance.mask, corrected_crop
                if correction_report is not None:
                    measurement_mask, measurement_image = rectify_instance(
                        instance.mask, (x, y), matrix, rectified_color)
                measured = measure_seed(measurement_mask, measurement_image, measurement_spatial,
                                        image_id=image_path.stem, dish_id=dish.dish_id,
                                        seed_id=instance.seed_id,
                                        seg_confidence=instance.confidence)
                if selected_backend == 'classical':
                    measured.qc_flag = ';'.join(filter(None, [measured.qc_flag,
                                                            'classical_unvalidated']))
                if spatial_warning:
                    measured.qc_flag += ';spatial_reference_disagreement'
                if correction_report is not None and not correction_report['accepted']:
                    measured.qc_flag += ';spatial_correction_qc_warning'
                traits.append(measured)
                save_image(mask_dir / f'dish_{dish.dish_id:02d}_seed_{instance.seed_id:03d}.png',
                           instance.mask.astype(np.uint8) * 255)
                labels[instance.mask.astype(bool)] = instance.seed_id
                contours, _ = cv2.findContours(instance.mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                shifted = [c + np.array([[[x, y]]], dtype=c.dtype) for c in contours]
                tint = (int(60 + (instance.seed_id * 71) % 195),
                        int(60 + (instance.seed_id * 43) % 195), 255)
                cv2.drawContours(preview, shifted, -1, tint, 3)
                bx, by, bw, bh = instance.bbox
                cv2.putText(preview, f'{dish.dish_id}.{instance.seed_id}',
                            (x + bx, y + by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, tint, 2)
                instance_records.append({'dish_id': dish.dish_id,
                                         'seed_id': instance.seed_id,
                                         'bbox_crop_xywh': instance.bbox,
                                         'confidence': instance.confidence,
                                         'shape_descriptors': shape_descriptors(
                                             measurement_mask, measured.length_mm, measured.width_mm),
                                         'metadata': instance.metadata})
            if progress_callback is not None:
                progress('seeds', 'running', completed_dish_id=dish.dish_id, crop_origin=(x, y), seed_count=len(instances), seeds=[dict(seed_id=i.seed_id, bbox=i.bbox, polygons=[c.reshape(-1, 2) for c in cv2.findContours(i.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]]) for i in instances])
            save_image(mask_dir / f'dish_{dish.dish_id:02d}.png', labels)
            save_image(crop_preview_dir / f'dish_{dish.dish_id:02d}.jpg',
                       preview[y:y+h, x:x+w])
        progress('seeds', 'done', seed_count_total=len(traits))
        progress('export', 'running')
        rows = [asdict(t) for t in traits]
        dish_rows = aggregate_dishes(traits, dish_ids=[d.dish_id for d in dishes])
        write_csv(out / 'seed_instances.csv', rows, SEED_COLUMNS)
        write_csv(out / 'dish_summary.csv', dish_rows, DISH_COLUMNS)
        write_json(out / 'instances.json', {'backend': selected_backend,
                                          'mask_encoding': 'uint16 PNG: 0 background, seed_id',
                                          'instances': instance_records})
        save_image(out / 'preview.jpg', preview)
        summary['seed_count_total'] = len(traits)
        summary['pipeline_status'] = 'ok'
        write_json(out / 'run_status.json', {**summary, 'backend': selected_backend})
        progress('export', 'done', seed_count_total=len(traits))
        return summary
    except Exception as exc:
        if stage == 'spatial_calibration':
            summary['spatial_calibration_status'] = 'failed'
        if stage == 'color_calibration':
            summary['color_calibration_status'] = 'failed'
        summary['error_message'] = f'{stage}: {type(exc).__name__}: {exc}'
        write_json(out / 'run_status.json', {**summary, 'backend': selected_backend})
        raise PipelineError(summary) from exc
