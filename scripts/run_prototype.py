"""One-command single-image or dataset P0 runner."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from grainmaster.pipeline import run_pipeline, PipelineError, write_csv, BATCH_COLUMNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Photograph or directory of photographs')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/prototype.yaml')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--backend', choices=['classical', 'yolo26'], default=None)
    args = parser.parse_args()
    root = args.output or ROOT / 'artifacts/prototype_p0'
    if args.input.is_dir():
        files = sorted(p for p in args.input.iterdir()
                       if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
        if not files:
            parser.error('No photographs found')
        rows = []
        for index, path in enumerate(files, 1):
            try:
                row = run_pipeline(path, args.config, root, args.backend)
            except PipelineError as exc:
                row = exc.summary
            rows.append(row)
            root.mkdir(parents=True, exist_ok=True)
            write_csv(root / 'batch_summary.csv', rows, BATCH_COLUMNS)
            print(f'[{index}/{len(files)}] {path.stem}: {row}', flush=True)
        return 0 if all(r['pipeline_status'] == 'ok' for r in rows) else 1
    try:
        row = run_pipeline(args.input, args.config, root, args.backend)
    except PipelineError as exc:
        print(exc.summary, file=sys.stderr)
        return 1
    print(row)
    print(f'Outputs: {(root / args.input.stem).resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
