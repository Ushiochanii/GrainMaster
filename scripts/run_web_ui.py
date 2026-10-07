"""Launch the local GrainMaster canvas and review workbench."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/workbench_yolo26.yaml")
    parser.add_argument("--state-root", type=Path, default=ROOT / "artifacts/web_ui_yolo26_v2")
    args = parser.parse_args()
    import uvicorn

    from grainmaster.web_server import create_app

    print(f"GrainMaster Mozume: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_app(ROOT, config_path=args.config, state_root=args.state_root), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
