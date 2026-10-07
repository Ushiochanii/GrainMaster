"""Run the read-only live training panel independently of the photo workbench."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    import uvicorn
    from grainmaster.training_monitor import create_monitor_app
    uvicorn.run(create_monitor_app(ROOT),host='127.0.0.1',port=args.port,log_level='warning')
