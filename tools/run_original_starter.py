"""Run the original instructor trainer in isolation without changing its source."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--team',type=int,default=4)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/'results'/f'team_{args.team:02d}'/'original_starter'
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        work=Path(folder)
        with zipfile.ZipFile(args.archive) as archive:
            for relative in ['train.py','app/model.py','app/__init__.py',f'data/team_{args.team:02d}_train.csv']:
                target=work/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.read('edge_ai_assignment1/'+relative))
        # Reproduce the original trainer; only constrain CPU thread count.
        environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        process=subprocess.Popen([sys.executable,'-u','train.py','--team',str(args.team)],
                                 cwd=work,env=environment,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True)
        with (output/'training.log').open('w') as log:
            for line in process.stdout:
                print(line,end='',flush=True)
                log.write(line);log.flush()
        if process.wait()!=0:
            raise RuntimeError('Original starter failed; see training.log')
        (output/'config.json').write_bytes((work/'model/config.json').read_bytes())
        (output/'train.py.reference').write_bytes((work/'train.py').read_bytes())
        print('Original starter completed; final submission model was not overwritten.')


if __name__=='__main__':
    main()
