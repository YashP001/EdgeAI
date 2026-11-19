"""Run the supplied benchmark inside Codespaces; preserve each raw measurement."""
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/team_04'


def run(command, folder):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'summary.txt').open('w') as log:
        process=subprocess.Popen(command,cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            print(line,end='',flush=True)
            log.write(line)
        if process.wait()!=0:
            raise RuntimeError(f'Failed command: {command}')


def main():
    url='http://localhost:8000'
    for label,extra in [('codespace_local_keepalive',[]),('codespace_local_no_keepalive',['--no-keepalive'])]:
        run([sys.executable,str(ROOT/'client/bench.py'),'--url',url,'--n','200',*extra],RESULTS/'benchmarks'/label)
    subprocess.run([sys.executable,str(ROOT/'client/verify.py'),'--url',url,'--team','4',
                    '--output',str(RESULTS/'codespace_endpoint_check.json')],check=True,cwd=ROOT)
    (RESULTS/'benchmarks/codespace_environment.json').write_text(json.dumps({
        'platform':platform.platform(),'python':platform.python_version(),
        'url':url,'benchmark_requests_per_mode':200,'complete':True},indent=2))
    print('CODESPACE_CHECKS_COMPLETE',flush=True)


if __name__=='__main__':
    main()
