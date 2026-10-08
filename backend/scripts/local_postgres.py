"""Control only this project's pre-initialized external-volume PostgreSQL."""
import argparse
from pathlib import Path
import shutil
import subprocess

from servemind.config.settings import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('start','stop','status'))
    action = parser.parse_args().action
    data = PROJECT_ROOT/'backend/runtime/postgres'
    socket = PROJECT_ROOT/'backend/runtime/pg-socket'
    if not (data/'PG_VERSION').is_file() or not socket.is_dir():
        raise RuntimeError('project_postgres_not_initialized')
    if not data.resolve().is_relative_to(PROJECT_ROOT.resolve()):
        raise RuntimeError('postgres_path_outside_project')
    binary = shutil.which('pg_ctl') or '/opt/homebrew/opt/postgresql@16/bin/pg_ctl'
    if not Path(binary).is_file():
        raise RuntimeError('postgres_runtime_missing')
    status = subprocess.run([binary,'-D',str(data),'status'],capture_output=True,text=True)
    if action=='status':
        print(status.stdout or status.stderr,end='')
        return status.returncode
    if action=='start' and status.returncode==0:
        print('project_postgres_already_running')
        return 0
    if action=='stop' and status.returncode!=0:
        print('project_postgres_already_stopped')
        return 0
    command = [binary,'-D',str(data),'-t','30']
    if action=='start':
        command += ['-l',str(PROJECT_ROOT/'backend/runtime/postgres.log'),'-o',
                    f"-p 5433 -k {socket} -c listen_addresses=''",'start']
    else:
        command += ['-m','fast','stop']
    return subprocess.run(command,timeout=40).returncode


if __name__=='__main__':
    raise SystemExit(main())
