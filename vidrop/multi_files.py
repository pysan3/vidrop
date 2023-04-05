from pathlib import Path
from multiprocessing import Pool, cpu_count
import sys
from rich import print
from .main import process_video
from .manager import Manager


def test_multi_process(mgr: Manager):
    mgr.print_config()
    import os
    mgr.log.info(f'Inside subprocess: {os.getpid()}')
    return os.getpid()


def main():
    args = sys.argv
    video_dir = Path(args[1])
    if not video_dir.exists() or not video_dir.is_dir():
        raise RuntimeError(f'{video_dir} is not a directory')
    video_mgrs = []
    for video_path in video_dir.iterdir():
        if not video_path.is_file() or video_path.suffix != '.mp4':
            continue
        args[1] = str(video_path)
        mgr = Manager.argparse(args[1:])
        mgr.parallel = True
        mgr.print_config()
        video_mgrs.append(mgr)
    print(f'Using {cpu_count() - 1} cpus.')
    with Pool(processes=cpu_count() - 1) as pool:
        results = pool.map(process_video, video_mgrs)
    for mgr, fr_id in zip(video_mgrs, results):
        print(f'{mgr.video}: {fr_id} -> {mgr.output}')
