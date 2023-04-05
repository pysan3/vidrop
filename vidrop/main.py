from pathlib import Path
import sys
import shutil
import tempfile

import cv2
from vidrop.media import Image, VideoIter
from .manager import Manager
from rich.progress import track
from rich import print
import ffmpeg


class VideoControl:
    def __init__(self, mgr: Manager) -> None:
        self.mgr = mgr

    def get_fps(self, video: Path):
        video_info = self.get_probe(video)
        if video_info is None:
            return 30
        fps = int(video_info['r_frame_rate'].split('/')[0]) // 1000
        return fps

    def get_probe(self, video: Path):
        probe = ffmpeg.probe(video)
        video_infos = list(s for s in probe['streams'] if s['codec_type'] == 'video')
        if len(video_infos) < 1:
            self.mgr.log.error(f'Cannot find video information of {video}')
            return None
        return video_infos[0]

    def get_num_frames(self, video: Path):
        video_info = self.get_probe(video)
        if video_info is None:
            return None
        return int(video_info['nb_frames'])

    @staticmethod
    def force_stream_to_file(stream, outfile: Path):
        return stream.output(str(outfile), format='mp4').overwrite_output().run(quiet=True)

    def stream_to_output(self, stream, outfile: Path, overwrite=True):
        if outfile.exists():
            if overwrite or input(f'{outfile} exists. Overwrite? [Y/n]').lower() in 'yes':
                pass
            else:
                return False
        if stream is None:
            return False
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as tmp:
            tmp_path = Path(tmp.name)
            if self.mgr.veryverbose:
                print(f'Writing to {tmp_path}')
            _, err = self.force_stream_to_file(stream, tmp_path)
            if err is None:
                return False
            self.mgr.log.info(f'Writing stream to {tmp_path} success.')
            shutil.copy(tmp_path, outfile)
            self.mgr.log.info(f'{tmp_path} copied to {outfile} success.')
            return True

    def truncate(self, video: Path, out: Path, at: int):
        # -ss: start time
        # -to: end time
        fps = self.get_fps(video)
        to = at // fps
        if to <= 1:
            self.mgr.log.error(f'Video length is not enough at: {to}')
            return False
        stream = ffmpeg.input(str(video), ss=0, t=to)
        print(stream.video)
        print(stream.audio)
        print(stream)
        self.mgr.log.info(f'Loaded input from {video}. ss={0}, t={to}')
        return self.stream_to_output(stream, out, overwrite=True)

    def drop(self, video: Path, out: Path, at: int):
        # https://stackoverflow.com/questions/72483584/how-to-remove-a-frame-with-ffmpeg-without-re-encoding
        raise NotImplementedError('Not available now')

    def crop_video_at(self, video: Path, out: Path, at: int):
        if self.mgr.truncate:
            return self.truncate(video, out, at)
        elif self.mgr.drop:
            return self.drop(video, out, at)
        raise NotImplementedError('No option set to modify the video file')


TEST_DIR = Path('./tmp')
TEST_DIR.mkdir(exist_ok=True, parents=True)


def process_video(mgr: Manager):
    v = VideoControl(mgr)
    fps = v.get_fps(mgr.video)
    num_frames = v.get_num_frames(mgr.video)
    mgr.log.info(f'{mgr.video} fps: {fps}')
    for fr_id, fr_img in track(VideoIter(mgr.video, mgr.frames[0], mgr.frames[1], fps),
                               description=f'Processing {mgr.video}...'):
        assert Image.is_rgb(fr_img)
        fr_gray = Image.to_grayscale(fr_img)
        fr_h, fr_w = fr_gray.shape[:2]
        assert Image.is_gray(fr_gray)
        min_score = fr_h * fr_w
        for path, sample in zip(mgr.images, mgr.images_gray):
            if mgr.veryverbose:
                cv2.imwrite(str(TEST_DIR / path.name), sample)
            sample_h, sample_w = sample.shape[:2]
            h_aligned, w_aligned = Image.check_resizable(fr_gray, sample)
            if h_aligned and w_aligned:
                offsets = [(0, 0)]
            elif h_aligned:
                offsets = [(0, 0), (0, fr_w - sample_w)]
            elif w_aligned:
                offsets = [(0, 0), (fr_h - sample_h, 0)]
            else:
                raise RuntimeError(f'fr_gray: {fr_gray.shape}, sample ({path}): {sample.shape} do not align')
            for offset in offsets:
                if mgr.veryverbose:
                    print(f'Working on offset: {offset}')
                fr_resize = Image.resize_to_align(fr_gray, sample.shape, offset)
                if mgr.veryverbose:
                    cv2.imwrite(str(TEST_DIR / f'{fr_id:05}_{int(bool(sum(offset)))}.pgm'), fr_resize)
                diff = Image.compare_grays(fr_resize, sample)
                min_score = min(diff, min_score)
                if diff < (sample_h * sample_w * 1e-2):
                    mgr.log.warn(f'{path} hit at {fr_id} with offset {offset}')
                    if v.crop_video_at(mgr.video, mgr.output, fr_id):
                        mgr.log.info(f'Cropped at {fr_id} to {mgr.output}. Exitting with success.')
                        return
                    break
            else:
                continue
            break
        mgr.log.debug(f'Working on frame: {fr_id:>5} / {num_frames}'
                      + (f' ({fr_id / num_frames * 100:>5.2f}%)' if num_frames is not None else '')
                      + f' = {fr_id // fps // 60:>3}m {fr_id // fps % 60:02}s: {min_score} pts')


def main():
    mgr = Manager.argparse()
    mgr.print_config()
    process_video(mgr)


if __name__ == "__main__":
    main()
