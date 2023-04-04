import sys
from vidrop.media import Image, VideoIter
from .manager import Manager
from rich.progress import track
from rich import print
import ffmpeg


def get_fps(mng: Manager):
    probe = ffmpeg.probe(mng.video)
    if mng.veryverbose:
        print(probe)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps = int(video_info['r_frame_rate'].split('/')[0])
    return fps // 1000


def truncate(mng: Manager, at: int):
    # -ss: start time
    # -to: end time
    return False


def drop(mng: Manager, at: int):
    # https://stackoverflow.com/questions/72483584/how-to-remove-a-frame-with-ffmpeg-without-re-encoding
    return False


def get_function(mng: Manager):
    if mng.truncate:
        return truncate
    elif mng.drop:
        return drop
    raise NotImplementedError('No option set to modify the video file')


def main():
    mng = Manager.argparse()
    mng.print_config()
    fps = get_fps(mng)
    mng.info(f'{mng.video} fps: {fps}')
    for fr_id, fr_img in track(VideoIter(mng.video, mng.frames[0], mng.frames[1], fps),
                               description=f'Processing {mng.video}...'):
        mng.debug(f'Working on frame: {fr_id}')
        assert Image.is_rgb(fr_img)
        fr_gray = Image.to_grayscale(fr_img)
        assert Image.is_gray(fr_gray)
        for path, sample in zip(mng.images, mng.images_gray):
            fr_resize = Image.resize_to_align(fr_gray, sample.shape)
            if Image.compare_grays(fr_resize, sample):
                mng.info(f'{path} hit at {fr_id}')
                if get_function(mng)(mng, fr_id):
                    return
                break


if __name__ == "__main__":
    main()
