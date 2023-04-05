from pathlib import Path
import ffmpeg
from rich import print
import cv2
import numpy as np
from numpy.typing import NDArray


class VideoIter:
    def __init__(self, video_path: Path, start: int = 0, stop: int = -1, step: int = 1) -> None:
        self.video = video_path
        assert self.check_video_isvalid(self.video), f'{self.video} not found'
        self.vidcap = cv2.VideoCapture(str(self.video))
        self.current_frame = -1
        self.frame_range = start, stop, step

    def check_valid_frame_range(self, frame: int):
        if frame < self.frame_range[0] or self.check_frame_end(frame):
            return False
        return (frame - self.frame_range[0]) % self.frame_range[2] == 0

    def check_frame_end(self, frame: int):
        return (0 < self.frame_range[1] <= frame)

    @staticmethod
    def check_video_isvalid(p: Path):
        return p.exists()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.check_frame_end(self.current_frame):
                raise StopIteration()
            res: tuple[bool, NDArray[np.uint8]] = self.vidcap.read()
            self.current_frame += 1
            success, image = res
            if not success:
                raise StopIteration()
            if self.check_valid_frame_range(self.current_frame):
                break
        return self.current_frame, image


class Image:
    @staticmethod
    def load(file_path: Path):
        img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        return np.array(img.tolist(), dtype=np.uint8)

    @staticmethod
    def is_rgb(img: NDArray[np.uint8]):
        return len(img.shape) == 3 and img.shape[-1] == 3 and img.dtype == np.uint8

    @staticmethod
    def is_gray(img: NDArray[np.uint8]):
        return len(img.shape) == 2 and img.dtype == np.uint8

    @staticmethod
    def to_grayscale(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        assert Image.is_rgb(img), f'{img.shape} is wrong. Last dim should be 3'
        img = (np.average(img, axis=-1).astype(np.uint8) // 10) * 10
        ave = np.average(img)
        return ((img > ave) * 255).astype(np.uint8)

    @staticmethod
    def compare_grays(a: NDArray[np.uint8], b: NDArray[np.uint8]):
        assert Image.is_gray(a), f'img a: {a.shape} is wrong. should be grayscale'
        assert Image.is_gray(b), f'img b: {b.shape} is wrong. should be grayscale'
        assert a.shape == b.shape, f'a: {a.shape}, b: {b.shape} is not equal. cannot compare'
        diff = np.abs(a.astype(np.int64) - b) > 5
        # cv2.imwrite('tmp/diff.pgm', (diff * 255).astype(np.uint8))
        return np.sum(diff, dtype=np.int64)

    @staticmethod
    def check_resizable(a: NDArray[np.uint8], b: NDArray[np.uint8]):
        a_h, a_w = a.shape[:2]
        b_h, b_w = b.shape[:2]
        return a_h == b_h, a_w == b_w

    @staticmethod
    def resize_to_align(img: NDArray[np.uint8], wh: tuple[int, int], offset: tuple[int, int] = (0, 0)):
        offx, offy = offset
        w, h = wh
        imgx, imgy = img.shape[:2]
        lenx, leny = imgx - offx, imgy - offy
        assert offx < imgx and offy < imgy, f'offx: {offx} < imgx: {imgx}, offy: {offy} < imgy: {imgy}'
        assert lenx <= imgx and leny <= imgy, f'lenx: {lenx} < imgx: {imgx}, leny: {leny} < imgy: {imgy}'
        return img[offx:offx + w, offy:offy + h]


if __name__ == "__main__":
    base_dir = Path('.')
    img1 = Image.load(base_dir / 'mahjong_end.pgm')
    img2 = Image.load(base_dir / 'tmp' / f'{990:05}_{1}.pgm')
    print(f'is_rgb={Image.is_rgb(img1)}')
    print(f'is_gray={Image.is_gray(img1)}')
    print(f'check to_grayscale={Image.is_gray(Image.to_grayscale(img1))}')
    print(f'compare_grays={Image.compare_grays(Image.to_grayscale(img1), Image.to_grayscale(img2))}')
