from pathlib import Path
import cv2
import numpy as np
from numpy.typing import NDArray


class VideoIter:
    def __init__(self, video_path: Path, start: int = 0, stop: int = -1, step: int = 1) -> None:
        self.video = video_path
        assert self.check_video_isvalid(self.video), f'{self.video} not found'
        self.vidcap = cv2.VideoCapture(str(self.video))
        self.current_frame = start - 1
        self.check_frame = self.check_frame_func(start, stop, step)

    @staticmethod
    def check_frame_func(start: int, stop: int, step: int):
        def check_frame(frame: int):
            if frame < start or frame >= stop:
                return False
            return (frame - start) % step == 0
        return check_frame

    @staticmethod
    def check_video_isvalid(p: Path):
        return p.exists()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.current_frame += 1
            res: tuple[bool, NDArray[np.uint8]] = self.vidcap.read()
            success, image = res
            if not success:
                raise StopIteration()
            if self.check_frame(self.current_frame):
                return self.current_frame, image


class Image:
    @staticmethod
    def load(file_path: Path):
        img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        print(f'img={img.shape}')
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
        return np.average(img, axis=-1)

    @staticmethod
    def compare_grays(a: NDArray[np.uint8], b: NDArray[np.uint8], thres: int = 5):
        assert Image.is_gray(a), f'img a: {a.shape} is wrong. should be grayscale'
        assert Image.is_gray(b), f'img b: {b.shape} is wrong. should be grayscale'
        assert a.shape == b.shape, f'a: {a.shape}, b: {b.shape} is not equal. cannot compare'
        return np.sum(np.abs(a - b)) < thres

    @staticmethod
    def resize_to_align(img: NDArray[np.uint8], wh: tuple[int, int], offset: tuple[int, int] = (0, 0)):
        offx, offy = offset
        w, h = wh
        lenx, leny = w - offx, h - offy
        imgx, imgy = img.shape[:2]
        assert offx < w and offy < h, f'offx: {offx} < w: {w}, offy: {offy} < h: {h}'
        assert lenx <= imgx and leny <= imgy, f'lenx: {lenx} < imgx: {imgx}, leny: {leny} < imgy: {imgy}'
        return img[offx:w, offy:h]
