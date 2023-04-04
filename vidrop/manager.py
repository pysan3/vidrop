import argparse
import sys
from enum import IntEnum, auto
from logging import Logger, getLogger
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from vidrop.media import Image


class LOG_LEVELS(IntEnum):
    DEBUG = auto()
    TRACE = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()
    OFF = auto()


parser = argparse.ArgumentParser('vidrop', 'Drop Video Frames Based On Image Files')
parser.add_argument('video', help='Path to video file')
parser.add_argument('images', nargs='+', help='Reference Images')
parser.add_argument('--frames', nargs=3, help='--frames <start> <stop> <step>')
parser.add_argument('-t', '--truncate', action='store_true',
                    help='Truncates video till end from the first occurrence of image')
parser.add_argument('-d', '--drop', action='store_true',
                    help='Drop only frames containing images. Overwritten by --truncate')
parser.add_argument('-o', '--output', help='Path to output files. (Default: add `_vidrop` at end of <video>)')
parser.add_argument('-l', '--log', choices=[e.name.lower() for e in list(LOG_LEVELS)], default='info',
                    help='Set logging level')
parser.add_argument('-v', '--verbose', action='store_true', help='Alias for --log debug')
parser.add_argument('-vv', '--veryverbose', action='store_true', help='Alias for --log debug and more detailed outputs')
default_args = parser.parse_args(sys.argv[1:])


@dataclass
class Manager:
    video: Path
    images: list[Path]

    logger: Optional[Logger] = None
    log_level: LOG_LEVELS = LOG_LEVELS.INFO
    veryverbose: bool = False

    frames: tuple[int, int, int] = (0, -1, 1)
    truncate: bool = False
    drop: bool = False
    _use_rich: bool = True
    _output: Optional[Path] = None

    _images_np: list[NDArray[np.uint8]] = field(default_factory=list)
    _images_gray: list[NDArray[np.uint8]] = field(default_factory=list)

    @property
    def output(self):
        o = self._output if self._output is not None else self.video.with_stem(f'{self.video.stem}_vidrop')
        o.parent.mkdir(exist_ok=True, parents=True)
        return o

    @property
    def images_np(self):
        if self._images_np is None or len(self._images_np) != len(self.images):
            self._images_np = [Image.load(f) for f in self.images]
        return self._images_np

    @property
    def images_gray(self):
        if self._images_gray is None or len(self._images_gray) != len(self.images):
            self._images_gray = [Image.to_grayscale(im) for im in self.images_np]
        return self._images_gray

    @property
    def log(self):
        assert self.logger is not None, f'Call {type(self)}.setup_logger() to set the correct logger'
        return self.logger

    @classmethod
    def argparse(cls, _args: Optional[list[str]] = None):
        args = parser.parse_args(_args) if _args is not None else default_args
        self = cls(
            Path(args.video),
            [Path(f) for f in args.images],
            truncate=args.truncate is True,
            drop=args.drop is True,
        )
        if args.frames is not None:
            self.frames = tuple(int(a) if a.isnumeic() else b for a, b in zip(args.frames, self.frames))
        if args.output is not None:
            self._output = Path(args.output)
        if (args.log or '').upper() in dir(LOG_LEVELS):
            self.log_level = LOG_LEVELS[args.log.upper()]
        if args.verbose:
            self.log_level = LOG_LEVELS.DEBUG
        if args.veryverbose:
            self.veryverbose = True
            self.log_level = LOG_LEVELS.DEBUG
        self.logger = self.setup_logger(self.log_level, self._use_rich)
        return self

    @staticmethod
    def setup_logger(level: int | LOG_LEVELS, use_rich=True):
        logger = getLogger(__name__)
        logger.setLevel(level)
        logger.handlers.clear()
        if use_rich:
            from rich.logging import RichHandler
            ch = RichHandler()
            logger.addHandler(ch)
        else:
            from logging import StreamHandler, Formatter
            ch = StreamHandler()
            ch.setFormatter(Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(ch)
        return logger

    def print_config(self):
        self.log.info('\n'.join([f'{k}: {v}' for k, v in asdict(self).items() if not k.startswith('_')]))

    def debug(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'DEBUG'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.debug("Houston, we have a %s", "thorny problem", exc_info=1)
        """
        self.log.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        """
        self.log.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        self.log.warning(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.log.warn(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'ERROR'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.error("Houston, we have a %s", "major problem", exc_info=1)
        """
        self.log.error(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """
        Convenience method for logging an ERROR with exception information.
        """
        self.error(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'CRITICAL'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.critical("Houston, we have a %s", "major disaster", exc_info=1)
        """
        self.log.critical(msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        """
        Don't use this method, use critical() instead.
        """
        self.log.fatal(msg, *args, **kwargs)
