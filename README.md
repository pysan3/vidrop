# Vidrop

> Drop Video Frames Based On Image Files

## Install

```bash
poetry install
```

## Run

### Truncate Video From the Frame Matching Any Reference Image

```bash
poetry run vidrop xxx.mp4 aaa.png bbb.png ccc.png --truncate
```

### Help

```bash
poetry run vidrop -h
```

```txt
usage: Drop Video Frames Based On Image Files

positional arguments:
  video                 Path to video file
  images                Reference Images

options:
  -h, --help            show this help message and exit
  --frames FRAMES FRAMES FRAMES
                        --frames <start> <stop> <step>
  -t, --truncate        Truncates video till end from the first occurrence of image
  -d, --drop            Drop only frames containing images. Overwritten by --truncate
  -o OUTPUT, --output OUTPUT
                        Path to output files. (Default: add `_vidrop` at end of <video>)
  -l {debug,trace,info,warn,error,off}, --log {debug,trace,info,warn,error,off}
                        Set logging level
  -v, --verbose         Alias for --log debug
  -vv, --veryverbose    Alias for --log debug and more detailed outputs
```
