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

### Process Multiple Video Files at Once

```bash
poetry run multi folder/ aaa.png bbb.png ccc.png --truncate
```

- This processes all `folder/*.mp4` videos and truncate using `{aaa,bbb,ccc}.png`.

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
  --overwrite           Overwrite the input file <video>
  --norun               Only show commands and do not run ffmpeg
  -l {debug,info,warn,error,off}, --log {debug,info,warn,error,off}
                        Set logging level
  -v, --verbose         Alias for --log debug
  -vv, --veryverbose    Alias for --log debug and more detailed outputs
```
