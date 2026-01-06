# BitMapper Destroyer 

Convert RGB images, GIFs, and videos to corrupted black and white versions with customizable glitch effects.

A CLI tool for batch corruption, dithering, downscaling, and artistic degradation of visual media. Perfect for pixel art, glitch aesthetic, data obfuscation, or experimental video processing.
All parameters are optional and combinable: pass multiple flags for more destructive corruption.
---

## Features

- Multi-format support: Static images (JPG, PNG), animated GIFs, videos (MP4, AVI, MOV, MKV, WebM, etc.)
- Dithering modes: Floyd-Steinberg or simple threshold
- Scaling effects: Downscale-then-upscale for intentional pixelation and glitch
- Resampling filters: LANCZOS, BICUBIC, BILINEAR, NEAREST, BOX, HAMMING
- Randomization: Stochastic threshold for corruption
- BMP export: Optional bitmap output for images
- Batch-friendly: Auto-generated output filenames, verbose logging
- Frame-by-frame processing: Full control over GIF and video frame pipeline
- Performance: NumPy-free, lightweight dependencies (Pillow, OpenCV)

---

## Installation

### Prerequisites
- python3 3.8+
- pip3 (python3 package manager)

### Setup

```bash
pip3 install Pillow opencv-python3
```

---

## Usage

### Basic Syntax
```bash
python3 bit_mapper_destroyer.py <source> [target] [options]
```

### Help
```bash
python3 bit_mapper_destroyer.py -h
```

Output:
```
usage: bit_mapper_destroyer.py [-h] [-t THRESHOLD] [-d [DITHER]] [-ds DOWNSCALE] [-rf RESAMPLING_FILTER] [-r] [-bmp] [-v] source_file_name [target_file_name]

BitMapper Convert images/GIFs/videos to corrupted B&W

positional arguments:
  source_file_name      Source file (image, GIF, or video)
  target_file_name      Output file (optional, auto-generated if omitted)

optional arguments:
  -h, --help            show this help message and exit
  -t THRESHOLD, --threshold THRESHOLD
                        B&W threshold value 0-255 (default: 128)
  -d [DITHER], --dither [DITHER]
                        Dithering: NONE or FLOYDSTEINBERG (optional)
  -ds DOWNSCALE, --downscale DOWNSCALE
                        Scale factor: <0 downscale+upscale, >1 upscale (default: 1)
  -rf RESAMPLING_FILTER, --resampling_filter RESAMPLING_FILTER
                        Resampling: LANCZOS, BICUBIC, BILINEAR, NEAREST, BOX, HAMMING (default: NEAREST)
  -r, --randomize       Add randomization to threshold
  -bmp, --bitmap        Also save as BMP (images only)
  -v, --verbose         Verbose output
```


### Images

Unprocessed images: </br>![sample_1](/samples/sample_1.gif) ![sample_2](/samples/sample_2.gif) 

Simple threshold conversion:
```bash
python3 bit_mapper_destroyer.py samples/sample_1.gif samples/sample_1_t_100.gif -t 100
python3 bit_mapper_destroyer.py samples/sample_2.gif samples/sample_2_t_100.gif -t 100  
```
Output: </br>

![sample_1_processed](/samples/sample_1_processed_t_100.gif)
![sample_2_processed](/samples/sample_2_processed_t_100.gif)

```bash
python3 bit_mapper_destroyer.py samples/sample_1.gif samples/sample_1_t_150.gif -t 150
python3 bit_mapper_destroyer.py samples/sample_2.gif samples/sample_2_t_150.gif -t 150 
```
Output: </br>

![sample_1_processed](/samples/sample_1_processed_t_150.gif)
![sample_2_processed](/samples/sample_2_processed_t_150.gif)

Floyd-Steinberg dithering:
```bash
python3 bit_mapper_destroyer.py samples/sample_1.gif samples/sample_1_d_FLOYDSTEINBERG.gif -d FLOYDSTEINBERG
python3 bit_mapper_destroyer.py samples/sample_2.gif samples/sample_2_d_FLOYDSTEINBERG.gif -d FLOYDSTEINBERG
```
Output: </br>

![sample_1_processed](/samples/sample_1_d_FLOYDSTEINBERG.gif)
![sample_2_processed](/samples/sample_2_d_FLOYDSTEINBERG.gif)

Pixelated effect (1/4 resolution downscale, upscaled with nearest neighbor):
```bash
python3 bit_mapper_destroyer.py samples/sample_1.gif samples/sample_1_ds_-4.gif -ds -4 -rf NEAREST -v
python3 bit_mapper_destroyer.py samples/sample_2.gif samples/sample_2_ds_-4.gif -ds -4 -rf NEAREST -v
```
Output: </br>

![sample_1_processed](/samples/sample_1_ds_-4.gif)
![sample_2_processed](/samples/sample_2_ds_-4.gif)

Randomized corruption with custom threshold:
```bash
python3 bit_mapper_destroyer.py samples/sample_1.gif samples/sample_1_r.gif -t 128 -r -v
python3 bit_mapper_destroyer.py samples/sample_2.gif samples/sample_2_r.gif -t 128 -r -v
```
Output: </br>

![sample_1_processed](/samples/sample_1_r.gif)
![sample_2_processed](/samples/sample_2_r.gif)

---

## Options Reference

| Flag | Type | Default | Description                                                         |
|------|------|---------|---------------------------------------------------------------------|
| `source` | PATH | required | Input file (image, GIF, or video)                                   |
| `target` | PATH | auto | Output file path (optional)                                         |
| `-t, --threshold` | INT (0-255) | 128 | Black and white threshold value                                     |
| `-d, --dither` | NONE, FLOYDSTEINBERG | none | Dithering algorithm (optional)                                      |
| `-ds, --downscale` | FLOAT | 1 | Scale factor (negative: downscale, positive: upscale, 1: no change) |
| `-rf, --resampling_filter` | LANCZOS, BICUBIC, BILINEAR, NEAREST, BOX, HAMMING | NEAREST | Resampling method for scaling                                       |
| `-r, --randomize` | BOOL | false | Add stochastic noise to threshold                                   |
| `-bmp, --bitmap` | BOOL | false | Save as BMP alongside output (images only)                          |
| `-v, --verbose` | BOOL | false | Verbose console logging                                             |

---

## License

MIT License
