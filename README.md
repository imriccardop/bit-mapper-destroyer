# BitMapper Destroyer

Convert RGB images, GIFs, and videos to corrupted black & white versions with customizable glitch effects.

A tool for batch corruption, dithering, downscaling, pixel sorting, datamoshing, and artistic degradation of visual media. Perfect for pixel art, glitch aesthetic, data obfuscation, or experimental video processing.

---

## Installation

### macOS App (recommended)

Download the latest `.dmg` from releases, mount it, and drag `BitMapper Destroyer.app` to `/Applications`.

### From source

```bash
pip3 install Pillow opencv-python numpy PySide6
python3 bitmapper_destroyer_gui.py
```

---

## GUI Features

### Input
- Drag & drop or file browser
- Supports JPG, PNG, GIF, MP4, AVI, MOV, MKV, WebM, and more
- Live preview with native video playback

### Dither Modes
| Mode | Description |
|------|-------------|
| None | PIL fixed threshold at 127 |
| Floyd-Steinberg | Error diffusion dithering |
| Custom Threshold | Manual threshold (1–255) with optional stochastic noise |
| Percent Threshold | Adaptive threshold based on target white % (histogram-based) |

### Glitch Effects
| Effect | Parameters |
|--------|-----------|
| Bloom | Radius |
| Data Corruption | Probability, Target (All/White/Black) |
| Block Shift | Probability, Block Size, Amplitude |
| Pixel Sorting | Probability, Direction (Ascending/Descending/Random) |
| Line Shift | Probability, Amplitude |
| Pixel Scatter | Probability, Radius |
| Mirror Segments | Probability, Max Size, Direction (Horizontal/Vertical/Random) |
| BW Reverse | On/Off |

### Scaling
- Presets: 1, 1/2, 1/3, 1/4, 1/8, 1/16
- Configurable resampling filter (LANCZOS, BICUBIC, BILINEAR, NEAREST, BOX, HAMMING)
- Resize after downscale option

### Output
- Auto Save: saves alongside source file
- Manual Save: saves to temp, then export on demand

---

## CLI Usage

```bash
python3 bit_mapper_destroyer.py <source> [target] [options]
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `source` | PATH | required | Input file (image, GIF, or video) |
| `target` | PATH | auto | Output file path (optional) |
| `-t, --threshold` | INT (1–255) | 128 | Black and white threshold value |
| `-d, --dither` | NONE, FLOYDSTEINBERG | none | Dithering algorithm |
| `-ds, --downscale` | FLOAT | 1 | Scale factor |
| `-rf, --resampling_filter` | LANCZOS, BICUBIC, BILINEAR, NEAREST, BOX, HAMMING | NEAREST | Resampling method |
| `-r, --randomize` | BOOL | false | Add stochastic noise to threshold |
| `-bmp, --bitmap` | BOOL | false | Save as BMP alongside output |
| `-v, --verbose` | BOOL | false | Verbose output |

### Examples

```bash
python3 bit_mapper_destroyer.py image.jpg -t 100
python3 bit_mapper_destroyer.py input.gif output.gif -d FLOYDSTEINBERG
python3 bit_mapper_destroyer.py video.mp4 -ds -4 -rf NEAREST -r -v
```

---

A pre-built `.dmg` is available on the [releases page](https://github.com/user/bit-mapper-destroyer/releases). Download, mount, and drag to `/Applications`.

---

## License

MIT License
