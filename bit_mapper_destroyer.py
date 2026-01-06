#!/usr/bin/env python3
"""
BitMapper Destroyer
Convert RGB images/GIFs/videos to corrupted black & white versions.
"""

import argparse
import glob
import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
from PIL import Image

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png"}
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".3gp", ".3g2"}
GIF_EXTENSION = ".gif"

RESAMPLING_FILTERS = {
    "LANCZOS": Image.Resampling.LANCZOS,
    "BICUBIC": Image.Resampling.BICUBIC,
    "BILINEAR": Image.Resampling.BILINEAR,
    "NEAREST": Image.Resampling.NEAREST,
    "BOX": Image.Resampling.BOX,
    "HAMMING": Image.Resampling.HAMMING,
}

DEFAULTS = {
    "threshold": 128,
    "downscale": 1,
    "resampling_filter": "NEAREST",
    "postfix": "_corrupted",
    "frame_prefix_image": "image_frame",
    "frame_prefix_video": "video_frame",
}

# ANSI Color Codes
ANSI = {
    "red": "\033[91m",
    "bold": "\033[1m",
    "underline": "\033[4m",
    "inverse": "\033[1;7m",
    "reset": "\033[0m",
}

GREY_LEVELS_ARRAY = [0 if random.randint(0, 255) < i else 255 for i in range(256)]

# ============================================================================
# LOGGING & CONSOLE FUNCTIONS
# ============================================================================

def setup_logger(verbose: bool) -> logging.Logger:
    """Configure logging with timestamp."""
    logger = logging.getLogger(__name__)
    if verbose:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"{ANSI['red']}{ANSI['bold']}[%(asctime)s]{ANSI['reset']} %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.CRITICAL)
    return logger


def log_barrier(logger: logging.Logger) -> None:
    """Log a visual barrier."""
    logger.info("=" * 70)


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def str_to_bool(value: str) -> bool:
    """Convert string to boolean."""
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def str_to_dither(value: str) -> Optional[int]:
    """Convert string to PIL dither method."""
    value_upper = value.upper()
    if value_upper == "NONE":
        return Image.NONE
    elif value_upper == "FLOYDSTEINBERG":
        return Image.FLOYDSTEINBERG
    raise argparse.ArgumentTypeError(
        f"Invalid dither value '{value}'. Use 'NONE' or 'FLOYDSTEINBERG'."
    )


def str_to_resampling_filter(value: str) -> int:
    """Convert string to PIL resampling filter."""
    try:
        return RESAMPLING_FILTERS[value.upper()]
    except KeyError:
        available = ", ".join(RESAMPLING_FILTERS.keys())
        raise argparse.ArgumentTypeError(
            f"Invalid resampling filter '{value}'. Available: {available}"
        )


def validate_source_file(filepath: str) -> Path:
    """Validate source file exists and has supported format."""
    path = Path(filepath)
    
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File not found: {filepath}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Not a file: {filepath}")
    
    ext = path.suffix.lower()
    all_supported = SUPPORTED_IMAGE_FORMATS | SUPPORTED_VIDEO_FORMATS | {GIF_EXTENSION}
    
    if ext not in all_supported:
        supported_str = ", ".join(sorted(all_supported))
        raise argparse.ArgumentTypeError(
            f"Unsupported format '{ext}'. Supported: {supported_str}"
        )
    
    return path.resolve()


def validate_target_file(filepath: str) -> Path:
    """Validate target file format."""
    path = Path(filepath)
    ext = path.suffix.lower()
    
    supported = SUPPORTED_IMAGE_FORMATS | {GIF_EXTENSION}
    if ext not in supported:
        supported_str = ", ".join(sorted(supported))
        raise argparse.ArgumentTypeError(
            f"Unsupported target format '{ext}'. Supported: {supported_str}"
        )
    
    return path


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed arguments."""
    if not (0 < args.threshold <= 255):
        raise ValueError("Threshold must be between 1 and 255")


# ============================================================================
# CORE IMAGE PROCESSING
# ============================================================================

def apply_threshold_or_dither(
    image: Image.Image,
    threshold: int,
    dither: Optional[int],
    randomize: bool,
) -> Image.Image:
    """
    Apply threshold or dithering to convert image to 1-bit black & white.
    
    Args:
        image: Grayscale PIL Image
        threshold: Threshold value (0-255)
        dither: PIL dither method (Image.NONE, Image.FLOYDSTEINBERG, or None)
        randomize: If True, add randomization to threshold
    
    Returns:
        1-bit Image
    """
    if dither is not None:
        return image.convert("1", dither=dither)
    
    # Custom threshold with optional randomization
    def threshold_fn(pixel: int) -> int:
        if randomize:
            return 0 if pixel < threshold or GREY_LEVELS_ARRAY[pixel] > pixel else 255
        return 0 if pixel < threshold else 255
    
    return image.point(threshold_fn, "1")


def apply_scaling(
    image: Image.Image,
    downscale: float,
    resampling_filter: int,
) -> Image.Image:
    """
    Apply downscaling/upscaling with optional reprocessing.
    
    Args:
        image: 1-bit PIL Image
        downscale: Scale factor (<1: downscale, >1: upscale, 1: no change)
        resampling_filter: PIL resampling method
    
    Returns:
        Scaled Image
    """
    if downscale == DEFAULTS["downscale"]:
        return image
    
    original_size = image.size
    
    if downscale < 0:
        scale_factor = 1 / abs(downscale)
    else:
        scale_factor = downscale
    
    new_width = round(original_size[0] * scale_factor)
    new_height = round(original_size[1] * scale_factor)
    
    # Downscale then upscale to original (glitch effect)
    if downscale < 0:
        image = image.resize((new_width, new_height), resampling_filter)
        return image.resize(original_size, resampling_filter)
    
    # Pure upscale
    return image.resize((new_width, new_height), resampling_filter)


def process_image_to_bitmap(
    image: Image.Image,
    threshold: int,
    dither: Optional[int],
    downscale: float,
    resampling_filter: int,
    randomize: bool,
) -> Image.Image:
    """
    Full pipeline: threshold/dither → scaling → reprocess.
    
    Returns:
        Final 1-bit Image
    """
    # Step 1: Convert to B&W
    bitmap = apply_threshold_or_dither(image, threshold, dither, randomize)
    
    # Step 2: Scale
    bitmap = apply_scaling(bitmap, downscale, resampling_filter)
    
    # Step 3: Reapply threshold/dither after scaling
    if downscale != DEFAULTS["downscale"]:
        bitmap = apply_threshold_or_dither(bitmap, threshold, dither, randomize)
    
    return bitmap


# ============================================================================
# FILE I/O
# ============================================================================

def generate_output_filename(
    source_path: Path,
    target_path: Optional[Path] = None,
) -> Path:
    """Generate output filename from source or use provided target."""
    if target_path:
        return target_path
    
    return source_path.with_stem(source_path.stem + DEFAULTS["postfix"])


def save_image(
    image: Image.Image,
    output_path: Path,
    logger: logging.Logger,
    also_save_bmp: bool = False,
) -> None:
    """Save image and optionally as BMP."""
    image.save(str(output_path))
    logger.info(f"✓ Created: {output_path}")
    
    if also_save_bmp:
        bmp_path = output_path.with_suffix(".bmp")
        image.save(str(bmp_path))
        logger.info(f"✓ Created: {bmp_path}")


# ============================================================================
# CONVERTERS
# ============================================================================

def convert_image(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Convert single image to corrupted B&W version."""
    start = time.perf_counter()
    logger.info("━ IMAGE CORRUPTION START")
    
    source = Path(args.source_file_name)
    target = Path(args.target_file_name) if args.target_file_name else None
    output = generate_output_filename(source, target)
    
    try:
        image = Image.open(source).convert("L")
        bitmap = process_image_to_bitmap(
            image,
            threshold=args.threshold,
            dither=args.dither,
            downscale=args.downscale,
            resampling_filter=args.resampling_filter,
            randomize=args.randomize,
        )
        save_image(bitmap, output, logger, also_save_bmp=args.bitmap)
        image.close()
        
        elapsed = time.perf_counter() - start
        logger.info(f"✓ IMAGE CORRUPTION ENDED ({elapsed:.3f}s)")
    except Exception as e:
        logger.error(f"✗ Image processing failed: {e}")
        raise


def convert_gif(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Convert animated GIF to corrupted B&W version."""
    start = time.perf_counter()
    logger.info("━ GIF CORRUPTION START")
    
    source = Path(args.source_file_name)
    target = Path(args.target_file_name) if args.target_file_name else None
    output = generate_output_filename(source, target)
    
    try:
        gif = Image.open(source)
        frame_count = gif.n_frames
        
        if frame_count <= 0:
            raise ValueError("GIF has 0 frames")
        
        logger.info(f"Processing {frame_count} frames...")
        
        # Determine padding for frame numbering
        num_digits = len(str(frame_count))
        
        # Extract frames to temp folder
        with tempfile.TemporaryDirectory(
            prefix=f"{DEFAULTS['frame_prefix_image']}_"
        ) as temp_dir:
            frames = []
            durations = []
            
            for i in range(frame_count):
                gif.seek(i)
                
                # Get frame duration
                duration = gif.info.get("duration", 100)
                durations.append(duration)
                
                # Process frame
                frame_image = gif.copy().convert("L")
                bitmap = process_image_to_bitmap(
                    frame_image,
                    threshold=args.threshold,
                    dither=args.dither,
                    downscale=args.downscale,
                    resampling_filter=args.resampling_filter,
                    randomize=args.randomize,
                )
                frames.append(bitmap)
                
                logger.info(f"  Frame {i+1}/{frame_count} ✓")
            
            # Save as GIF
            frames[0].save(
                str(output),
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                optimize=False,
                disposal=2,
            )
            logger.info(f"✓ Created: {output}")
            
            gif.close()
        
        elapsed = time.perf_counter() - start
        logger.info(f"✓ GIF CORRUPTION ENDED ({elapsed:.3f}s)")
    except Exception as e:
        logger.error(f"✗ GIF processing failed: {e}")
        raise


def convert_video(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Convert video to corrupted B&W version frame-by-frame."""
    start = time.perf_counter()
    logger.info("━ VIDEO CORRUPTION START")
    
    source = Path(args.source_file_name)
    target = Path(args.target_file_name) if args.target_file_name else None
    output = generate_output_filename(source, target)
    
    try:
        # Open video
        video = cv2.VideoCapture(str(source))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        if frame_count <= 0:
            raise ValueError("Video has 0 frames")
        
        logger.info(f"Video: {frame_count} frames @ {fps:.1f} fps")
        
        num_digits = len(str(frame_count))
        
        # Extract and process frames in temp directory
        with tempfile.TemporaryDirectory(
            prefix=f"{DEFAULTS['frame_prefix_video']}_"
        ) as temp_dir:
            logger.info("Extracting frames...")
            
            # Extract all frames
            frame_idx = 0
            while True:
                success, frame_bgr = video.read()
                if not success:
                    break
                
                frame_idx += 1
                frame_path = Path(temp_dir) / f"frame_{frame_idx:0{num_digits}d}.png"
                cv2.imwrite(str(frame_path), frame_bgr)
            
            video.release()
            logger.info(f"✓ Extracted {frame_idx} frames")
            
            # Process each frame
            logger.info("Processing frames...")
            frame_files = sorted(glob.glob(str(Path(temp_dir) / "*.png")))
            
            for idx, frame_path in enumerate(frame_files, 1):
                frame_image = Image.open(frame_path).convert("L")
                bitmap = process_image_to_bitmap(
                    frame_image,
                    threshold=args.threshold,
                    dither=args.dither,
                    downscale=args.downscale,
                    resampling_filter=args.resampling_filter,
                    randomize=args.randomize,
                )
                bitmap.save(frame_path)
                
                if idx % max(1, frame_idx // 10) == 0:
                    logger.info(f"  {idx}/{frame_idx} ✓")
            
            logger.info("✓ Frames processed")
            
            # Reconstruct video
            logger.info("Reconstructing video...")
            
            # Get dimensions from first frame
            first_frame = cv2.imread(frame_files[0])
            height, width = first_frame.shape[:2]
            
            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
            
            for frame_path in frame_files:
                frame = cv2.imread(frame_path)
                writer.write(frame)
            
            writer.release()
            logger.info(f"✓ Created: {output}")
        
        elapsed = time.perf_counter() - start
        logger.info(f"✓ VIDEO CORRUPTION ENDED ({elapsed:.3f}s)")
    except Exception as e:
        logger.error(f"✗ Video processing failed: {e}")
        raise


# ============================================================================
# CLI & MAIN
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="BitMapper Convert images/GIFs/videos to corrupted B&W"
    )
    
    parser.add_argument(
        "source_file_name",
        type=validate_source_file,
        help="Source file (image, GIF, or video)",
    )
    
    parser.add_argument(
        "target_file_name",
        nargs="?",
        default=None,
        type=validate_target_file,
        help="Output file (optional, auto-generated if omitted)",
    )
    
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=DEFAULTS["threshold"],
        help=f"B&W threshold value 0-255 (default: {DEFAULTS['threshold']})",
    )
    
    parser.add_argument(
        "-d",
        "--dither",
        type=str_to_dither,
        default=None,
        nargs="?",
        help="Dithering: NONE or FLOYDSTEINBERG (optional)",
    )
    
    parser.add_argument(
        "-ds",
        "--downscale",
        type=float,
        default=DEFAULTS["downscale"],
        help=(
            "Scale factor: <0 downscale+upscale, >1 upscale (default: "
            f"{DEFAULTS['downscale']})"
        ),
    )
    
    parser.add_argument(
        "-rf",
        "--resampling_filter",
        type=str_to_resampling_filter,
        default=RESAMPLING_FILTERS[DEFAULTS["resampling_filter"]],
        help=f"Resampling: {', '.join(RESAMPLING_FILTERS.keys())} (default: {DEFAULTS['resampling_filter']})",
    )
    
    parser.add_argument(
        "-r",
        "--randomize",
        action="store_true",
        help="Add randomization to threshold",
    )
    
    parser.add_argument(
        "-bmp",
        "--bitmap",
        action="store_true",
        help="Also save as BMP (images only)",
    )
    
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    return parser


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(args.verbose)
    
    # Validate arguments
    validate_args(args)
    logger.info(f"Args: {vars(args)}")
    
    log_barrier(logger)
    logger.info("FILE CORRUPTION STARTING...")
    log_barrier(logger)
    
    try:
        file_ext = args.source_file_name.suffix.lower()
        
        if file_ext in SUPPORTED_IMAGE_FORMATS:
            convert_image(args, logger)
        elif file_ext == GIF_EXTENSION:
            convert_gif(args, logger)
        elif file_ext in SUPPORTED_VIDEO_FORMATS:
            convert_video(args, logger)
        else:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        log_barrier(logger)
        logger.info("✓✓✓ FILE CORRUPTED SUCCESSFULLY ✓✓✓")
        log_barrier(logger)
    except Exception as e:
        logger.error(f"\n✗✗✗ CORRUPTION FAILED ✗✗✗\n{e}")
        raise


main()