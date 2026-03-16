import os
import subprocess
from typing import List

import numpy as np
from PIL import Image


def save_image_sequence(
    path, images: List[np.ndarray], prefix: str = "frame", ext: str = "png"
):
    os.makedirs(path, exist_ok=True)
    for i, img in enumerate(images):
        out_path = os.path.join(path, f"{prefix}_{i:06d}.{ext}")
        Image.fromarray(img).save(out_path)


def save_video(output_path, pattern, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True)
