import os
from datetime import datetime, timezone
from torchvision.transforms import PILToTensor
from typing import *


dt_str = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%Z")
pil_to_tensor = PILToTensor()
tensor_to_numpy = lambda x: x.squeeze().permute(1, 2, 0).numpy()

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]
VID_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".3gp", ".m4v"]
IMG_EXTENSIONS += list(map(str.upper, IMG_EXTENSIONS))
VID_EXTENSIONS += list(map(str.upper, VID_EXTENSIONS))
IMG_EXTENSIONS = tuple(IMG_EXTENSIONS)
VID_EXTENSIONS = tuple(VID_EXTENSIONS)


def get_all_files(
    path,
    prefix: Union[str, Tuple] = "",
    suffix: Union[str, Tuple] = "",
    contains="",
):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory.")
    files = []
    for pre, dirs, basenames in os.walk(path):
        for name in basenames:
            if name.startswith(prefix) and name.endswith(suffix) and contains in name:
                files.append(os.path.join(pre, name))
    return files


def list_dir(
    path,
    prefix: Union[str, Tuple] = "",
    suffix: Union[str, Tuple] = "",
    contains="",
):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory.")
    return [
        f"{path}/{name}"
        for name in os.listdir(path)
        if name.startswith(prefix) and name.endswith(suffix) and contains in name
    ]


get_filename = lambda path: os.path.splitext(os.path.basename(path))[0]
