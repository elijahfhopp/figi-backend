import logging
import os
from typing import List

log = logging.getLogger("figi.tree_crawler")

SUPPORTED_IMAGE_EXTENSIONS = [
    ".bmp",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".pbm",
    ".pgm",
    ".ppm",
    ".tiff",
    ".tif",
    ".webp",
    ".avif",
]


def has_image_file_extension(path: str) -> bool:
    _, extension = os.path.splitext(path)
    return extension in SUPPORTED_IMAGE_EXTENSIONS


def crawl_for_images(path: str) -> List[str]:
    start_dir = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    image_paths = walk_tree_for_images("")
    os.chdir(start_dir)
    return image_paths


def walk_tree_for_images(path: str) -> List[str]:
    log.debug(f"Entering directory: {path}")
    image_paths: List[str] = []
    # to use cwd at start while avoiding adding leading dot to generated paths
    if not path:
        dir = os.listdir(".")
    else:
        dir = os.listdir(path)

    for listing in dir:
        listing = os.path.join(path, listing)
        if os.path.isdir(listing):
            image_paths.extend(walk_tree_for_images(listing))
        else:
            if has_image_file_extension(listing):
                image_paths.append(listing)
    return image_paths
