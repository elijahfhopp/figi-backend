import os
from dataclasses import dataclass
from typing import List

from db.models import ImagesModel
from image.embeddings_generator import ExtractedFace, FaceExtractor
from image.tree_crawler import crawl_for_images
from peewee import chunked


@dataclass
class ImageIndexEntry:
    path: str
    filetype: str
    size: int
    embeddings: List[ExtractedFace]


@dataclass
class ImageIndexer:
    face_extractor: FaceExtractor

    def index_folder(self, path: str):
        image_paths = crawl_for_images(path)

        # Eliminate already indexed files in a highly inefficient manner (see NOTES.md)
        indexed_paths: List[ImagesModel] = []

        for path_batch in chunked(image_paths, 100):
            models = ImagesModel.select().where(ImagesModel.path.in_(path_batch))
            indexed_paths.extend([model.path for model in models])

        new_entries = []
        for image_path in image_paths:
            filetype = os.path.splitext(image_path)[1]
            full_path = os.path.join(path, image_path)
            stat = os.stat(full_path)
            size = stat.st_size
            faces = self.face_extractor.extract_faces(full_path)
            entry = ImageIndexEntry(image_path, filetype, size, faces)
            new_entries.append(entry)

        print(new_entries)
