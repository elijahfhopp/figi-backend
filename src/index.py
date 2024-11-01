import os
from dataclasses import dataclass
from typing import List

from db.models import DB_CONNECTION, FacesModel, ImagesModel
from image.face_extractor import ExtractedFace, FaceExtractor
from image.tree_crawler import crawl_for_images
from peewee import chunked


@dataclass
class ImageIndexEntry:
    path: str
    filetype: str
    size: int
    faces: List[ExtractedFace]


@dataclass
class ImageIndexer:
    face_extractor: FaceExtractor

    def index_and_load_to_db(self, path: str):
        image_paths = crawl_for_images(path)

        # Eliminate already indexed files in a highly inefficient manner (see NOTES.md)
        indexed_paths: List[ImagesModel] = []

        for path_batch in chunked(image_paths, 100):
            models = ImagesModel.select().where(ImagesModel.path.in_(path_batch))
            indexed_paths.extend([model.path for model in models])

        image_paths = [x for x in image_paths if x not in indexed_paths]

        new_entries: List[ImageIndexEntry] = []
        for image_path in image_paths:
            filetype = os.path.splitext(image_path)[1]
            full_path = os.path.join(path, image_path)
            stat = os.stat(full_path)
            size = stat.st_size
            faces = self.face_extractor.extract_faces(full_path)
            entry = ImageIndexEntry(image_path, filetype, size, faces)
            new_entries.append(entry)

        # Could be optimized to add images and faces separately.
        # You'd have to keep track of primary key of each or switch the foreign
        # key to the path, or something like that.
        # Additionally, this is unfortunate because the data
        with DB_CONNECTION.atomic():
            for entry in new_entries:
                image = ImagesModel.create(
                    path=entry.path, filetype=entry.filetype, size=entry.size
                )

                new_faces: List[tuple] = []
                for e in entry.faces:
                    f = e.
                    new_faces.append(
                        (image.id, f.score, f.x, f.y, f.top, f.left, e.embedding)
                    )
                # NOTE: Order-dependant.
                FacesModel.insert_many(new_faces).execute()
