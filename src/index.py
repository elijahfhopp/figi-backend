from dataclasses import dataclass

from image.embeddings_generator import EmbeddingsGenerator
from image.tree_crawler import crawl_for_images


class ImageInfo:
    pass


@dataclass
class ImageIndexer:
    embedding_gen: EmbeddingsGenerator

    def index_folder(self, path: str):
        image_paths = crawl_for_images(path)
        print(image_paths)
