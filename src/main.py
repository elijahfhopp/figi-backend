import logging

from image.tree_crawler import crawl_for_images
from rich.logging import RichHandler

IMAGES_FOLDER = "test_data/images"

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%Y-%m-%dT%H:%M:%S%Z]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
# gen = EmbeddingsGenerator(".")
print(f"Final product is {crawl_for_images(IMAGES_FOLDER)}")
# img = cv2.imread("JJJ.jpg")
# cv2.imshow("hi", img)
# cv2.waitKey()
# faces = gen.extract_faces_from_file("JJJ.jpg")
# print(faces)
