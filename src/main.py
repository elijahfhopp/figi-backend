import logging

from db.models import FacesModel, ImagesModel

from image.face_extractor import FaceExtractor
from image.tree_crawler import crawl_for_images

from index import ImageIndexer
from rich.logging import RichHandler

IMAGES_FOLDER = "test_data/images"

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%Y-%m-%dT%H:%M:%S%Z]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("figi.main")

print(f"Final product is {crawl_for_images(IMAGES_FOLDER)}")

# try:
#     ImagesModel.drop_table(cascade=True)
#     FacesModel.drop_table()
# except Exception as e:
#     log.debug(f"{e}")

ImagesModel.create_table(True)
FacesModel.create_table(True)


# ImagesModel.insert(
#     path="a/George_W_Bush_0001.jpg", filetype=".jpg", size=1000
# ).execute()

extractor = FaceExtractor(".")
indexer = ImageIndexer(extractor)
indexer.index_and_load_to_db("test_data/images")

# img = cv2.imread("JJJ.jpg")
# cv2.imshow("hi", img)
# cv2.waitKey()
# faces = gen.extract_faces_from_file("JJJ.jpg")
# print(faces)
