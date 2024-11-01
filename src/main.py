import logging
import warnings

from db.models import FacesModel, ImagesModel

from image.face_extractor import FaceExtractor

from index import ImageIndexer
from rich.logging import RichHandler

from tqdm import TqdmExperimentalWarning

# IMAGES_FOLDER = "test_data/images"
IMAGES_FOLDER = "local_test_data"

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%Y-%m-%dT%H:%M:%S%Z]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("figi.main")


# try:
#     ImagesModel.drop_table(cascade=True)
#     FacesModel.drop_table()
# except Exception as e:
#     log.debug(f"{e}")

ImagesModel.create_table(True)
FacesModel.create_table(True)

extractor = FaceExtractor(".")
indexer = ImageIndexer(extractor)
indexer.index_and_load_to_db(IMAGES_FOLDER)
