import logging
import warnings

from figi.config import CONFIG
IMAGES_FOLDER = CONFIG["FIGI_IMAGES_PATH"]
print(IMAGES_FOLDER)
exit()
import fastapi

import figi.routers as routers

from figi.db.models import FacesModel, ImagesModel

from figi.image.face_extractor import FaceExtractor

from figi.index.index import ImageIndexer
from rich.logging import RichHandler

from tqdm import TqdmExperimentalWarning

# IMAGES_FOLDER = "test_data/images"
IMAGES_FOLDER = CONFIG[ConfigKeys.IMAGES_PATH]
print(IMAGES_FOLDER)
exit()

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

UPDATE_INDEX = False
if UPDATE_INDEX:
    extractor = FaceExtractor(".")
    indexer = ImageIndexer(extractor)
    indexer.index_and_load_to_db(IMAGES_FOLDER)


app = fastapi.FastAPI()


app.include_router(routers.graphql, prefix="/graphql")
