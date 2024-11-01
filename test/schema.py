import logging
import sys
import warnings

from figi.graphql.schema import load_image_by_id

from rich.logging import RichHandler

from tqdm import TqdmExperimentalWarning

sys.path.append("../")


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%Y-%m-%dT%H:%M:%S%Z]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("figi.main")

# Stupid, but necessary integration test to validate
# that caching worked.
load_image_by_id(1)
load_image_by_id(1)
load_image_by_id(1)
load_image_by_id(1)
load_image_by_id(1)
load_image_by_id(1)
load_image_by_id(1)
load_image_by_id(1)
