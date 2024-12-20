import os
from typing import Any, Dict

from dotenv import dotenv_values

CONFIG: Dict[str, Any] = {
    "FIGI_MAX_VECTOR_SEARCH_LIMIT": 100,
    "FIGI_IMAGES_PATH": "images",
    "FIGI_MODEL_PATH": ".",
    "FIGI_INDEX": False,
    **dotenv_values(".env"),
    **os.environ,
}
