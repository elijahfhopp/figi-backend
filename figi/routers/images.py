import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from figi.config import CONFIG
from figi.db.models import ImagesModel

images = APIRouter()


@images.get("/{image_id}")
async def get_image(id: int) -> FileResponse:
    image_model: ImagesModel = ImagesModel.get_by_id(id)
    if not image_model:
        raise HTTPException(
            status_code=404, detail=f"No image record of id {id} found on database."
        )

    root = CONFIG["IMAGES_PATH"]
    path = os.path.join(root, str(image_model.path))
    return FileResponse(path=path)
