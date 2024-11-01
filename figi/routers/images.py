import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from figi.config import CONFIG
from figi.db.models import ImagesModel

images = APIRouter()


@images.get("/{image_id}")
async def get_image(image_id: int) -> FileResponse:
    print(image_id)
    image_model: ImagesModel = ImagesModel.get_or_none(ImagesModel.id == image_id)
    if not image_model:
        raise HTTPException(
            status_code=404,
            detail=f"No image record of id {image_id} found on database.",
        )

    root = CONFIG["FIGI_IMAGES_PATH"]
    path = os.path.join(root, str(image_model.path))
    return FileResponse(path=path)
