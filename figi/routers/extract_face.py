import numpy
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse

from figi.config import CONFIG
from figi.image.face_extractor import FaceExtractor

extract_face = APIRouter()

extractor = FaceExtractor(CONFIG["FIGI_MODEL_PATH"])


@extract_face.post("/")
async def get_image(image: UploadFile) -> JSONResponse:
    print(image)
    image_bytes = numpy.frombuffer(await image.read(), dtype=numpy.uint8)
    faces = extractor.extract_faces_from_array(image_bytes)
    json_faces = [
        {
            "score": float(face.score),
            "x": int(face.x),
            "y": int(face.y),
            "width": int(face.width),
            "height": int(face.height),
            "embedding": str(face.embedding.tolist()),
        }
        for face in faces
    ]
    return JSONResponse(json_faces)
