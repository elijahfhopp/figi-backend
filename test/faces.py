# data = {"image": open("terminator_family.jpg", "rb")}
# res = requests.post("http://localhost:8000/extract_faces", files=data)
# print(res.text)
# print(res.json())
# print(res.status_code)
# print(res.reason)

import numpy

from figi.image.face_extractor import FaceExtractor

extractor = FaceExtractor(".")

with open("terminator_family.jpg", "rb") as file:
    image_bytes = numpy.frombuffer(file.read(), dtype=numpy.uint8)
    faces = extractor.extract_faces_from_array(image_bytes)
    # [print(face.embedding) for face in faces]
    print(faces[0].embedding == faces[1].embedding)
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
