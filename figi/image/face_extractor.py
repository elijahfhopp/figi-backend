import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy

# From OpenCV docs (https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html):
# faces	detection results stored in a 2D cv::Mat of shape [num_faces, 15]
#     0-1: x, y of bbox top left corner
#     2-3: width, height of bbox
#     4-5: x, y of right eye (blue point in the example image)
#     6-7: x, y of left eye (red point in the example image)
#     8-9: x, y of nose tip (green point in the example image)
#     10-11: x, y of right corner of mouth (pink point in the example image)
#     12-13: x, y of left corner of mouth (yellow point in the example image)
#     14: face score


@dataclass
class DetectedFace:
    markers: numpy.ndarray

    @property
    def score(self):
        return self.markers[14]

    @property
    def x(self):
        return self.markers[0]

    @property
    def y(self):
        return self.markers[1]

    @property
    def width(self):
        return self.markers[2]

    @property
    def height(self):
        return self.markers[3]


@dataclass
class ExtractedFace(DetectedFace):
    embedding: numpy.ndarray

    @classmethod
    def from_face(cls, face: DetectedFace, embedding: numpy.ndarray):
        return cls(face.markers, embedding)


class FaceExtractor:
    detector: cv2.FaceDetectorYN
    recognizer: cv2.FaceRecognizerSF

    def __init__(
        self,
        model_path: str,
    ):
        detector_model_path = os.path.join(
            model_path, "face_detection_yunet_2023mar.onnx"
        )
        self.detector = cv2.FaceDetectorYN.create(
            detector_model_path,
            "",
            (0, 0),
            score_threshold=0.8,
            nms_threshold=0.3,
            top_k=5000,
        )

        recognizer_model_path = os.path.join(
            model_path, "face_recognition_sface_2021dec.onnx"
        )
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_model_path, "")

    def extract_faces(self, image_path) -> List[ExtractedFace]:
        image = cv2.imread(image_path)
        size = (image.shape[1], image.shape[0])
        faces = self.detect_faces(image, size)
        return self.extract_face_embeddings(image, faces)

    def extract_faces_from_array(
        self, image_bytes: numpy.ndarray
    ) -> List[ExtractedFace]:
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        size = (image.shape[1], image.shape[0])
        faces = self.detect_faces(image, size)
        return self.extract_face_embeddings(image, faces)

    def detect_faces(self, image, size) -> List[DetectedFace]:
        self.detector.setInputSize(size)
        (num_faces, faces) = self.detector.detect(image)

        if num_faces < 1 or faces is None:
            return []

        return [DetectedFace(face) for face in faces]

    # Embeddings shape is (128,)
    def extract_face_embeddings(
        self, image: numpy.ndarray, faces: List[DetectedFace]
    ) -> List[ExtractedFace]:
        facecrop = []
        extracted_faces = []
        for face in faces:
            bbox = numpy.array([face.x, face.y, face.width, face.height])
            facecrop.append(bbox)
            face_image = self.recognizer.alignCrop(image, face.markers)
            # recognizer outputs (1, 128)
            embedding = self.recognizer.feature(face_image)
            # print(embedding)
            # to (128,)
            embedding = embedding.reshape(128)
            extracted_faces.append(ExtractedFace.from_face(face, embedding))
        return extracted_faces
