import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy

# Many thanks to the (supurb) OpenCV docs: https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html
# Default thresholds/k from there:
# score_threshold = 0.9
# nms_threshold = 0.3
# top_k = 5000


@dataclass
class BoundingBox:
    left: float
    top: float
    x: float
    y: float

    @classmethod
    def from_float_array(cls, list: numpy.ndarray):
        return cls(*list.T)

    def to_float_array(self) -> numpy.ndarray:
        return numpy.array([self.x, self.y, self.left, self.top])


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
    score: float
    bounding_box: BoundingBox

    @classmethod
    def from_raw_result(cls, results: numpy.ndarray):
        return cls(results[14], BoundingBox.from_float_array(results[:4]))


@dataclass
class ExtractedFace:
    face: DetectedFace
    embedding: numpy.ndarray


class FaceExtractor:
    detector: cv2.FaceDetectorYN

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

    # def extract_faces(self, image) -> List[FaceEntry] | None:
    #     size = (image.shape[1], image.shape[0])
    #     self.detector.setInputSize(size)
    #     faces = self.detector.detect(image)
    #     # print(num_faces)
    #     print(repr(faces))
    #     return None
    #     # return faces
    def detect_faces(self, image, size) -> List[DetectedFace]:
        self.detector.setInputSize(size)
        (num_faces, faces) = self.detector.detect(image)

        if num_faces < 1:
            return []

        return [DetectedFace.from_raw_result(face) for face in faces]

    # Embeddings shape is (1,128)
    def extract_face_embeddings(
        self, image: numpy.ndarray, faces: List[DetectedFace]
    ) -> List[ExtractedFace]:
        cropped_image = None
        extracted_faces = []
        for face in faces:
            box = face.bounding_box
            face_image = self.recognizer.alignCrop(
                image, box.to_float_array(), cropped_image
            )
            embedding = self.recognizer.feature(face_image)
            extracted_faces.append(ExtractedFace(face, embedding))
        return extracted_faces
