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
    def fromFloatArray(cls, list: numpy.ndarray):
        return cls(*list.T)


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
    def fromRawResult(cls, results: numpy.ndarray):
        return cls(results[14], BoundingBox.fromFloatArray(results[:4]))


@dataclass
class FaceEntry:
    index_in_image: int
    embedding: numpy.ndarray
    detection_data: List[float]

    @property
    def bounding_box(self) -> List[float]:
        return self.detection_data[0:3]

    @property
    def landmarks(self) -> List[float]:
        return self.detection_data[3:]


class EmbeddingsGenerator:
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

        return [DetectedFace.fromRawResult(face) for face in faces]

    def detect_faces_in_file(self, image_path) -> List[DetectedFace]:
        image = cv2.imread(image_path)
        size = (image.shape[1], image.shape[0])
        return self.detect_faces(image, size)
