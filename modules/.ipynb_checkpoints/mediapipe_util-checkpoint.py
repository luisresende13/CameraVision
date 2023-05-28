
# LOAD OBJECT DETECTOR MODEL - METHOD #1

import mediapipe as mp
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class MediapipeDetector:
    
    def __init__(self, model_asset_path='models/mediapipe/efficientdet_lite0.tflite', score_threshold=0.0, max_results=None, category_allowlist=[]):
        model_asset_buffer = open(model_asset_path, 'rb').read()
        base_options = BaseOptions(
            model_asset_buffer=model_asset_buffer,
        )
        options = ObjectDetectorOptions(
            base_options=base_options,
            max_results=max_results,
            score_threshold=score_threshold,
            running_mode=VisionRunningMode.IMAGE,
            category_allowlist=category_allowlist,
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
        self.class_names = class_names
        
    def detect(self, frame):
        # Load the input image.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Detect objects in the input image.
        detection_result = self.detector.detect(image)

        # Formatted mediapipe object detection
        detections = self.format_mediapipe_detection(detection_result)
        
        # return mediapipe detections
        return detections
        
    def format_mediapipe_detection(self, detection_result):
        detections = []
        for detection in detection_result.detections:
            ctgr = detection.categories[0]
            class_name = ctgr.category_name
            confidence = ctgr.score
            bbox = detection.bounding_box

            x_min, y_min = bbox.origin_x, bbox.origin_y
            width, height = bbox.width, bbox.height
            x_max, y_max = x_min + width, y_min + height
            bbox = [x_min, y_min, x_max, y_max]
            detections.append([class_name, confidence, bbox])

        return detections
        
class_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
    4: 'airplane', 5: 'bus', 6: 'train',
    7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
    13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep',
    19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie',
    28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball',
    33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup',
    42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple',
    48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv',
    63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}

class_ids = {value: key for key, value in class_names.items()}