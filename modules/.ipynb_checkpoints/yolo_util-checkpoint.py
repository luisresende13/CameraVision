from ultralytics import YOLO

class YoloWrap:
    
    def __init__(self, model="yolov8n.pt"):
        
        # initialize YOLO object detection model
        self.model = YOLO(model)
        self.class_names = self.model.names
    
    def detect(self, frame):
        # run the YOLO model on the frame
        detections = self.model(frame)[0]

        # formatted yolo detections
        detections = self.formatted_detections(detections)

        # return standard format detections
        return detections
    
    def formatted_detections(self, detections):
        """
        Formats the YOLO detection results.

        Args:
            detection (object): The detection object.
            class_names (dict): Dict of class names by class id.

        Returns:
            list: Formatted detection results.
        """
        formatted_detection = []

        for data in detections.boxes.data.tolist():
            # Get the bounding box and the class id
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

            # Extract the confidence (i.e., probability) associated with the prediction
            confidence = data[4]

            # Get class id and name
            class_id = int(data[5])
            class_name = self.class_names[class_id]

            # Get bounding box
            bbox = [xmin, ymin, xmax, ymax]

            # Add standard format detections
            formatted_detection.append([class_id, class_name, confidence, bbox])

        return formatted_detection
