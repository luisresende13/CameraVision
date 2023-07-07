from ultralytics import YOLO
import cv2

class YoloWrap:
    
    def __init__(self, model="yolov8x.pt"):
        
        # initialize YOLO object detection model
        self.model = YOLO(model)
        self.class_names = self.model.names
        self.names_ids = {name: _id for _id, name in self.model.names.items()}
        # initialize set for track ids 
        self.unique_track_ids = set()
        self.device = 'cpu'  # gpu device
        self.result = None
        # is_cuda = cv2.cuda.getCudaEnabledDeviceCount()  > 0
        # self.device = 0 if is_cuda else 'cpu'
        # print(f'YOLO Ultralytics model initializing · IS-DEVICE-CUDA: {is_cuda}')

    def detect(self, frame):
        # run the YOLO model on the frame
        results = self.model.predict(
            frame, save=False, show=False,
            imgsz=640,
            conf=0.3, iou=0.7,
            max_det=300,  # vid_stride=0,
            stream=False, device=self.device, verbose=False,
        )

        result = results[0]
        self.result = result
        
        # formatted yolo detections
        detections = self.formatted_detections(result)

        # return standard format detections
        return detections
    
    def update_tracks(self, frame, detections, start, conf=0.3, iou=0.7, classes=None):
        
        ######################################
        # RUN TRACKING

        results = self.model.track(
            frame, save=False, show=False,
            classes=classes,
            conf=conf,
            imgsz=640,
            iou=iou,
            max_det=300,
            vid_stride=0,
            stream=False,
            device=self.device,
            persist=True,
            verbose=False,
        )  # , tracker="bytetrack.yaml")

        result = results[0]
        self.result = result

        # formatted yolo detections
        # detections = self.formatted_detections(result)

        # initialize list for formatted tracker output
        tracking = []

        # list tracking result
        # print(result.boxes)
        boxes = result.boxes
        if boxes.id is not None:
            for track_id, class_id, confidence, bbox in zip(boxes.id.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), boxes.data.tolist()):
                class_name = result.names[class_id]
                # Get bounding box
                xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                bbox = [xmin, ymin, xmax, ymax]
                # append attributes of tracked objects
                tracking.append([track_id, class_name, confidence, bbox])
        
        ######################################
        # GET NEW IDENTIFIED OBJECTS

        # initialize list for newly detected objects
        new_objects = []

        # loop over the formatted tracks and get newly identified objects
        for track in tracking:

            # get track attributes
            track_id, class_name, confidence, bbox = track

            # check if track ID is unique
            if track_id not in self.unique_track_ids:

                # prepare record of newly identified object
                record = {
                    # 'class_label': class_label,
                    'class_name': class_name,
                    'confidence': confidence,
                    'timestamp': start,
                    'track_id': track_id,
                    'bbox': list(bbox),
                }

                # append record to list of new objects
                new_objects.append(record)

                # add the tracked object ID to the set of unique track IDs
                self.unique_track_ids.add(track_id)
                
        return tracking, new_objects

    
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
