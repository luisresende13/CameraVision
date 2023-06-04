from motrackers import CentroidTracker, IOUTracker, CentroidKF_Tracker

class CentroidTrackerWrap:

    def __init__(self, confidence_threshold=None, allowed_objects=None, class_names=None, **kwargs):
        self.confidence_threshold = confidence_threshold
        self.allowed_objects = allowed_objects
        self.class_names = class_names
        # initialize DeepSORT real-time tracker
        self.tracker = IOUTracker(max_lost=16) #, tracker_output_format='mot_challenge') # CentroidTracker, IOUTracker, CentroidKF_Tracker(...), SORT(...)
        # initialize set for track ids 
        self.unique_track_ids = set()
            
    def update_tracks(self, frame, detections, start=None):

        # Extract bounding boxes, confidences, and class IDs from the detections
        detection_bboxes = []
        detection_confidences = []
        detection_class_ids = []

        for detection in detections:

            # get detected object attributes
            class_id, class_name, confidence, bbox = detection
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            
            # scores = detection[5:]
            # class_id = scores.argmax()
            # confidence = scores[class_id]

            if confidence > 0.5:
                # center_x = int(detection[0] * width)
                # center_y = int(detection[1] * height)
                # bbox_width = int(detection[2] * width)
                # bbox_height = int(detection[3] * height)

                # Convert center coordinates to top-left coordinates
                # bbox_left = int(center_x - bbox_width / 2)
                # bbox_top = int(center_y - bbox_height / 2)

                detection_bboxes.append([bbox[0], bbox[1], bbox_width, bbox_height])
                detection_confidences.append(confidence)
                detection_class_ids.append(class_id)

        # Update the tracker with the detection results
        output_tracks = self.tracker.update(detection_bboxes, detection_confidences, detection_class_ids)

        # initialize list for formatted tracker output
        tracking = []

        # Draw bounding boxes and annotations on the frame
        for track in output_tracks:
            frame, track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence, _, _, _ = track
            # class_name = self.class_names[class_id]

            bbox = (bbox_left, bbox_top, bbox_left + bbox_width, bbox_top + bbox_height)
            # append attributes of tracked objects
            tracking.append([track_id, '', confidence, bbox])

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
