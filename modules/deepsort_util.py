from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortWrap:

    def __init__(self, max_age=3, confidence_threshold=None, allowed_objects=None, **kwargs):
        self.confidence_threshold = confidence_threshold
        self.allowed_objects = allowed_objects
        # initialize DeepSORT real-time tracker
        self.deepsort = DeepSort(max_age=max_age, **kwargs)
        # initialize set for track ids 
        self.unique_track_ids = set()
            
    def update_tracks(self, frame, detections, start=None):

        # initialize list for tracker input
        tracker_input = []

        # set up tracker input from detections
        for det in detections:

            # get detected object attributes
            class_id, class_name, confidence, bbox = det

            # filter out weak detections by ensuring the 
            # confidence is greater than the minimum confidence
            if self.confidence_threshold is not None and float(confidence) < self.confidence_threshold:
                continue

            # filter out unwanted objects  
            if self.allowed_objects is not None and class_name not in self.allowed_objects:
                continue

            # if the confidence is greater than the minimum confidence,
            # get the bounding box and the class id
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # add the bounding box (x, y, w, h), confidence and class id to the results list
            tracker_input.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_name])

        ######################################
        # RUN TRACKING

        # update the tracker with the new detections
        tracks = self.deepsort.update_tracks(tracker_input, frame=frame)

        # initialize list for formatted tracker output
        tracking = []

        # list tracking result
        for track in tracks:

            # if the track is not confirmed, ignore it
            if not track.is_confirmed():
                continue

            # get attributes of tracked object
            track_id = track.track_id
            class_name = track.det_class
            confidence = track.det_conf
            bbox = track.to_ltrb()

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
