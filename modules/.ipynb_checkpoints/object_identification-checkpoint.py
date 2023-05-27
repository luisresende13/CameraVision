# code for tracking and detection from: https://www.thepythoncode.com/article/real-time-object-tracking-with-yolov8-opencv

import pandas as pd
from IPython.display import clear_output as co
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import asyncio
import datetime
import pytz

# Get the Brazil time zone
brazil_tz = pytz.timezone('America/Sao_Paulo')

from modules.yolo_util import formatted_yolo_detection
from modules.video_processing import Video

def tracking_reid(
    video_path,
    confidence_threshold=0.3,
    allowed_objects=None,
    max_frames=10,
    post_processing_function=None,
    post_processing_args={},
    proccess_each=1,
    run_detection_each=1,
    frame_annotator=None,
    to_video_path=None,
    generator=False
):
    
    # initialize YOLO object detection model
    model = YOLO("yolov8n.pt")

    # Get class names from model
    class_names = model.names

    # initialize DeepSORT real-time tracker
    deepsort = DeepSort(max_age=6)
        
    # initialize the video capture object
    video_cap = cv2.VideoCapture(video_path)
    
    # check if video capture is a live http image stream
    is_video_stream = video_path.startswith('http')

    # total frames of video file
    total_frames = None if is_video_stream else int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) # if capture is from video file

    if to_video_path is not None:
        # Get the frames per second (fps)
        fps = video_cap.get(cv2.CAP_PROP_FPS)

        # Get the frame dimensions (shape)
        w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video writer instance
        video = Video(codec='MP4V', fps=fps, shape=(w, h), overwrite=True)
        WRITER = video.writer(to_video_path)

    # initialize set for track ids 
    unique_track_ids = set()

    # initialize post processing output list
    post_processing_output = []
    
    i = -1
    while True:

        # update number of frames processed or skipped
        i += 1
        
        # break loop if `max_frames` are processed
        if max_frames is not None and max_frames == i:
            break

        # Get the current date and time in the Brazil time zone
        start = datetime.datetime.now(brazil_tz)

        # Convert the datetime to a string rounded to seconds
        timestamp = start.strftime('%Y-%m-%d %H:%M:%S')

        # read video frame
        ret, frame = video_cap.read()
        if not ret:
            break
        
        # continue if `i` is not zero nor a multiple of `process_each`. continue for i = 0
        if i % proccess_each != 0:
            continue
            
        # skip detection and tracking if `i` is not zero nor a multiple of `run_detection_each`. skip for i = 0
        if i % run_detection_each == 0:

            ######################################
            # RUN DETECTION · Obs. Choose standard model method for prediction and wrap models that use other methods before passing then to the function.

            # run the YOLO model on the frame
            yolo_detection = model(frame)[0]

            # formatted yolo detections
            detections = formatted_yolo_detection(yolo_detection, class_names=class_names)

            # initialize list for tracker input
            tracker_input = []

            # set up tracker input from detections
            for det in detections:

                # get detected object attributes
                class_id, class_name, confidence, bbox = det

                # filter out weak detections by ensuring the 
                # confidence is greater than the minimum confidence
                if float(confidence) < confidence_threshold:
                    continue

                # filter out unwanted objects  
                if allowed_objects is not None and class_name not in allowed_objects:
                    continue

                # if the confidence is greater than the minimum confidence,
                # get the bounding box and the class id
                xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # add the bounding box (x, y, w, h), confidence and class id to the results list
                tracker_input.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

            ######################################
            # RUN TRACKING

            # update the tracker with the new detections
            tracks = deepsort.update_tracks(tracker_input, frame=frame)

            # initialize list for formatted tracker output
            tracking = []

            # list tracking result
            for track in tracks:

                # if the track is not confirmed, ignore it
                if not track.is_confirmed():
                    continue

                # get attributes of tracked object
                track_id = track.track_id
                class_label = track.det_class
                class_name = class_names[class_label]
                confidence = track.det_conf
                bbox = track.to_ltrb()

                # append attributes of tracked objects
                tracking.append([track_id, class_label, class_name, confidence, bbox, start])

            ######################################
            # GET NEW IDENTIFIED OBJECTS

            # initialize list for newly detected objects
            new_objects = []

            # loop over the formatted tracks and get newly identified objects
            for track in tracking:

                # get track attributes
                track_id, class_label, class_name, confidence, bbox, timestamp = track

                # check if track ID is unique
                if track_id not in unique_track_ids:

                    # prepare record of newly identified object
                    record = {
                        'class_label': class_label,
                        'class_name': class_name,
                        'confidence': confidence,
                        'timestamp': timestamp,
                        'track_id': track_id,
                        'bbox': list(bbox),
                    }

                    # append record to list of new objects
                    new_objects.append(record)

                    # add the tracked object ID to the set of unique track IDs
                    unique_track_ids.add(track_id)

        ######################################
        # PROCESS RESULT
        
        # end time to compute the fps
        end = datetime.datetime.now(brazil_tz)
        
        # ANNOTATE FRAME WITH DETECTION OUTPUTS
        if frame_annotator is not None:
            annotated_frame = frame_annotator(frame, detections, tracking, new_objects, start, end)
            
        # WRITE FRAME TO VIDEO FILE
        if to_video_path is not None:
            selected_frame = frame if frame_annotator is None else annotated_frame
            WRITER.write(selected_frame)
        
        # POST PROCESSING
        # call arbitrary post processing function on frame and on detection and tracking outputs
        if post_processing_function is not None:
            post_processing_output.append(post_processing_function(frame, detections, tracking, new_objects, start, end, **post_processing_args))

        # report progress and time to process the current frame
        # if total_frames is not None:
            # co(True); print(f"Time to process frame {i}/{total_frames}: {(end - start).total_seconds() * 1000:.0f} milliseconds")

        # YIELD FRAME IN GENERATOR MODE
        if generator:
            selected_frame = frame if frame_annotator is None else annotated_frame
            ret, buffer = cv2.imencode('.jpg', selected_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
    if to_video_path is not None:
        # Optional · release video writer
        WRITER.release() # run after writing is finished

    # release video capture
    video_cap.release()
    cv2.destroyAllWindows()
    
    # return post processing results
    return post_processing_output
