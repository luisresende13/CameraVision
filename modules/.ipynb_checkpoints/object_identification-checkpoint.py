# code for tracking and detection from: https://www.thepythoncode.com/article/real-time-object-tracking-with-yolov8-opencv

import pandas as pd
from IPython.display import clear_output as co
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import asyncio
import datetime
from time import time
from flask import request
import pytz

# Get the Brazil time zone
brazil_tz = pytz.timezone('America/Sao_Paulo')

from modules.video_processing import Video
from modules.yolo_util import YoloWrap
from modules.mediapipe_util import MediapipeDetector

object_detection_models = {
    'yolo': YoloWrap("yolov8n.pt"),
    # 'mediapipe': MediapipeDetector(
    #     model_asset_path='models/mediapipe/efficientdet_lite0.tflite',
    #     score_threshold=0.9,
    #     category_allowlist=['person'],
    #     max_results=None,
    # ),
}

# initialize DeepSORT real-time tracker
deepsort = DeepSort(max_age=3)

def tracking_reid(
    url,
    model='yolo',
    confidence_threshold=0.3,
    allowed_objects=None,
    max_frames=None,
    post_processing_function=None,
    post_processing_args={},
    proccess_each=1,
    run_detection_each=1,
    frame_annotator=None,
    to_url=None, # save video or annotaded video to file path `to_url`
    generator=False,
    secs=10,
    exec_secs=None,
    log_secs=30,
    fps=3,
    max_retries=5,
):
    
    # initialize object detection model
    if model == 'yolo':
        model = object_detection_models[model]
    elif model == 'mediapipe':
        model = MediapipeDetector(
            model_asset_path='models/mediapipe/efficientdet_lite0.tflite',
            score_threshold=confidence_threshold,
            category_allowlist=allowed_objects,
            max_results=None,
        )
        
    # Get class names from model
    # class_names = model.class_names
    
    # check if video capture is a live http image stream
    is_video_stream = url.startswith('http')

    # initialize the video capture object
    cap = cv2.VideoCapture(url)
    
    # total frames of video file
    total_frames = None if is_video_stream else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # if capture is from video file

    if to_url is not None:
        # Get the frames per second (fps)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get the frame dimensions (shape)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video writer instance
        video = Video(codec='MP4V', fps=fps, shape=(w, h), overwrite=True)
        WRITER = video.writer(to_url)

    # initialize set for track ids 
    unique_track_ids = set()

    # initialize post processing output list
    post_processing_output = []
    
    # get start time reference to measure execution time
    start_time = time()
    
    # error handler for stream loop
    try:
        
        # start stream loop
        n_frames = -1
        while True:

            # update number of frames and seconds
            n_frames += 1
            video_seconds = n_frames / fps

            # break loop if `max_frames` is reached
            if max_frames is not None and max_frames == n_frames:
                break

            # break loop if execution seconds `exec_secs` is reached
            if exec_secs is not None and time() - start_time >= exec_secs: # Exit stream if max execution time exceeds
                break

            # break loop if video seconds `secs` is reached
            if secs is not None and video_seconds >= secs: # Exit stream if max stream time exceeds
                break

            # log streaming progress
            if log_secs is not None and video_seconds % log_secs == 0:
                print(f'STREAMING · TIME: {round(n_frames / fps, 1)} s · URL: {url}')

            # get the current time and date for Brazil time zone
            start = datetime.datetime.now(brazil_tz)

            # read video frame
            success, frame = cap.read()
            # if not success:
            #     break

            # retry if read capture not successful
            retries = 0
            while not success and retries < max_retries:
                cap = cv2.VideoCapture(url)
                success, frame = cap.read()
                retries += 1
                
            # break loop if `max_retries` reached · valid frame is available after here
            if retries == max_retries:
                break

            # continue if `n_frames` is not zero nor a multiple of `process_each`. continue for n_frames = 0
            if n_frames % proccess_each != 0:
                continue

            # skip only detection and tracking if `n_frames` is not zero nor a multiple of `run_detection_each`. skip for n_frames = 0
            if n_frames % run_detection_each == 0:

                ######################################
                # RUN DETECTION · Obs. Choose standard model method for prediction and wrap models that use other methods before passing then to the function.

                # formatted yolo detections
                detections = model.detect(frame)

                # initialize list for tracker input
                tracker_input = []

                # set up tracker input from detections
                for det in detections:

                    # get detected object attributes
                    class_name, confidence, bbox = det

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
                    tracker_input.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_name])

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
                    if track_id not in unique_track_ids:

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
                        unique_track_ids.add(track_id)

            ######################################
            # PROCESS RESULT

            # end time to compute the fps
            end = datetime.datetime.now(brazil_tz)

            # ANNOTATE FRAME WITH DETECTION OUTPUTS
            if frame_annotator is not None:
                annotated_frame = frame_annotator(frame, detections, tracking, new_objects, start, end)

            # WRITE FRAME TO VIDEO FILE
            if to_url is not None:
                selected_frame = frame if frame_annotator is None else annotated_frame
                WRITER.write(selected_frame)

            # POST PROCESSING
            # call arbitrary post processing function on frame and on detection and tracking outputs
            if post_processing_function is not None:
                post_processing_output.append(post_processing_function(frame, detections, tracking, new_objects, start, end, **post_processing_args))

            # YIELD FRAME IF IN GENERATOR MODE
            if generator:
                selected_frame = frame if frame_annotator is None else annotated_frame
                ret, buffer = cv2.imencode('.jpg', selected_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    except Exception as e:
        print(f'STREAMING (EXCEPTION CAUGHT) · ERROR: {e}')

    finally:
        
        if to_url is not None:
            # Optional · release video writer
            WRITER.release() # run after writing is finished

        # release video capture
        cap.release()
        cv2.destroyAllWindows()

        # Report end of stream 
        print(f'STREAMING FINISHED · STREAM-TIME: {round(n_frames / fps, 1)} s · EXEC-TIME: {round(time() - start_time, 1)} s · URL {url}')
                
        # return post processing results
        return post_processing_output
