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
import traceback

# Get the Brazil time zone
brazil_tz = pytz.timezone('America/Sao_Paulo')

from modules.video_processing import Video
from modules.yolo_util import YoloWrap
from modules.mediapipe_util import MediapipeDetector

object_detection_models = {
    'yolo': YoloWrap("yolov8n.pt"),
    'mediapipe': MediapipeDetector(
        model_asset_path='models/mediapipe/efficientdet_lite0.tflite',
        score_threshold=0.99,
        category_allowlist=['apple'],
        max_results=None,
    ),
}

def tracking_reid(
    url,
    model='yolo',
    confidence_threshold=0.3,
    allowed_objects=None,
    proccess_each=1,
    run_detection_each=1,
    post_processing_function=None,
    post_processing_args={},
    frame_annotator=None,
    generator=False,
    to_url=None, # save video or annotaded video to file path `to_url`
    max_frames=None,
    secs=10,
    exec_secs=None,
    log_secs=30,
    fps=3,
    max_retries=5,
    resize_shape=(300, 300),
    stream_shape=(),
):
    
    # initialize detection model instance
    if model == 'yolo':
        model = object_detection_models[model]
    elif model == 'mediapipe':
        t = time()
        model = MediapipeDetector(
            model_asset_path='models/mediapipe/efficientdet_lite0.tflite',
            score_threshold=confidence_threshold,
            category_allowlist=allowed_objects,
            max_results=None,
        ); print(f'MEDIAPIPE DETECTOR LOADED · {time() - t} s')

    # initialize DeepSORT real-time tracker
    deepsort = DeepSort(max_age=3)

    # Get class names from model
    # class_names = model.class_names
    
    # check if video capture is a live http image stream
    is_video_stream = url.startswith('http')

    if max_frames is None and secs is None and exec_secs is None:
        raise "At least one of `max_frames`, `secs` or `exec_secs` should be specified."
    
    # initialize the video capture object
    cap = cv2.VideoCapture(url)
    
    # total frames of video file
    total_frames = None if is_video_stream else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # if capture is from video file

    # Get the frames per second (fps)
    fps = fps if fps is not None else cap.get(cv2.CAP_PROP_FPS)

    # Get the frame dimensions (shape)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if to_url is not None:
        # Video writer instance
        video = Video(codec='MP4V', fps=fps, shape=(width, height), overwrite=True)
        WRITER = video.writer(to_url)
    
    # error handler for stream loop
    try:
        
        # initialize set for track ids 
        unique_track_ids = set()

        # initialize post processing output list
        post_processing_output = []

        # get start time reference to measure execution time
        start_time = time()

        # initialize number of frames processed
        n_frames = 0

        # start stream loop
        while True:

            # get the current time and date for Brazil time zone
            start = datetime.datetime.now(brazil_tz)

            # update video time in seconds
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

                if resize_shape is not None:
                    # Resize the image using the specified width and height
                    original_frame = frame.copy()
                    frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)
                
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

            # inference end time 
            end = datetime.datetime.now(brazil_tz)
            end_time = time()

            # update number of frames processed
            n_frames += 1

            # JOIN INFERENCE OUTPUTS AND TIME METADATA
            inference = (detections, tracking, new_objects)
            time_info = (n_frames, start, end, start_time, end_time)
            
            # POST PROCESSING
            # call arbitrary post processing function on frame and on detection and tracking outputs
            if post_processing_function is not None:
                post_processing_output.append(post_processing_function(frame, inference, time_info, **post_processing_args))

            # ANNOTATE FRAME WITH DETECTION OUTPUTS
            if frame_annotator is not None:
                annotator_input_frame = frame if resize_shape is None else original_frame
                frame = frame_annotator(annotator_input_frame, inference, time_info, resize_shape)

            # # Resize the image using the specified width and height
            # if resize_shape is not None:
            #     frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # WRITE FRAME TO VIDEO FILE
            if to_url is not None:
                WRITER.write(frame)

            # YIELD FRAME IF IN GENERATOR MODE
            if generator:
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    # handle exception inside video capture loop
    except Exception as e:
        print(f'STREAMING (EXCEPTION) · ERROR: {str(e)}')
        traceback.print_exc()
        
    # finish video capture
    finally:
        
        if to_url is not None:
            # release video writer after file is finished
            WRITER.release()

        # release video capture
        cap.release()
        cv2.destroyAllWindows()

        # Report end of stream 
        print(f'STREAMING FINISHED · STREAM-TIME: {round(n_frames / fps, 1)} s · EXEC-TIME: {round(time() - start_time, 1)} s · URL {url}')
    
    # return post processing results
    return post_processing_output
