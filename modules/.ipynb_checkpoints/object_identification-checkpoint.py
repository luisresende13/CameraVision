# code for tracking and detection from: https://www.thepythoncode.com/article/real-time-object-tracking-with-yolov8-opencv

import pandas as pd
from IPython.display import clear_output as co
# from ultralytics import YOLO
import cv2
# from deep_sort_realtime.deepsort_tracker import DeepSort
import asyncio
import datetime
from time import time
from flask import request
import pytz
import traceback
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get the Brazil time zone
brazil_tz = pytz.timezone('America/Sao_Paulo')

from modules.video_processing import Video
from modules.yolo_util import YoloWrap
from modules.mediapipe_util import MediapipeDetector, class_names as mediapipe_class_names
from modules.deepsort_util import DeepSortWrap
from modules.motracker_util import CentroidTrackerWrap

object_detection_models = {
    'yolo': YoloWrap("models/yolo/yolov8l.pt"),
    # 'mediapipe': MediapipeDetector(
    #     model_asset_path='models/mediapipe/efficientdet_lite0.tflite',
    #     score_threshold=0.99,
    #     category_allowlist=['apple'],
    #     max_results=None,
    # ),
}

def tracking_reid(
    url,
    model='yolo',
    tracker='deepsort',
    confidence_threshold=0.3,
    iou=0.7,
    allowed_objects=None,
    process_each=1,
    run_detection_each=1,
    post_processing_function=None,
    post_processing_args={},
    frame_annotator=None,
    generator=False,
    to_url=None, #  file path to save optionally annotaded video
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
        class_names = model.class_names
    elif model == 'mediapipe':
        t = time()
        model = MediapipeDetector(
            model_asset_path='models/mediapipe/efficientdet_lite0.tflite',
            score_threshold=confidence_threshold,
            category_allowlist=allowed_objects,
            max_results=None,
        )
        print(f'MEDIAPIPE DETECTOR LOADED · {time() - t} s')
        logging.info(f'MEDIAPIPE DETECTOR LOADED · {time() - t} s')
        class_names = mediapipe_class_names
    else:
        model = object_detection_models['yolo']
        # model = YoloWrap(model)

    # initialize DeepSORT real-time tracker
    max_age = 3
    is_yolo_tracker = False
    if tracker == 'deepsort':
        tracker = DeepSortWrap(max_age, confidence_threshold, allowed_objects)
    elif tracker == 'centroid':
        tracker = CentroidTrackerWrap(class_names=class_names)
    elif tracker == 'yolo':
        is_yolo_tracker = True
        tracker = model
        if allowed_objects is not None:
            allowed_objects = [model.names_ids[name] for name in allowed_objects]
        
    # Get class names from model
    # class_names = model.class_names
    
    # check if video capture is a live http image stream
    is_video_stream = url.startswith('http')

    # if max_frames is None and secs is None and exec_secs is None:
    #     raise "At least one of `max_frames`, `secs` or `exec_secs` should be specified."
    
    # initialize the video capture object
    cap = cv2.VideoCapture(url)
    
    # total frames of video file
    total_frames = None if is_video_stream else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # if capture is from video file

    # if 'CODE' not in url:
        # process_each = 6
    
    # Get the frames per second (fps)
    fps = fps if 'CODE' in url else cap.get(cv2.CAP_PROP_FPS)

    # set the streaming time
    secs = secs if 'CODE' in url else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    
    # Get the frame dimensions (shape)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if to_url is not None:
        # Video writer instance
        video = Video(codec='MP4V', fps=fps, shape=(width, height), overwrite=True)
        WRITER = video.writer(to_url)
    
    # error handler for stream loop
    try:
        
        # initialize post processing output list
        post_processing_output = []

        # get start time reference to measure execution time
        start_time = time()

        # initialize number of frames processed
        n_frames = -1

        # start stream loop
        while True:

            # get the current time and date for Brazil time zone
            start = datetime.datetime.now(brazil_tz)

            # update number of frames processed
            n_frames += 1

            # update video time in seconds
            video_seconds = n_frames / fps

            # break loop if `max_frames` is reached
            if max_frames is not None and n_frames >= max_frames:
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
                logging.info(f'STREAMING · TIME: {round(n_frames / fps, 1)} s · URL: {url}')
                
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
            if n_frames % process_each != 0:
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
                detections = None
                if not is_yolo_tracker:
                    detections = model.detect(frame)

                ######################################
                # RUN TRACKING

                # update the tracker with the new detections
                if is_yolo_tracker:
                    tracking, new_objects = tracker.update_tracks(
                        frame, detections, start,
                        conf=confidence_threshold,
                        iou=iou,
                        classes=allowed_objects,
                    )
                    
                else:
                    tracking, new_objects = tracker.update_tracks(frame, detections, start)

                # add the tracked object ID to the set of unique track IDs
                unique_track_ids = tracker.unique_track_ids

                
            ######################################
            # PROCESS RESULT

            # inference end time 
            end = datetime.datetime.now(brazil_tz)
            end_time = time()

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
                result = model.result if not is_yolo_tracker else tracker.result
                frame = frame_annotator(annotator_input_frame, inference, time_info, resize_shape, ultralytics_result=result)
                    
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
        print(f'STREAMING (EXCEPTION CAUGHT) · ERROR: {str(e)}')
        logging.error(f'STREAMING (EXCEPTION CAUGHT) · ERROR: {str(e)}')
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
        logging.info(f'STREAMING FINISHED · STREAM-TIME: {round(n_frames / fps, 1)} s · EXEC-TIME: {round(time() - start_time, 1)} s · URL {url}')
    
    # return post processing results
    return post_processing_output
