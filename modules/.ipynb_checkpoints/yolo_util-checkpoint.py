import cv2
import numpy as np
import pytz
import traceback
from ultralytics import YOLO
from apiflask import HTTPError
from time import time, sleep
from datetime import datetime as dt

# Get the Brazil time zone
brazil_tz = pytz.timezone('America/Sao_Paulo')

# Video output file parameters
writer_params = {
    "output_file": "yolo-test.mp4",
    "codec": "mp4v",    
    "fps": 3,
}

# Detection and tracking parameters
model_params = {
    "save": False,
    "show": False,
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.7,
    "max_det": 300,
    "vid_stride": 1,
    "device": 'cpu',
    "verbose": True,
    "persist": True,
    "tracker": "botsort.yaml",
}

# Set the expected specifications for the frame
expected_shape = (480, 854)  # (height, width)
expected_channels = 3       # 3 channels for RGB image
expected_dtype = np.uint8     # uint8 data type (8-bit)

# ---
# Run yolo inference for video source

def yolo_watch(
    source='http://187.111.99.18:9004/?CODE=1646',  # video source
    model="yolov8s.pt",  # ultralytics yolov8 model name
    task="track",  # ultralytics yolo task
    model_params=model_params,  # detection or tracking parameters
    max_frames=10,  # number of frames to capture
    seconds=None,  # number of video seconds to capture
    execution_seconds=None,  # maximum exection time in seconds 
    log_seconds=10,  # time between logs in seconds. set it to `None` to supress logging 
    fps=3,  # video frames per second to calculate video stream time
    writer_params=writer_params, # parameters to pass to open-cv video writer instance
    post_processing_function=None,
    post_processing_args=None,
    annotator=None,
    generator=False,
    capture='yolo',
):

    # Load a model
    yolo = YOLO(f'models/{model}')  # load an official detection model

    # Get classes names
    class_ids = {name: class_id for class_id, name in yolo.names.items()}

    # Validate stream time strategy
    if sum([max_frames is not None, seconds is not None, execution_seconds is not None]) > 1:
        raise Exception("YOLO ERROR: Cannot set more than one of `max_frames` ,`seconds` , `execution_seconds`.")

    # Process `objects` custom model parameter
    if "objects" in model_params:
        # if "classes" in model_params:
        #     raise Exception("YOLO ERROR: Cannot set both `objects` and `classes` at the same time.")
        
        # Override `classes` model parameters with `objects` provided value
        objects = model_params['objects']
        del model_params['objects']
        if objects is None:
            model_params["classes"] = None
        else:
            if type(objects) is str:
                if objects:
                    # objects is a comma delimited word list string before splitting
                    objects = objects.split(",")
                else:
                    objects = []
            # objects should be a list by now
            objects = [name.strip().lower() for name in objects]    
            model_params["classes"] = [class_ids[name] for name in objects]

    # Set YOLO V8 model parameters dictionary
    # model_params = {
        # "source": source,
        # "stream": True,
        # **model_params        
    # }

    # Detection and tracking specific settings
    if task == "predict":
        # Select `predict` method
        predict = yolo.predict

        # Filter out tracking specific model parameters
        if "tracker" in model_params:
            del model_params["tracker"]
        if "persist" in model_params:
            del model_params["persist"]
        
    elif task == "track":
        # Select `track` method
        predict = yolo.track


    # Initialize variable for `results` generator
    results = None
        
    # Error handler for stream loop
    try:

        if capture == 'opencv':
            # model_params["stream"] = False
            results = opencv_capture_predict(source, predict, model_params)

        elif capture == 'yolo':
            model_params["source"] = source
            model_params["stream"] = True
            # Perform detection inference
            results = predict(**model_params)

        # Start frame count
        n_frames = 0

        # initialize post processing output list
        post_processing_outputs = []

        # Get start time reference to measure execution time
        start_time = time()

        # Loop through the video frames results
        for result in results:
            # Get result timestamp in Brazil timezone
            timestamp = dt.now(brazil_tz)

            # Start video writer on first frame
            if n_frames == 0 and writer_params is not None:
                # Get the video frame dimensions
                height, width = result.orig_shape
                # Define the output video file
                fourcc = cv2.VideoWriter_fourcc(*writer_params["codec"])
                out = cv2.VideoWriter(writer_params["output_file"], fourcc, writer_params["fps"], (width, height))

            # POST PROCESSING
            # call arbitrary post processing function on frame and detection/tracking outputs
            if post_processing_function is not None:
                post_processing_outputs.append(post_processing_function(result, timestamp, post_processing_outputs, **post_processing_args))

            # ANNOTATE FRAME WITH DETECTION OUTPUTS IF NECESSARY
            need_annotated_image = writer_params is not None or generator
            if need_annotated_image:
                
                    # Annotate image
                    if annotator is not None:
                        # Arbitrary annotator function
                        annotated_image = annotator(result, timestamp, post_processing_outputs, **post_processing_args)
                    else:
                        # Ultralytics default annotated image
                        annotated_image = result.plot()

                    # Assert that the frame is healthy and meets the expected specifications
                    assert_frame_health(annotated_image, expected_shape, expected_channels, expected_dtype)

            # Save the annotated frame to the output video file
            if writer_params is not None:
                out.write(annotated_image)

            # YIELD FRAME IF GENERATOR MODE IS ACTIVE
            if generator:
                ret, buffer = cv2.imencode('.jpg', annotated_image)
                if ret:
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    print("YOLO ERROR: OpenCV Image Encode Error. Skipping image stream...")

            # yields post processsing results
            elif post_processing_function is not None:
                yield post_processing_outputs[-1]


            # Update number of frames
            n_frames += 1

            # update video time in seconds
            video_seconds = n_frames / fps

            # Update execution time in seconds
            exec_seconds = time() - start_time

            # Break loop if `max_frames` is reached
            if max_frames is not None and n_frames >= max_frames:
                break

            # Break loop if execution seconds `exec_secs` is reached
            if execution_seconds is not None and exec_seconds >= execution_seconds: # Exit stream if max execution time exceeds
                break

            # Break loop if video seconds `secs` is reached
            if seconds is not None and video_seconds >= seconds: # Exit stream if max stream time exceeds
                break

            # Log streaming progress
            if log_seconds is not None and video_seconds % log_seconds == 0:
                print(f'STREAMING · N-FRAMES: {n_frames} · VIDEO-TIME: {round(video_seconds, 1)} s · EXECUTION-TIME: {round(exec_seconds, 1)} s · URL: {source}')

        # Report end of stream
        print(f'STREAMING FINISHED · N-FRAMES: {n_frames} · STREAM-TIME: {round(video_seconds, 1)} s · EXECUTION-TIME: {round(exec_seconds, 1)} s · URL {source}')

    # handle exception inside video capture loop
    except Exception as e:
        print(f'STREAMING (EXCEPTION) · ERROR: {str(e)}')
        # Print the traceback to console
        traceback.print_exc()
        # Get the traceback as a string
        traceback_str = traceback.format_exc()        
        raise HTTPError(500, "Internal Server Error During YOLO ULTRALYTICS Video Streaming", traceback_str)

    # finish video capture
    finally:

        # Release ultralytics result generator
        results, post_processing_outputs, annotated_image, yolo = None, None, None, None
        
        # Release the output video file writer
        if writer_params is not None:
            out.release()


# ---
# Capture video source using opencv and run yolo inference            
            
def opencv_capture_predict(source, predict, model_params, max_retries=10):

    # initialize the video capture object
    cap = cv2.VideoCapture(source)

    # error handler for stream loop
    try:
        
        # stream loop
        while True:

            # read video frame
            success, frame = cap.read()

            # retry if read capture not successful
            retries = 0
            while not success and retries < max_retries:
                sleep(0.2)
                cap = cv2.VideoCapture(source)
                success, frame = cap.read()
                retries += 1
                
            # break loop if `max_retries` reached · valid frame is available after here
            if retries == max_retries:
                break

            # Assert that the frame is healthy and meets the expected specifications
            assert_frame_health(frame, expected_shape, expected_channels, expected_dtype)

            # update model parameters source and stream attributes
            model_params["source"] = frame
            model_params["stream"] = False
            
            # run inference on the current frame
            results = predict(**model_params)
            
            # get the frame result
            result = results[0]

            # stream the result
            yield result
    
    # handle exception inside video capture loop
    except Exception as e:
        print(f'OPENCV WRAP STREAMING (EXCEPTION) · ERROR: {str(e)}')
        # Print the traceback to console
        traceback.print_exc()
        # Get the traceback as a string
        traceback_str = traceback.format_exc()        
        raise HTTPError(500, "Internal Server Error During OPEN-CV Video Streaming", traceback_str)
        
    # finish video capture
    finally:
        # release video capture
        cap.release()
        cv2.destroyAllWindows()


# ---
# Assert opencv frame is loaded correctly
        
def assert_frame_health(frame, expected_shape, expected_channels, expected_dtype):
    """
    Asserts that a loaded OpenCV frame is healthy and meets the expected specifications.

    Parameters:
        frame (numpy.ndarray): The loaded OpenCV frame.
        expected_shape (tuple): Tuple specifying the expected shape (height, width) of the frame.
        expected_channels (int): The expected number of channels of the frame (1 for grayscale, 3 for RGB).
        expected_dtype (numpy.dtype): The expected data type of the frame (e.g., np.uint8).

    Raises:
        AssertionError: If the frame fails any of the specified checks.
    """
    assert frame is not None, "Frame is None, image loading failed or invalid image path."
    assert isinstance(frame, np.ndarray), "Frame is not a valid image (not an ndarray)."
    assert frame.shape[:2] == expected_shape, f"Frame shape ({frame.shape[:2]}) does not match the expected shape ({expected_shape})."
    assert len(frame.shape) == 3 and frame.shape[2] == expected_channels, f"Frame does not have the expected number of channels ({expected_channels})."
    assert frame.dtype == expected_dtype, f"Frame data type ({frame.dtype}) does not match the expected data type ({expected_dtype})."

    
# # Example usage:
# if __name__ == "__main__":
#     # Load an example frame (replace 'path_to_image.jpg' with the actual path to your image)
#     image_path = 'path_to_image.jpg'
#     frame = cv2.imread(image_path)

#     # Set the expected specifications for the frame
#     expected_shape = (480, 640)  # (height, width)
#     expected_channels = 3       # 3 channels for RGB image
#     expected_dtype = np.uint8    # uint8 data type (8-bit)

#     try:
#         # Assert that the frame is healthy and meets the expected specifications
#         assert_frame_health(frame, expected_shape, expected_channels, expected_dtype)

#         print("Frame is healthy and meets the expected specifications.")
#     except AssertionError as e:
#         print("Error:", e)


# ---
# YOLO auxiliary functions to get detections, identifications and new identified objects

def detected_objects(result, timestamp):
    """
    Formats the YOLO detection results.

    Args:
        result (object): The detection object.
        timestamp (datetime string): Dict of class names by class id.

    Returns:
        list: Formatted detection results.
    """
    # initialize list for formatted tracker output
    detections = []

    # list tracking result
    boxes = result.boxes
    for class_id, confidence, bbox in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.data.tolist()):
        # Get obeject class name
        class_name = result.names[class_id]
        # Get the bounding box
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # Set detection objectc dictionary
        detection = {
            "timestamp": timestamp,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": bbox
        }
        # Append tracked object attributes
        detections.append(detection)
    
    return detections

def identified_objects(result, timestamp):
    # formatted yolo detections
    detections = detected_objects(result, timestamp)

    # initialize list for formatted tracker output
    tracking = []

    # list tracking result
    if result.boxes.id is not None:
        track_ids = result.boxes.id.tolist()
        for track_id, detection in zip(track_ids, detections):
            tracking.append({"track_id": track_id, **detection})

    return tracking


def new_objects_from(tracking, unique_track_ids):
    ######################################
    # GET NEW IDENTIFIED OBJECTS

    # initialize list for newly detected objects
    new_objects = []

    # loop over the formatted tracks and get newly identified objects
    for track in tracking:

        # check if track ID is unique
        if track["track_id"] not in unique_track_ids:
            # append record to list of new objects
            new_objects.append(track)

            # add the tracked object ID to the set of unique track IDs
            unique_track_ids.append(track["track_id"])
    
    return new_objects, unique_track_ids
