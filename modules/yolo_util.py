import cv2
import numpy as np
import pytz
import traceback
from ultralytics import YOLO
from apiflask import HTTPError
from time import time, sleep
from datetime import datetime as dt
import torch

# Custom python modules
from modules.bigquery_util import get_camera_from_bq_table
from modules.post_processing import default_post_processing, bigquery_post_new_objects, trigger_post_url_new_objects, bigquery_post_and_trigger_new_objects, fps_annotator

# Get the Brazil time zone
brazil_tz = pytz.timezone('America/Sao_Paulo')

# Set cuda device
# torch._C._cuda_setDevice(0)
# torch._C._cuda_init()
# torch._C._cuda_emptyCache()
# torch._C._cuda_setDevice(-1)

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
    # "persist": True,
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
    retries=5,
    retry_delay=2,  # seconds
):
    
    # Load a model
    if isinstance(model, str):
        yolo = YOLO(f'models/{model}')  # load an official detection model
    else:
        # print('REUTILIZING EXISTING YOLO MODEL')
        yolo = model
        
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

    # Iterate retry loop
    for retry in range(1, retries + 1):
        
        # Error handler for stream loop
        try:

            # reLoad the model
            if retry > 1:
                if isinstance(model, str):
                    yolo = YOLO(f'models/{model}')  # load an official detection model

            # Detection and tracking specific settings
            if task == "predict":
                # Select `predict` method
                predict = yolo.predict
                
                # Filter out tracking specific model parameters
                if "tracker" in model_params:
                    del model_params["tracker"]
                # if "persist" in model_params:
                    # del model_params["persist"]

            elif task == "track":
                # Select `track` method
                predict = yolo.track

            # Start time counting
            video_seconds = exec_seconds = 0
                
            # Initialize variable for `results` generator
            results = None            
    
            if capture == 'opencv':
                # model_params["persist"] = True
                results = opencv_capture_predict(source, predict, model_params)

            elif capture == 'yolo':
                model_params["source"] = source
                model_params["stream"] = True
                # model_params["persist"] = False
                # Perform detection inference
                results = predict(**model_params)

            # Start frame count
            n_frames = 0

            # initialize post processing output list
            post_processing_outputs = []

            # Initialize `start_time` variable
            start_time = None

            # Loop through the video frames results
            for result in results:
                # Get start time reference to measure execution time
                if start_time is None:
                    start_time = time()

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
                
                # Clear result when done with it
                result = None
                
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
            print(f'STREAMING SUCCESS · N-FRAMES: {n_frames} · STREAM-TIME: {round(video_seconds, 1)} s · EXECUTION-TIME: {round(exec_seconds, 1)} s · URL {source}')

            # Simulating an error for demonstration purposes (remove this in your actual code)
            # if retry < retries:
                # raise ValueError("Simulated error on retry")

            # If the code succeeds, clear GPU memory
            try:
                torch.cuda.empty_cache()
            except:
                print("FAILED TO CLEAR TORCH CUDA CACHE AFTER SUCCESS.")

            # If the code succeeds, break out of the loop
            break


        # handle exception inside video capture loop
        except Exception as e:
            print(f'STREAMING EXCEPTION · ATTEMPT: {retry}/{retries} · DELAY: {retry_delay} s · SOURCE: {source}')

            # If the code fails, clear GPU memory
            try:
                torch.cuda.empty_cache()
            except:
                print("FAILED TO CLEAR TORCH CUDA CACHE AFTER EXCEPTION.")
            
            if retry < retries:
                sleep(retry_delay)
            else:
                # Print the traceback to console
                traceback.print_exc()
                # Get the traceback as a string
                traceback_str = traceback.format_exc()
                raise HTTPError(500, "Internal Server Error During YOLO ULTRALYTICS Video Streaming", traceback_str)
        
        finally:
            # Release ultralytics result generator
            results, post_processing_outputs, annotated_image = None, None, None
            
            if isinstance(model, str):
                yolo = None
            
            # Destroy cv2 windows
            cv2.destroyAllWindows()

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
        # print(f'OPENCV WRAP STREAMING (EXCEPTION) · ERROR: {str(e)}')
        
        # Print the traceback to console
        # traceback.print_exc()
        # Get the traceback as a string
        traceback_str = traceback.format_exc()        
        raise HTTPError(500, "Internal Server Error During OPEN-CV Video Streaming", traceback_str)
        
    # finish video capture
    finally:
        # release video capture
        cap.release()
        cv2.destroyAllWindows()

        
# ---
# Predict with yolo for either `source` value or registere `camera_id`
        
post_processing_functions_dict = {
    'none': None,
    'console-log': default_post_processing,
    'bigquery': bigquery_post_new_objects,
    'trigger': trigger_post_url_new_objects,
    'bigquery-trigger': bigquery_post_and_trigger_new_objects,
}

annotators_dict = {
    'none': None,
    'fps': fps_annotator,
}

# Assuming that 'metadata' is a dictionary containing metadata for the fields
    
def yolo_watch_camera(
    source=None,
    camera_id=None,
    post_url=None,
    post_scheme=None,
    model="yolov8l.pt",
    task="track",
    max_frames=None,
    seconds=None,
    execution_seconds=None,
    log_seconds=10,
    fps=3,
    process="none",
    annotator="none",
    capture="opencv",
    stream=False,
    retries=1,
    retry_delay=1.0,
    objects=None,
    classes=None,
    conf=0.3,
    iou=0.7,
    max_det=300,
    vid_stride=1,
    imgsz=640,
    device="gpu",
    tracker="botsort.yaml",
    persist=True,
    augment=False,
    save=False,
    show=False, 
    verbose=False
):
    """
    Perform YOLO camera watch with the specified arguments.

    Parameters:
    - source (str): The source of the video stream.
    - camera_id (int): The ID of the camera.
    - post_url (str): URL to send POST requests from inference results.
    - post_scheme (str): JSON schema to send as the body of the POST request to `post_url` from inference results.
    - model (str or YOLO instance): The YOLO model file to use or the YOLO model instance. Default is "yolov8l.pt".
    - task (str): The YOLO task to perform, either "predict" or "track". Default is "track".
    - max_frames (int): Maximum number of frames to process.
    - seconds (int): The number of seconds to run the watch process.
    - execution_seconds (int): The maximum number of seconds for inference execution.
    - log_seconds (int): Number of seconds to log the output.
    - fps (int): Frames per second for video playback.
    - process (str): Process mode, one of "none", "console-log", "bigquery", "trigger", or "bigquery-trigger".
    - annotator (str): Annotator mode, either "none" or "fps".
    - capture (str): Capture mode, either "opencv" or "yolo".
    - stream (bool): Whether to stream the video or not.
    - retries (int): Number of retries for inference.
    - retry_delay (float): Delay in seconds between retries.
    - objects (List[str]): List of objects to detect.
    - classes (List[int]): List of classes to detect.
    - conf (float): Confidence threshold for detections.
    - iou (float): IOU (Intersection over Union) threshold for detections.
    - max_det (int): Maximum number of detections.
    - vid_stride (int): Stride for processing frames from a video.
    - imgsz (int): Size of the input images for inference.
    - device (str): Device for running inference, either "cpu" or "gpu". Default is "gpu".
    - tracker (str): The tracker configuration file to use, either "botsort.yaml" or "bytetracker.yaml".
    - persist (bool): Whether to persist video files after processing or not. Default is True.
    - augment (bool): Whether to apply data augmentation during inference or not. Default is False.
    - save (bool): Whether to save inference results or not. Default is False.
    - show (bool): Whether to display video playback or not. Default is False.
    - verbose (bool): Whether to enable verbose mode or not. Default is False.
    """
    # Implementation of the yolo_watch_camera function using the provided arguments

    # Convert all arguments to a dictionary using locals()
    query = locals()
    # query.pop('self', None)  # Remove the 'self' key if this function is within a class
    
    # Show params
    if verbose:
        print('YOLO CAMERA QUERY:', query)

    device = query["device"]
    if device == "gpu":
        device = 0

    source = query["source"]
    objects = query["objects"]
    classes = query["classes"]

    camera = None
    if query['camera_id'] is not None:
        # Get camera object from bigquery table
        camera_id = query['camera_id']
        camera = get_camera_from_bq_table(camera_id)
        
        # override parameters based on registered camera data
        source = camera["url"]
        if objects is not None:
            objects = [name.strip() for name in camera["objects"].split(",") if name != ""]

    if objects is not None and len(objects) == 0:
        objects = None
    if classes is not None and len(classes) == 0:
        classes = None
    
    # Detection/tracking model parameters
    model_params = {
        "objects": objects,
        "classes": classes,
        "imgsz": query["imgsz"],
        "conf": query["conf"],
        "iou": query["iou"],
        "max_det": query["max_det"],
        "vid_stride": query["vid_stride"],
        "device": device,
        "tracker": query["tracker"],
        # "persist": query["persist"],
        "augment": query["augment"],
        "save": query["save"],
        "show": query["show"],
        "verbose": query["verbose"],
    }
    
    if query['task'] == 'track' and query['capture'] == 'opencv':
        model_params['persist'] = True
        
    post_processing_args_dict = {
        'none': None,
        'console-log': {},
        'bigquery': {
            'camera': camera,
        },
        'trigger': {
            'camera': camera,
        },
        'bigquery-trigger': {
            'camera': camera,
        },
    }

    post_processing_function = post_processing_functions_dict[query['process']]
    post_processing_args = post_processing_args_dict[query['process']]
    annotator = annotators_dict[query['annotator']]

    yolo_params_dict = {
        "source": source,
        "model": query["model"],
        "task": query["task"],
        "model_params": model_params,
        "max_frames": query["max_frames"],
        "seconds": query["seconds"],
        "execution_seconds": query["execution_seconds"],
        "log_seconds": query["log_seconds"],
        "fps": query["fps"],
        "writer_params": None,
        "post_processing_function": post_processing_function,
        "post_processing_args": post_processing_args,
        "annotator": annotator,
        "generator": query["stream"],
        "capture": query["capture"],
        "retries": query["retries"],
        "retry_delay": query["retry_delay"],
    }

    if verbose:
        # print("INFERENCE REQUEST · QUERY ARGS:", query)
        print("YOLO REQUEST:", yolo_params_dict)
    
    results = yolo_watch(**yolo_params_dict) # generator function, e.i returns generator object

    return results # return generator object


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
