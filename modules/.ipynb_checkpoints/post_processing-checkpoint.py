# Standard Python modules

import cv2, requests, json, traceback
from apiflask import HTTPError


# Custom Python modules

from modules.bigquery_util import bqclient, objects_table
from modules.yolo_util import detected_objects, identified_objects, new_objects_from


# DEFAULT POST PROCESSING FUNCTION

def default_post_processing(result, timestamp, post_processing_outputs, **kwargs):
    # Get unique tracking ids
    unique_track_ids = []
    if len(post_processing_outputs):
        previous_output = post_processing_outputs[-1]
        unique_track_ids = previous_output["unique_track_ids"].copy()

    # get list of objects identified on the frame
    detections = detected_objects(result, timestamp)
    tracking = identified_objects(result, timestamp)
    new_objects, unique_track_ids = new_objects_from(tracking, unique_track_ids)

    return {'unique_track_ids': unique_track_ids, "timestamp": timestamp, "n_detected": len(detections), "n_tracked": len(tracking), 'n_tracked_new': len(new_objects), "n_tracked_total": len(unique_track_ids), **kwargs}


# INSERT RECORDS OF NEW OBJECTS INTO BIGQUERY DATABASE

def bigquery_post_new_objects(result, timestamp, post_processing_outputs, **kwargs):
    # get camera object
    camera = kwargs["camera"]

    # get `camera` attributes
    url = camera["url"]
    camera_id = camera["id"]
    camera_name = camera["name"]

    # Get unique tracking ids
    unique_track_ids = []
    if len(post_processing_outputs):
        previous_output = post_processing_outputs[-1]
        unique_track_ids = previous_output["unique_track_ids"].copy()

    # get list of objects identified on the frame
    detections = detected_objects(result, timestamp)
    tracking = identified_objects(result, timestamp)
    new_objects, unique_track_ids = new_objects_from(tracking, unique_track_ids)
    
    # initialize list for errors
    errors = []
    
    # initialize rows to insert to the objects_table
    rows = []

    # if there's any new object
    if len(new_objects):
        # get camera object
        camera = kwargs["camera"]
        
        # drop unwanted fields
        for obj in new_objects:
            '''obj keys:
                - track_id
                - timestamp
                - class_id
                - class_name
                - confidence
                - bbox
            '''
            row = {
                "timestamp": obj["timestamp"],
                "class_name": obj["class_name"],
                "confidence": round(obj['confidence'], 2),
                "camera_id": camera_id,
                "camera_name": camera_name,
                "url": url,
            }
            rows.append(row)
        
        # insert records of new objects into BigQuery objects_table
        errors = bqclient.insert_rows(objects_table, rows)

        # log errors if any
        if errors:
            print('Error inserting records into BigQuery:', str(errors))
            # logging.error('Error inserting records into BigQuery:', errors)

    return {'unique_track_ids': unique_track_ids, "timestamp": timestamp, "n_detected": len(detections), "n_tracked": len(tracking), 'n_tracked_new': len(new_objects), "n_tracked_total": len(unique_track_ids), "bigquery_errors": errors, **kwargs}

post_keys_to_english = {
    'objeto': 'class_name',
    'confianca': 'confidence',
    'hora': 'timestamp',
    'id_rastreio': 'track_id',
    'caixa': 'bbox',
    'url': 'url',
    'id_camera': 'camera_id',
    'nome_camera': 'camera_name',
}

def trigger_post_url_new_objects(result, timestamp, post_processing_outputs, **kwargs):
    # get camera object
    camera = kwargs["camera"]
    
    # get `camera` attributes
    url = camera["url"]
    camera_id = camera["id"]
    camera_name = camera["name"]
    post_url = camera["post_url"]
    post_scheme = camera["post_scheme"]
    
    # Get unique tracking ids
    unique_track_ids = []
    if len(post_processing_outputs):
        previous_output = post_processing_outputs[-1]
        unique_track_ids = previous_output["unique_track_ids"].copy()

    # get list of objects identified on the frame
    detections = detected_objects(result, timestamp)
    tracking = identified_objects(result, timestamp)
    new_objects, unique_track_ids = new_objects_from(tracking, unique_track_ids)
    
    # drop unwanted fields
    responses = []

    # if there's any new object
    if len(new_objects):
        
        # get dictionary from json string
        try:
            post_scheme = json.loads(post_scheme)
        except Exception as e:
            # Get the traceback as a string
            traceback_str = traceback.format_exc()        
            raise HTTPError(500, "Internal Server Error in POST PROCESSING (TRIGGER-POST-URL)", traceback_str)
            
            
        for obj in sorted(new_objects, key=lambda obj: obj['class_name']):
            '''
            obj keys:
                - class_name
                - confidence
                - timestamp
                - track_id
                - bbox
            '''
            # add `camera` fields to `obj` dict so its available to `trigger_post_body` dict
            obj['url'] = url
            obj['camera_id'] = camera_id
            obj['camera_name'] = camera_name
            
            # build post request body based on previous configuration
            trigger_post_body = {}
            for key, value in post_scheme.items():
                trigger_post_body[key] = value if value not in post_keys_to_english else obj[post_keys_to_english[value]]
                if value == 'hora':
                    trigger_post_body[key] = trigger_post_body[key].strftime('%Y-%m-%d %H:%M:%S')
            
            # post request to `post_url`s
            res = requests.post(post_url, json=trigger_post_body)
            responses.append({'status_code': res.status_code, 'message': res.reason})

    return {'unique_track_ids': unique_track_ids, "timestamp": timestamp, "n_detected": len(detections), "n_tracked": len(tracking), 'n_tracked_new': len(new_objects), "n_tracked_total": len(unique_track_ids), 'post_url_responses': responses, **kwargs}

def bigquery_post_and_trigger_new_objects(result, timestamp, post_processing_outputs, **kwargs):
    post_status = bigquery_post_new_objects(result, timestamp, post_processing_outputs, **kwargs)
    trigger_result = trigger_post_url_new_objects(result, timestamp, post_processing_outputs, **kwargs)
    result = {**post_status, **trigger_result}
    return result


# ---
# FPS Annotator

# set up color scheme
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

def fps_annotator(result, timestamp, post_processing_outputs, **kwargs):

    # default ultralytics result plot
    annotate_image = result.plot()

    n_frames = len(post_processing_outputs)

    if n_frames >= 2:
    # calculate the average frames per second
        initial_timestamp = post_processing_outputs[0]["timestamp"]
        total_time = (timestamp - initial_timestamp).microseconds * 1000
        avg_fps = (n_frames - 1) / total_time
    
        # draw the average fps on the frame
        fps = f"FPS: {avg_fps:.2f}"
        width = annotate_image.shape[1]
        cv2.putText(annotate_image, fps, (width - 125, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLUE, 4)

    # return annotated frame
    return annotate_image
