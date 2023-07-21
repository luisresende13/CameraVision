# Python standard modules

import requests, json, cv2

# Custom modules

from modules.yolo_util import detected_objects, identified_objects, new_objects_from

# ---
# Google Cloud BigQuery set up

from google.cloud import bigquery

# set up the BigQuery client using the service account key file
credentials_path = 'auth/octacity-iduff.json'  # Replace with the path to your JSON with the path to your service account key file

# set up the dataset and objects_table ids
dataset_id = 'video_analytics'  # Replace with your dataset ID
objects_table_id = 'objetos_identificados'      # Replace with your objects_table ID

# BigQuery client
bqclient = bigquery.Client.from_service_account_json(credentials_path)

# get the BigQuery client and objects_table instances
objects_table_ref = bqclient.dataset(dataset_id).table(objects_table_id)
objects_table = bqclient.get_table(objects_table_ref)

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

    return {"timestamp": timestamp, "n_detected": len(detections), "n_tracked": len(unique_track_ids), "unique_track_ids": unique_track_ids, 'n_new': len(new_objects), 'new_objects': new_objects, 'kwargs': kwargs}


# INSERT RECORDS OF NEW OBJECTS INTO BIGQUERY DATABASE

def bigquery_post_new_objects(result, timestamp, post_processing_outputs, **kwargs):
    # Get unique tracking ids
    unique_track_ids = []
    if len(post_processing_outputs):
        previous_output = post_processing_outputs[-1]
        unique_track_ids = previous_output["unique_track_ids"].copy()

    # get list of objects identified on the frame
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
                "camera_id": camera["id"],
                "camera_name": camera["name"],
                "url": camera["url"],
            }
            rows.append(row)
        
        # insert records of new objects into BigQuery objects_table
        errors = bqclient.insert_rows(objects_table, rows)

        # log errors if any
        if errors:
            print('Error inserting records into BigQuery:', str(errors))
            # logging.error('Error inserting records into BigQuery:', errors)

    return {"timestamp": timestamp, "unique_track_ids": unique_track_ids, 'url': camera["url"], 'camera_id': camera['id'], 'new_objects': len(new_objects), "bigquery_errors": errors}

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
    tracking = identified_objects(result, timestamp)
    new_objects, unique_track_ids = new_objects_from(tracking, unique_track_ids)
    
    # drop unwanted fields
    responses = []

    # if there's any new object
    if len(new_objects):
        
        # get dictionary from json string
        post_scheme = json.loads(post_scheme)

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

    return {"timestamp": timestamp, "unique_track_ids": unique_track_ids, 'url': url, 'camera_id': camera_id, 'post_url': post_url, 'new_objects': len(new_objects), 'post_url_responses': responses}

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
