from time import time; runtime = time()
import os
from tempfile import NamedTemporaryFile
import pandas as pd, numpy as np
import cv2
from urllib.parse import quote
import warnings
warnings.filterwarnings("ignore"); # Suppress warnings
import datetime
import pytz
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Usage example
# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
# logging.error('This is an error message')

# Deployment fixes:
# 1. lap module in requirements
# 2. camera.html detector as `ultralytics` and parameters
# 3. yolo_util device set to gpu


# Get the Brazil time zone
brazil_tz = pytz.timezone('America/Sao_Paulo')

# Flask

from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Flask API

from apiflask import APIFlask, Schema, abort
from apiflask.fields import Integer, String, Float, Boolean, Dict, List, DelimitedList, DateTime
from apiflask.validators import Length, OneOf

# Custom modules

from modules.object_identification import tracking_reid
from modules.video_processing import Video, get_video_metadata

# ---
# Base API url

baseurl = 'https://analytics.octacity.dev' # us-central1
# baseurl = 'https://video-analytics-oayt5ztuxq-ue.a.run.app'
# baseurl = 'http://127.0.0.1:5000' # development environment

# ---
# Google Cloud BigQuery set up

from google.cloud import bigquery

# set up the BigQuery client using the service account key file
credentials_path = 'auth/octacity-iduff.json'  # Replace with the path to your JSON with the path to your service account key file

# set up the dataset and table ids
dataset_id = 'video_analytics'  # Replace with your dataset ID
table_id = 'objetos_identificados'      # Replace with your table ID

# BigQuery client
bqclient = bigquery.Client.from_service_account_json(credentials_path)

# get the BigQuery client and table instances
table_ref = bqclient.dataset(dataset_id).table(table_id)
table = bqclient.get_table(table_ref)

# ---
# Google Cloud Storage

from google.cloud import storage
from google.oauth2 import service_account

# Set up the Google Cloud Storage client using JSON credentials
credentials_path = 'auth/octacity-iduff.json'  # Replace with the path to your JSON credentials file
credentials = service_account.Credentials.from_service_account_file(credentials_path)


# Util

def now(fmt="%Y-%m-%d %H:%M:%S"):
    return dt.now().strftime(fmt)


# INSERT RECORDS OF NEW OBJECTS INTO BIGQUERY DATABASE

def bigquery_post_new_objects(frame, inference, time_info, url, **kwargs):
    
    # get list of objects identified on the frame
    new_objects = inference[2]
    
    # initialize list for errors
    errors = []
    
    # if there's any new object
    if len(new_objects):
        
        # drop unwanted fields
        for obj in new_objects:
            '''
            obj keys:
                - class_name
                - confidence
                - timestamp
                - track_id
                - bbox
            '''
            
            # Delete object fields
            del obj['track_id']
            del obj['bbox']
            
            # Crate or update object fields
            obj['confidence'] = round(obj['confidence'], 2)
            obj['url'] = url # associate current caamera url
            # for key, value in kwargs.items():
                # obj[key] = value
        
        # insert records of new objects into BigQuery table
        errors = bqclient.insert_rows(table, new_objects)

        # log errors if any
        if errors:
            # print('Error inserting record into BigQuery:', str(errors))
            logging.error('Error inserting record into BigQuery:', errors)

    # return list with errors
    return {'n_new_objects': len(new_objects), 'n_errors': len(errors), 'errors': errors}

import requests, json

post_keys_to_english = {
    'objeto': 'class_name',
    'confianca': 'confidence',
    'hora': 'timestamp',
    'id_rastreio': 'track_id',
    'caixa': 'bbox',
    'url': 'url'
}

def trigger_post_url_new_objects(frame, inference, time_info, url, post_url, post_scheme, **kwargs):
    
    # get list of objects identified on the frame
    new_objects = inference[2]

    # if there's any new object
    if len(new_objects):
        
        # get dictionary from json string
        post_scheme = json.loads(post_scheme)

        # drop unwanted fields
        responses = []
        for obj in sorted(new_objects, key=lambda obj: obj['class_name']):
            '''
            obj keys:
                - class_name
                - confidence
                - timestamp
                - track_id
                - bbox
            '''
            # add `url` field to `obj` dict so its available to `trigger_post_body` dict
            obj['url'] = url
            
            # build post request body based on previous configuration
            trigger_post_body = {}
            del
            for key, value in post_scheme.items():
                trigger_post_body[key] = value if value not in post_keys_to_english else obj[post_keys_to_english[value]]
                if value == 'hora':
                    trigger_post_body[key] = trigger_post_body[key].strftime('%Y-%m-%d %H:%M:%S')
            
            # post request to `post_url`s
            res = requests.post(post_url, json=trigger_post_body)
            responses.append({'status_code': res.status_code, 'message': res.reason})

    return {'message': 'success', 'url': url, 'post_url': post_url, 'n_objects': len(new_objects), 'responses': responses}

def bigquery_post_and_trigger_new_objects(frame, inference, time_info, url, post_url, post_scheme, **kwargs):
    post_status = bigquery_post_new_objects(frame, inference, time_info, url, **kwargs)
    trigger_result = trigger_post_url_new_objects(frame, inference, time_info, url, post_url, post_scheme, **kwargs)
    result = {'message': 'success', 'url': url, 'post_url': post_url, 'n_objects': trigger_result['n_objects'], **post_status}
    print(f'POST-TRIGGER: {result}')
    return result

# set up color scheme
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

def ultralytics_plot(frame, inference, time_info, resize_shape, ultralytics_result):
    
    return ultralytics_result.plot()
    
    
def write_demo(frame, inference, time_info, resize_shape=None, **kwargs):

    tracking = inference[1]
    n_frames, process_start, process_end, start_time, end_time = time_info

    # loop over the formatted tracks and get newly identified objects
    for track in tracking:

        # get track attributes
        track_id, class_name, confidence, bbox = track

        # get pixel values from track bounding box
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # resize bounding boxes if image has been resized for preprocessing
        if resize_shape is not None:
            height, width = frame.shape[:2]
            new_width, new_height = resize_shape
            resize_factor_x = new_width / width
            resize_factor_y = new_height / height
            xmin = int(xmin / resize_factor_x)
            xmax = int(xmax / resize_factor_x)
            ymin = int(ymin / resize_factor_y)
            ymax = int(ymax / resize_factor_y)
            
        # draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 30), (xmin + 10 + 11 * (len(str(track_id)) + len(class_name)), ymin), GREEN, -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        cv2.putText(frame, class_name, (xmin + 12 + 10 * len(str(track_id)), ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    # calculate the average frames per second
    total_time = end_time - start_time
    avg_fps = n_frames / total_time

    # draw the average fps on the frame
    width = frame.shape[1]
    fps = f"FPS: {avg_fps:.2f}"
    cv2.putText(frame, fps, (width - 125, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLUE, 4)

    # return annotated frame
    return frame


# ---
# FLASK APP CONFIG

app = APIFlask(__name__); CORS(app)

# Flask ngrok
# from flask_ngrok import run_with_ngrok
# run_with_ngrok(app)

@app.after_request
def apply_caching(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

app.config['TIMEOUT'] = None  # Set timeout to None (no timeout)

app.config['SERVERS'] = [{'name': 'Google Cloud Run Server', 'url': f'{baseurl}/'}]

app.config['EXTERNAL_DOCS'] = {
    'description': 'Find more info here',
    'url': 'http://octacity.org'
}

app.config['INFO'] = {
    'title': 'Video Analytics',
    'version': '1.0',
    'description': open('README.MD').read(),
    'contact': {
        'name': 'OCTA CITY SOLUTIONS',
        'url': 'http://octacity.org',
        'email': 'luisresende@id.uff.br'
    },
}

app.config['TAGS'] = [{
    'name': 'Web Apps',
    'description': ''
},{
    'name': 'Big Query',
    'description': ''
},{
    'name': 'Streaming',
    'description': ''
},{
    'name': 'Upload',
    'description': ''
}]



# REQUEST AND RESPONSE SCHEMAS

version = '0.5'

@app.get("/init")
def initialize():
    name = os.environ.get("NAME", "unamed")
    return f'Server `{name}` version `v{version}` is running!'


# User login and signup

class LoginIn(Schema):
    email = String(required=True)
    password = String(required=True)
    
@app.post('/login')
@app.input(LoginIn)
@app.doc(tags=['Web Apps'])
def login(data):
    email = data['email']
    password = data['password']

    if not email or not password:
        return jsonify({'error': 'Invalid email or password'}), 400

    # Check if the user exists in the BigQuery database
    query = f"SELECT * FROM `octacity.video_analytics.users` WHERE email='{email}'"
    query_job = bqclient.query(query)
    rows = query_job.result()
    
    # Validate the password
    is_empty = True
    for row in rows:
        is_empty = False
        if row['password'] != password:
            return jsonify({'error': 'Invalid password'}), 401
        
    # Validate the username
    if is_empty:
        return jsonify({'error': 'User not found'}), 404

    # Login successful
    return jsonify({'message': 'Login successful'})

class CameraIn(Schema):
    url = String(required=True)
    objects = DelimitedList(String(), load_default=None, sep=[',', ', '])
    post_url = String(load_default='')
    post_scheme = String(load_default='')

@app.post('/camera')
@app.input(CameraIn)
@app.doc(tags=['Web Apps'])
def post_camera(data):
    url = data['url']
    objects = data['objects']
    post_url = data['post_url']
    post_scheme = data['post_scheme']
    timestamp = datetime.datetime.now(brazil_tz)  # get the current time and date for Brazil time zone

    # Check if the camera exists in the BigQuery database
    query = f"SELECT * FROM `octacity.video_analytics.cameras` WHERE url='{url}'"
    query_job = bqclient.query(query)
    rows = query_job.result()

    for row in rows:
        return jsonify({'error': 'A Câmera já existe.'}), 409

    # Create a new user in the BigQuery database
    objects = ', '.join(objects)
    query = f"INSERT INTO `octacity.video_analytics.cameras` (url, objects, post_url, post_scheme, timestamp) VALUES ('{url}', '{objects}', '{post_url}', '{post_scheme}', '{timestamp}')"
    query_job = bqclient.query(query)
    query_job.result()

    # Sign-up successful
    return jsonify({'message': 'Camera registration successful', 'data': data})

class DeleteCameraIn(Schema):
    url = String(required=True)

@app.delete('/camera')
@app.post('/camera/delete')
@app.input(DeleteCameraIn)
@app.doc(tags=['Web Apps'])
def delete_camera(data):
    url = data['url']

    # Check if the camera exists in the BigQuery database
    query = f"DELETE FROM `octacity.video_analytics.cameras` WHERE url='{url}'"
    try:
        # Submit the query
        job = bqclient.query(query)

        # Wait for the query to complete
        job.result()

        # Check if the query was successful
        if job.state == "DONE":
            return jsonify({'message': 'Record deleted successfully.'}), 200
        else:
            return jsonify({'error': 'Error occurred while deleting record.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

class EditCameraIn(Schema):
    url = String(required=True)
    objects = DelimitedList(String(), load_default=None, sep=[',', ', '], allow_none=True)
    post_url = String(load_default=None, allow_none=True)
    post_scheme = String(load_default=None, allow_none=True)
    
@app.put('/camera')
@app.post('/camera/put')
@app.input(EditCameraIn)
@app.doc(tags=['Web Apps'])
def edit_camera(data):
    url = data['url']
    objects = data['objects']
    post_url = data['post_url']
    post_scheme = data['post_scheme']

    # Check if the camera exists in the BigQuery database
    print('REQUEST URL:', url)
    query = f"SELECT * FROM `octacity.video_analytics.cameras` WHERE url='{url}'"
    query_job = bqclient.query(query)
    rows = query_job.result()

    # if rows.total_rows == 0:
        # return jsonify({'error': 'Camera URL does not exist'}), 404

    # Prepare the update query
    update_fields = []
    update_dict = {}
    if objects is not None:
        objects_string = ', '.join(objects)
        update_fields.append(f"objects='{objects_string}'")
        update_dict['objects'] = objects_string
    if post_url is not None:
        update_fields.append(f"post_url='{post_url}'")
        update_dict['post_url'] = post_url
    if post_scheme is not None:
        update_fields.append(f"post_scheme='{post_scheme}'")
        update_dict['post_scheme'] = post_scheme

    # Check if any fields were provided for update
    if not update_fields:
        return jsonify({'message': 'No fields provided for update'}), 200
    
    for row in rows:
        # Update the camera record in the BigQuery database if record exists
        update_query = f"UPDATE `octacity.video_analytics.cameras` SET {', '.join(update_fields)} WHERE url='{url}'"
        query_job = bqclient.query(update_query)
        query_job.result()
        
        # Update successful ---
        
        # Build updated record
        record = dict(row)
        for key, value in update_dict.items():
            record[key] = value
            
        # Success message
        return jsonify({'message': 'Camera record updated successfully', 'record': record}), 200
    
    # Return if the camera does not exist in the BigQuery database
    return jsonify({'error': 'Camera URL does not exist'}), 404
    

@app.get('/cameras')
@app.doc(tags=['Web Apps'])
def get_cameras():
    query = f"SELECT ROW_NUMBER() OVER () AS id, * FROM (SELECT * FROM `octacity.video_analytics.cameras` ORDER BY timestamp ASC) ORDER BY timestamp DESC"
    query_job = bqclient.query(query)
    rows = query_job.result()
    result = [dict(row) for row in rows]
    return jsonify(result)


class SignUpIn(Schema):
    email = String(required=True)
    password = String(required=True)

@app.post('/signup')
@app.input(SignUpIn)
@app.doc(tags=['Web Apps'])
def signup(data):
    email = data['email']
    password = data['password']

    if not email or not password:
        return jsonify({'error': 'Invalid email or password'}), 400

    # Check if the user exists in the BigQuery database
    query = f"SELECT * FROM `octacity.video_analytics.users` WHERE email='{email}'"
    query_job = bqclient.query(query)
    rows = query_job.result()

    for row in rows:
        return jsonify({'error': 'User already exists'}), 409

    # Create a new user in the BigQuery database
    query = f"INSERT INTO `octacity.video_analytics.users` (email, password) VALUES ('{email}', '{password}')"
    query_job = bqclient.query(query)
    query_job.result()

    # Sign-up successful
    return jsonify({'message': 'Sign up successful'})


# Define the forgot-password endpoint
@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    email = request.json.get('email')

    # Perform additional validation if needed

    # Query the BigQuery database to check if the email exists
    query = f"SELECT COUNT(*) as count FROM `octacity.video_analytics.users` WHERE email = '{email}'"
    query_job = bqclient.query(query)
    result = query_job.result()

    # Get the count value from the result
    count = next(result).get('count')

    if count == 0:
        return jsonify(message='Email does not exist'), 404

    # If the email exists, send a password reset link to the user's email
    send_password_reset_email(email)

    # Return a success message
    return jsonify(message='Password reset link sent to your email'), 200

from flask_mail import Mail, Message

app.config['MAIL_SERVER'] = 'your_mail_server'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_username'
app.config['MAIL_PASSWORD'] = 'your_password'

mail = Mail(app)

# Helper function to send the password reset email
def send_password_reset_email(email):
    # Replace this with your email sending code
    # Example: send an email with a password reset link to the provided email address
    # You can use libraries like Flask-Mail or third-party email services
    msg = Message('Password Reset', sender='luisresende@id.uff.br', recipients=[email])
    msg.body = "Please click on the following link to reset your password: <reset_link>"
    mail.send(msg)

# WEB APPS


@app.get("/camera_app")
@app.doc(tags=['Web Apps'])
def camera_control():
    """
    Object Identification WebApp 2.0
    Cloud based responsive web app that runs object detection, tracking and identification for live cameras image streaming. 
    """
    return render_template('camera.html')    

@app.get("/tracker")
@app.get("/playground")
@app.doc(tags=['Web Apps'])
def camera_tracker():
    """
    Object Identification WebApp 2.0
    Cloud based responsive web app that runs object detection, tracking and identification for live cameras image streaming. 
    """
    return render_template('tracker.html')    

@app.get("/tracker/v2")
@app.doc(tags=['Web Apps'])
def camera_tracker_v2():
    """
    Object Identification WebApp 2.0
    Cloud based responsive web app that runs object detection, tracking and identification for live cameras image streaming. 
    """
    return render_template('tracker-v2.html')    

@app.get("/tracker/v1")
@app.doc(tags=['Web Apps'])
def camera_tracker_v1():
    """
    Object Identification WebApp 1.0
    Cloud based responsive web app that runs object detection, tracking and identification for live cameras image streaming. 
    """
    return render_template('tracker-v1.html')    


@app.get("/upload_video")
@app.doc(tags=['Web Apps'])
def home():
    """
    Video File Web App
    Uploads a video file, runs object detection, tracking and identification and download processed video file. 
    """
    return render_template('video_upload_demo.html')    


# BigQuery Database

class ObjectsIn(Schema):
    url = String(load_default=None, allow_none=True)

@app.get("/objects")
@app.input(ObjectsIn, "query")
@app.doc(tags=['Big Query'])
def get_objects_from_bigquery(query):
    """
    Identified Objects
    Returns a JSON list of identified objects from the BigQuery table. Optionally filter by camera URL.
    """

    # Replace 'YOUR_PROJECT_ID' with your actual project ID
    project_id = 'octacity'

    # Replace 'YOUR_DATASET_ID' and 'YOUR_TABLE_ID' with your actual dataset and table IDs
    dataset_id = 'video_analytics'
    table_id = 'objetos_identificados'

    client = bigquery.Client(project=project_id, credentials=credentials)

    # Execute BigQuery query
    bq_query = f'SELECT ROW_NUMBER() OVER () AS id, * FROM `{project_id}.{dataset_id}.{table_id}`'
    if query['url'] is not None:
        bq_query += f' WHERE url = "{query["url"]}"'  # Add quotation marks around the URL value
    bq_query += ' ORDER BY timestamp DESC LIMIT 10'
    query_job = client.query(bq_query)
    rows = query_job.result()

    # Convert the BigQuery result rows to a list of dictionaries
    result = [dict(row) for row in rows]

    return jsonify(result)


# IMAGE TRANSMISSION

class TrackIn(Schema):
    url = String(required=True)
    objects = DelimitedList(String(), sep=[',', ', '], allow_none=True, load_default=None)
    confidence = Float(load_default=0.3)
    iou = Float(load_default=0.7)
    detector = String(load_default='yolo')
    tracker = String(load_default='botsort.yaml')
    seconds = Integer(load_default=None)
    exec_seconds = Integer(load_default=None)
    max_frames = Integer(allow_none=True, load_default=None)
    process_each = Integer(load_default=1)
    run_detection_each = Integer(load_default=1)
    fps = Integer(load_default=3)
    
    
@app.get("/track")
@app.input(TrackIn, 'query')
@app.doc(tags=['Streaming'])
def view_and_post_track(query):
    """
    Object Identification Live Stream
    Runs object detection, tracking and identification for live camera image streaming. Streams the annotated images.
    
    """
    
    allowed_objects = query['objects']
    if allowed_objects is not None:
        allowed_objects = None if len(allowed_objects) == 0 else [class_name.strip() for class_name in query['objects']]

    if query['detector'] == 'ultralytics':
        model = 'models/yolo/yolov8l.pt'
        tracker = 'yolo'
    else:
        model = query['detector']
        tracker = 'deepsort'
    
    return Response(stream_with_context(tracking_reid(
        url=query['url'],
        model=model,
        tracker=tracker,
        tracker_type=query['tracker'],
        confidence_threshold=query['confidence'],
        iou=query['iou'],
        allowed_objects=allowed_objects,
        secs=query['seconds'],
        max_frames=query['max_frames'],
        exec_secs=query['exec_seconds'],
        fps=query['fps'],
        process_each=query['process_each'],
        run_detection_each=query['run_detection_each'],
        post_processing_function=bigquery_post_new_objects, # posts new identified objects to database
        post_processing_args={'url': query['url']},
        frame_annotator=ultralytics_plot, # annotates frames using detection output
        to_url=None,
        generator=True, # yields annotated frames
        resize_shape=None,
    )), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.get("/track/view")
@app.input(TrackIn, 'query')
@app.doc(tags=['Streaming'])
def view_track(query):
    """
    Object Identification Live Stream
    Runs object detection, tracking and identification for live camera image streaming. Streams the annotated images.
    
    """
    
    allowed_objects = query['objects']
    if allowed_objects is not None:
        allowed_objects = None if len(allowed_objects) == 0 else [class_name.strip() for class_name in query['objects']]

    if query['detector'] == 'ultralytics':
        model = 'models/yolo/yolov8l.pt'
        tracker = 'yolo'
    else:
        model = query['detector']
        tracker = 'deepsort'
    
    return Response(stream_with_context(tracking_reid(
        query['url'],
        model=model,
        tracker=tracker,
        tracker_type=query['tracker'],
        confidence_threshold=query['confidence'],
        iou=query['iou'],
        allowed_objects=allowed_objects,
        post_processing_function=None,
        post_processing_args={},
        process_each=query['process_each'],
        run_detection_each=query['run_detection_each'],
        frame_annotator=ultralytics_plot, # write_demo, # annotates frames using detection output
        to_url=None,
        generator=True, # yields annotated frames
        secs=query['seconds'],
        exec_secs=query['exec_seconds'],
        max_frames=query['max_frames'],
        fps=query['fps'],
        resize_shape=None,
    )), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.get("/track/post")
@app.input(TrackIn, 'query')
@app.doc(tags=['Streaming'])
def post_track(query):
    """
    Post Object Identification
    Runs object detection, tracking and identification for live camera image streaming and posts identified objects to database in real time.
    
    """
    
    allowed_objects = query['objects']
    if allowed_objects is not None:
        allowed_objects = None if len(allowed_objects) == 0 else [class_name.strip() for class_name in query['objects']]
    
    if query['detector'] == 'ultralytics':
        model = 'models/yolo/yolov8l.pt'
        tracker = 'yolo'
    else:
        model = query['detector']
        tracker = 'deepsort'

    results = list(tracking_reid(  # returns generator object
        query['url'],
        model=model,
        tracker=tracker,
        tracker_type=query['tracker'],
        confidence_threshold=query['confidence'],
        iou=query['iou'],
        allowed_objects=allowed_objects,
        secs=query['seconds'],
        exec_secs=query['exec_seconds'],
        max_frames=query['max_frames'],
        post_processing_function=bigquery_post_new_objects, # posts new identified objects to database
        post_processing_args={'url': query['url']}, # pass camera url which is expected by the bigquery table
        process_each=query['process_each'],
        run_detection_each=query['run_detection_each'],
        frame_annotator=None, # annotates frames using detection output
        to_url=None,
        generator=False, # yields annotated frames if true
        fps=query['fps'],
        resize_shape=None,
    ))

     # requires `list` function to iterate over the generator object
    # results = list(post_new_objects_records)
    
    return {'MESSAGE': 'Track post successfull', 'RESULTS': results}

class TrackTriggerIn(Schema):
    url = String(required=True)
    post_url = String(required=True)
    post_scheme = String(required=True)
    objects = DelimitedList(String(), sep=[',', ', '], allow_none=True, load_default=None)
    confidence = Float(load_default=0.3)
    iou = Float(load_default=0.7)
    detector = String(load_default='yolo')
    tracker = String(load_default='botsort.yaml')
    seconds = Integer(load_default=None)
    exec_seconds = Integer(load_default=None)
    max_frames = Integer(allow_none=True, load_default=None)
    process_each = Integer(load_default=1)
    run_detection_each = Integer(load_default=1)
    fps = Integer(load_default=3)

@app.post("/track/trigger")
@app.input(TrackTriggerIn)
@app.doc(tags=['Streaming'])
def post_track_trigger(data):
    """
    Post Object Identification
    Runs object detection, tracking and identification for live camera image streaming and posts identified objects to database in real time.
    
    """
    
    try:

        allowed_objects = data['objects']
        if allowed_objects is not None:
            allowed_objects = None if len(allowed_objects) == 0 else [class_name.strip() for class_name in data['objects']]

        if data['detector'] == 'ultralytics':
            model = 'models/yolo/yolov8l.pt'
            tracker = 'yolo'
        else:
            model = data['detector']
            tracker = 'deepsort'

        results = list(tracking_reid(  # returns generator object
            data['url'],
            model=model,
            tracker=tracker,
            tracker_type=data['tracker'],
            confidence_threshold=data['confidence'],
            iou=data['iou'],
            allowed_objects=allowed_objects,
            secs=data['seconds'],
            exec_secs=data['exec_seconds'],
            max_frames=data['max_frames'],
            post_processing_function=bigquery_post_and_trigger_new_objects, # posts new identified objects to database
            post_processing_args={'url': data['url'], 'post_url': data['post_url'], 'post_scheme': data['post_scheme']},
            process_each=data['process_each'],
            run_detection_each=data['run_detection_each'],
            frame_annotator=None, # annotates frames using detection output
            to_url=None,
            generator=False, # yields annotated frames if true
            fps=data['fps'],
            resize_shape=None,
        ))

         # requires `list` function to iterate over the generator object
        # results = list(post_trigger_new_objects_records)

        return {'MESSAGE': 'Track post/trigger successfull', 'RESULTS': results}
    
    except Exception as e:
        print(f'ERROR IN TRACK-TRIGGER · ERROR: {e}')
        return {'MESSAGE': 'Track post/trigger failed', 'ERROR': str(e)}
        
# class TrackTriggerTestIn(Schema):
    # pass

@app.route("/track/trigger/test", methods=["POST"])
@app.route("/teste", methods=["POST"])
@app.route("/imprimi", methods=["POST"])
# @app.input(TrackTriggerTestIn)
@app.doc(tags=['Inference'])
def post_track_trigger_test():
    data = request.json
    print(f'NOVO OBJETO: {data}')
    return data

# VIDEO UPLOAD AND PROCESSING

@app.post('/upload')
@app.doc(tags=['Upload'])
def upload_video():
    """
    Runs object detection, tracking and identification for uploaded video file and returns processed video.
    """
    if 'video' not in request.files:
        return 'No video file found', 400

    uploaded_video = request.files['video']
    if uploaded_video.filename == '':
        return 'No selected file', 400

    # WRITE ANANOTATED VIDEO
    
    uploaded_temp_file = NamedTemporaryFile(suffix='.mp4', delete=False)
    annotated_temp_file = NamedTemporaryFile(suffix='.mp4', delete=False)
    uploaded_temp_file_name = uploaded_temp_file.name
    annotated_temp_file_name = annotated_temp_file.name
    
    # Create a named temporary file and save uploaded video
    # uploaded_temp_file = tempfile.NamedTemporaryFile(delete=False)
    uploaded_video.save(uploaded_temp_file_name)

    # Create a named temporary file for annotated video
    # annotated_temp_file = tempfile.NamedTemporaryFile(delete=False)

    # run tracking and identification
    post_processing_output = tracking_reid(
        uploaded_temp_file_name,
        model='yolo',
        confidence_threshold=0.4,
        allowed_objects=None,
        max_frames=None,
        post_processing_function=None,
        process_each=2,
        run_detection_each=2,
        frame_annotator=write_demo,
        to_video_path=annotated_temp_file_name,
    )

    # POST ANNOTATE VIDEO TO FILE SYSTEM CLOUD STORAGE
    # Save the video file to Google Cloud Storage
    storage_client = storage.Client(credentials=credentials)
    bucket_name = 'video-analytics-octacity'  # Replace with your bucket name
    bucket = storage_client.get_bucket(bucket_name)

    filename = secure_filename(uploaded_video.filename)
    blob_name = f'annotated/{filename}'
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(annotated_temp_file_name)

    # # Close the temporary file
    uploaded_temp_file.close()
    annotated_temp_file.close()

    return blob.public_url


if __name__ == "__main__":
    app.run()
    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))