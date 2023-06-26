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

def bigquery_post_new_objects(frame, inference, time_info, **kwargs):
    
    # get list of objects identified on the frame
    new_objects = inference[2]
    
    # initialize list for errors
    errors = []
    
    # if there's any new object
    if len(new_objects):
        
        # drop unwanted fields
        for obj in new_objects:
            del obj['track_id']
            del obj['bbox']
            for key, value in kwargs.items():
                obj[key] = value
        
        # insert records of new objects into BigQuery table
        errors = bqclient.insert_rows(table, new_objects)

        # log errors if any
        if errors:
            print('Error inserting record into BigQuery:', errors)

    # return list with errors
    return {'n_new_objects': len(new_objects), 'n_errors': len(errors), 'errors': errors}


# set up color scheme
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

def write_demo(frame, inference, time_info, resize_shape=None):

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

@app.after_request
def apply_caching(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

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

@app.get("/init")
def initialize():
    name = os.environ.get("NAME", "unamed")
    return f'Server `{name}` is running!'


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
    objects = DelimitedList(String(), sep=[',', ', '], allow_none=True, load_default=None)

@app.post('/camera')
@app.input(CameraIn)
@app.doc(tags=['Web Apps'])
def post_camera(data):
    url = data['url']
    objects = data['objects']
    # get the current time and date for Brazil time zone
    timestamp = datetime.datetime.now(brazil_tz)

    # Check if the camera exists in the BigQuery database
    query = f"SELECT * FROM `octacity.video_analytics.cameras` WHERE url='{url}'"
    query_job = bqclient.query(query)
    rows = query_job.result()

    for row in rows:
        return jsonify({'error': 'Camera URL already exists'}), 409

    # Create a new user in the BigQuery database
    query = f"INSERT INTO `octacity.video_analytics.cameras` (url, objects, timestamp) VALUES ('{url}', '{objects}', '{timestamp}')"
    query_job = bqclient.query(query)
    query_job.result()

    # Sign-up successful
    return jsonify({'message': 'Camera registration successful'})


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

@app.get("/tracker")
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
    seconds = Integer(load_default=10)
    confidence = Float(load_default=0.4)
    fps = Integer(load_default=3)
    detector = String(load_default='yolo')

@app.get("/track")
@app.input(TrackIn, 'query')
@app.doc(tags=['Streaming'])
def view_and_post_track(query):
    """
    Object Identification Live Stream
    Runs object detection, tracking and identification for live camera image streaming. Streams the annotated images.
    
    """
    allowed_objects = [class_name.strip() for class_name in query['objects']] if len(query['objects']) > 0 else None
    return Response(stream_with_context(tracking_reid(
        query['url'],
        model=query['detector'],
        confidence_threshold=query['confidence'],
        allowed_objects=allowed_objects,
        post_processing_function=bigquery_post_new_objects, # posts new identified objects to database
        post_processing_args={'url': query['url']},
        proccess_each=1,
        run_detection_each=1,
        frame_annotator=write_demo, # annotates frames using detection output
        to_url=None,
        generator=True, # yields annotated frames
        secs=float(query['seconds']),
        fps=3
    )), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.get("/track/view")
@app.input(TrackIn, 'query')
@app.doc(tags=['Streaming'])
def view_track(query):
    """
    Object Identification Live Stream
    Runs object detection, tracking and identification for live camera image streaming. Streams the annotated images.
    
    """
    allowed_objects = [class_name.strip() for class_name in query['objects']] if len(query['objects']) > 0 else None
    if query['detector'] == 'ultralytics':
        model = 'yolo'
        tracker = 'yolo'
    else:
        model = query['detector']
        tracker = 'deepsort'
    
    return Response(stream_with_context(tracking_reid(
        query['url'],
        model=model,
        tracker=tracker,
        confidence_threshold=query['confidence'],
        allowed_objects=allowed_objects,
        post_processing_function=None,
        post_processing_args={},
        proccess_each=1,
        run_detection_each=1,
        frame_annotator=write_demo, # annotates frames using detection output
        to_url=None,
        generator=True, # yields annotated frames
        secs=float(query['seconds']),
        fps=3
    )), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.get("/track/post")
@app.input(TrackIn, 'query')
@app.doc(tags=['Streaming'])
def post_track(query):
    """
    Post Object Identification
    Runs object detection, tracking and identification for live camera image streaming and posts identified objects to database in real time.
    
    """
    allowed_objects = [class_name.strip() for class_name in query['objects']] if len(query['objects']) > 0 else None
    post_new_objects_records = tracking_reid(
        query['url'],
        model=query['detector'],
        confidence_threshold=query['confidence'],
        allowed_objects=allowed_objects,
        max_frames=int(query['seconds'] * query['fps']),
        post_processing_function=bigquery_post_new_objects, # posts new identified objects to database
        post_processing_args={'url': query['url']},
        proccess_each=1,
        run_detection_each=1,
        frame_annotator=None, # annotates frames using detection output
        to_url=None,
        generator=False, # yields annotated frames if true
        secs=float(query['seconds']),
        fps=3
    )
    # print('post_new_objects_records: ', post_new_objects_records)
    return list(post_new_objects_records)


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
        proccess_each=2,
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
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))