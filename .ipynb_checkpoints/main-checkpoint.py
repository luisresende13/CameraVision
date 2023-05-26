from time import time; runtime = time()
import os
from tempfile import NamedTemporaryFile
import pandas as pd, numpy as np
import cv2
from urllib.parse import quote

# Flask

from flask import Flask, render_template, request, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Flask API

from apiflask import APIFlask, Schema, abort
from apiflask.fields import Integer, String, Float, Boolean, Dict, List, DelimitedList, DateTime
from apiflask.validators import Length, OneOf

# Custom

from modules.object_identification import tracking_reid
from modules.video_processing import Video, get_video_metadata

# ---
# Base API url

# baseurl = 'https://video-analytics.octacity.dev' # us-central1
baseurl = 'https://video-analytics-oayt5ztuxq-ue.a.run.app'
# baseurl = 'http://127.0.0.1:5000' # development environment

# ---
# Google Cloud BigQuery set up

from google.cloud import bigquery

# set up the BigQuery client using the service account key file
credentials_path = 'auth/octacity-iduff.json'  # Replace with the path to your JSON with the path to your service account key file

# set up the dataset and table ids
dataset_id = 'video_analytics'  # Replace with your dataset ID
table_id = 'objetos_identificados'      # Replace with your table ID

# get the BigQuery client and table instances
client = bigquery.Client.from_service_account_json(credentials_path)
table_ref = client.dataset(dataset_id).table(table_id)
table = client.get_table(table_ref)

# ---
# Google Cloud Storage

from google.cloud import storage
from google.oauth2 import service_account

# Set up the Google Cloud Storage client using JSON credentials
credentials_path = 'auth/octacity-iduff.json'  # Replace with the path to your JSON credentials file
credentials = service_account.Credentials.from_service_account_file(credentials_path)
storage_client = storage.Client(credentials=credentials)

    
# Util

def now(fmt="%Y-%m-%d %H:%M:%S"):
    return dt.now().strftime(fmt)
    
# set up color scheme
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

def write_demo(frame, detections, tracking, new_objects, process_start, process_end):

    # loop over the formatted tracks and get newly identified objects
    for track in tracking:

        # get track attributes
        track_id, class_label, class_name, confidence, bbox, timestamp = track

        # get pixel values from track bounding box
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 30), (xmin + 10 + 11 * (len(str(track_id)) + len(class_name)), ymin), GREEN, -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        cv2.putText(frame, class_name, (xmin + 12 + 10 * len(str(track_id)), ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (process_end - process_start).total_seconds():.2f}"
    w = frame.shape[1]
    cv2.putText(frame, fps, (w - 125, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLUE, 4)

    # return annotated frame
    return frame

# INSERT RECORDS OF NEW OBJECTS INTO BIGQUERY DATABASE

def bigquery_post_new_objects(frame, detections, tracking, new_objects, process_start, process_end, **kwargs):
    
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
        errors = client.insert_rows(table, new_objects)

        # log errors if any
        if errors:
            print('Error inserting record into BigQuery:', errors)

    # return list with errors
    return {'n_new_objects': len(new_objects), 'n_errors': len(errors), 'errors': errors}


# ---
# FLASK APP CONFIG

app = APIFlask(__name__, docs_ui='elements'); CORS(app)

@app.after_request
def apply_caching(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

app.config['SERVERS'] = [{'name': 'Google Cloud Run Server', 'url': f'{baseurl}/'}]

app.config['EXTERNAL_DOCS'] = {
    'description': 'Find more info here',
    'url': 'http://octacity.org'
}

app.config['INFO'] = {
    'title': 'Video Analytics',
    'version': '1.0',
    'description': 'API by Octa City Solutions to provide video analytics services.',
    'contact': {
        'name': 'OCTA CITY SOLUTIONS',
        'url': 'http://octacity.org',
        'email': 'luisresende@id.uff.br'
    },
}

app.config['TAGS'] = [{
#     'name': 'Cameras',
#     'description': 'Collection of city cameras in Rio de Janeiro'
}]


# REQUEST AND RESPONSE SCHEMAS

class inputIn(Schema):
    test = String()

# API ENDPOINTS

@app.get("/")
def hello_world():
    name = os.environ.get("NAME", "World")
    return "Hello {}!".format(name)

@app.get("/upload_video")
def home():
    return render_template('video_upload_demo.html')    

@app.get("/tracker")
def camera_detector():
    return render_template('tracker.html')    

@app.route('/upload', methods=['POST'])
def upload_video():

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
        confidence_threshold=0.4,
        objects_allowed=None,
        max_frames=None,
        post_processing_function=None,
        proccess_each=None,
        frame_annotator=write_demo,
        to_video_path=annotated_temp_file_name,
    )

    # POST ANNOTATE VIDEO TO FILE SYSTEM CLOUD STORAGE
    # Save the video file to Google Cloud Storage
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


# IMAGE TRANSMISSION

class TrackIn(Schema):
    url = String(required=True)
    objects = DelimitedList(String(), sep=[',', ', '], allow_none=True, load_default=None)
    seconds = Integer(load_default=10)
    confidence = Float(load_default=0.4)
    fps = Integer(load_default=3)
    
@app.get("/track")
@app.input(TrackIn, 'query')
def view_track_and_post(query):

    objects_allowed = [class_name.strip() for class_name in query['objects']] if len(query['objects']) > 0 else None
    print('objects_allowed:', objects_allowed)
    print('confidence:', query['confidence'])
    return Response(stream_with_context(tracking_reid(
        query['url'],
        confidence_threshold=query['confidence'],
        objects_allowed=objects_allowed,
        max_frames=int(query['seconds'] * query['fps']),
        post_processing_function=bigquery_post_new_objects,
        post_processing_args={'url': query['url']},
        proccess_each=None,
        frame_annotator=write_demo,
        to_video_path=None,
        generator=True,
    )), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))