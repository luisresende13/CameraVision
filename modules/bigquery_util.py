# Flask methods
from apiflask import HTTPError

# BIGQUERY SET UP ----------------

from google.cloud import bigquery

# set up the BigQuery client using the service account key file
credentials_path = 'auth/octacity-iduff.json'  # Replace with the path to your JSON with the path to your service account key file

# BigQuery client
bqclient = bigquery.Client.from_service_account_json(credentials_path)

# set up the dataset and table ids
dataset_id = 'video_analytics'  # Replace with your dataset ID
objects_table_id = 'objetos_identificados'      # Replace with your table ID

# get the BigQuery client and objects_table instances
objects_table_ref = bqclient.dataset(dataset_id).table(objects_table_id)
objects_table = bqclient.get_table(objects_table_ref)

def get_camera_from_bq_table(camera_id):
    try:
        # Fetch the camera from the BigQuery database based on the provided camera_id
        query = f"SELECT * FROM (SELECT *, ROW_NUMBER() OVER(ORDER BY timestamp) as id FROM `octacity.video_analytics.cameras`) WHERE id = {camera_id}"
        query_job = bqclient.query(query)
        rows = query_job.result()

        camera = None
        for row in rows:
            camera = dict(row)
            
        if camera:
            return camera
        else:
            raise HTTPError(404, f'Camera with ID {camera_id} not found')

    except Exception as e:
        msg = str(e)
        raise HTTPError(500, f"FAILED TO GET CAMERA FROM BIGQUERY TABLE. ERROR: {e}")
