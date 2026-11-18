import json
import base64
import boto3
import os
import mlflow


TEST_RUN = os.getenv("TEST_RUN", "False") == "True"
if TEST_RUN:
    print("Running in test mode. Predictions will not be sent to Kinesis stream.")
    os.environ['AWS_PROFILE'] = 'sergio'
else:
    print("Running in production mode. Predictions will be sent to Kinesis stream.")

PREDICTIONS_STREAM_NAME = os.getenv("PREDICTIONS_STREAM_NAME", None)
if PREDICTIONS_STREAM_NAME is None:
    raise ValueError("PREDICTIONS_STREAM_NAME environment variable is not set. Please set it to the name of the Kinesis stream where predictions should be sent.")
else:
    print(f"Predictions will be sent to Kinesis stream: {PREDICTIONS_STREAM_NAME}")

RUN_ID = os.getenv("RUN_ID", None)
if RUN_ID is None:
    raise ValueError("RUN_ID environment variable is not set. " \
    "Please set it to the MLflow run ID of the model you want to use.")
else:
    print(f"Using MLflow model from run ID: {RUN_ID}")

LOGGED_MODEL = f's3://mlflow-artifacts-remote-786875389232/38/models/m-{RUN_ID}/artifacts/'
print(f"Loading model from: {LOGGED_MODEL}")
model = mlflow.pyfunc.load_model(LOGGED_MODEL)
kinesis_client = boto3.client('kinesis', region_name='eu-north-1')


def prepare_features(ride): 
    features = {}
    features["PULocationID"] = str(ride["PULocationID"])
    features["DOLocationID"] = str(ride["DOLocationID"])
    return features


def predict(features):
    return model.predict([features])[0]


def lambda_handler(event, context):
    predictions = []

    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        ride_event = json.loads(decoded_data)
        
        ride = ride_event['ride']
        ride_id = ride_event['ride_id']

        features  = prepare_features(ride)
        prediction = predict(features)

        prediction_event = {
            'model': 'ride_prediction_model',
            'version': '123',
            'prediction': {
                'ride_duration': str(prediction),
                'ride_id': ride_id
            }
        }

        if not TEST_RUN:
            kinesis_client.put_record(
                StreamName=PREDICTIONS_STREAM_NAME,
                Data=json.dumps(prediction_event),
                PartitionKey=str(ride_id)
            )

        predictions.append(prediction_event)

    print(f"Predictions sent to Kinesis stream '{PREDICTIONS_STREAM_NAME}': {predictions}")
    
    return {
        "predictions": predictions
    }
