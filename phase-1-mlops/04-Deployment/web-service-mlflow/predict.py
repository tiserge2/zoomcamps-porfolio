import os
import pickle
import xgboost as xgb

import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify

os.environ['AWS_PROFILE'] = 'sergio'
TRACKING_URI = '13.61.191.33'
mlflow.set_tracking_uri(f'http://{TRACKING_URI}:5000')
mlflow.set_experiment('nyc-taxi-experiment-deployment')

RUN_ID = "269683ecb9a54ee5ac6ea3e01c4f7806"
LOGGED_MODEL = f'runs:/{RUN_ID}/model'

model = mlflow.xgboost.load_model(LOGGED_MODEL)

ARTIFICAT_URI = f'runs:/{RUN_ID}/preprocessor/preprocessor.b'
path = mlflow.artifacts.download_artifacts(artifact_uri=ARTIFICAT_URI)
print(f"Downloading preprocessor artifact to {path}")

with open(path, "rb") as f_in:
    dv = pickle.load(f_in)

def prepare_features(ride): 
    features = {}
    features["PULocationID"] = str(ride["PULocationID"])
    features["DOLocationID"] = str(ride["DOLocationID"])
    return features


def predict(features):
    X = dv.transform([features])
    print(f"Input shape: {X.shape}")
    X = xgb .DMatrix(X)
    y_pred = model.predict(X)
    return y_pred[0]


app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': str(pred),
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='9696')