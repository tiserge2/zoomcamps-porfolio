#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from pprint import pprint

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG

# configure AWS profile and MLflow tracking URI
os.environ['AWS_PROFILE'] = 'sergio'
TRACKING_URI = '13.61.191.33'
mlflow.set_tracking_uri(f'http://{TRACKING_URI}:5000')
mlflow.set_experiment('nyc-taxi-experiment-orchestration')

# configure the DAG
default_args = {
    'owner': 'sergio',
    'depends_on_past': False,
    'start_date': pd.Timestamp.now(),
    'retries': 1,
}

# create DAG instance
dag = DAG(
    dag_id='nyc_taxi_orchestration',
    default_args=default_args,
    description='Orchestrate the NYC Taxi experiment',
    schedule=timedelta(minutes=1),
)

def read_data(month, year):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)
    print(f"Number of columns for {url}: {len(df.columns)}")
    # print(df.dtypes)
    #convert to datetime64[ns] to int64
    converted_start_end_df = df.astype({"lpep_pickup_datetime": "int", "lpep_dropoff_datetime": "int"})
    df['duration'] = (converted_start_end_df['lpep_dropoff_datetime'] - converted_start_end_df['lpep_pickup_datetime']) * 1.6666666666667e-8
    # print(df.duration.describe())
    print(f"min: {df.duration.min()}")
    print(f"max: {df.duration.max()}")
    df_b060 = df[(df.duration >= 1) & (df.duration <= 60)]
    # Q:What fraction of the records left after you dropped the outliers?
    fraction = len(df_b060) / len(df)
    print(f"fraction: {fraction*100:.2f}")
    return df_b060[["PULocationID", "DOLocationID", "duration"]]


def create_X(df, dv):
    categorical = ["PULocationID", "DOLocationID"]
    dicts = df[categorical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    mlflow.autolog()
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        validation = xgb.DMatrix(X_val, label=y_val)

        best_result = {'learning_rate': np.float64(0.20844413031306103), 
                    'max_depth': np.int64(99.0), 
                    'min_child_weight': np.float64(1.5432280770679585), 
                    'reg_alpha': np.float64(0.04953891913131428), 
                    'reg_lambda': np.float64(0.017060826809340544)}
        # mlflow.log_params(best_result)
        booster = xgb.train(
            params=best_result,
            dtrain=train,
            num_boost_round=2,
            evals=[(validation, "validation")],
            early_stopping_rounds=10
        )
        y_pred = booster.predict(validation)

        return run.info.run_id


def run(year, month):
    if month < 12:
        next_month = month + 1
        next_year = year
    else:
        next_month = 1
        next_year = year + 1

    train_df = read_data(month, year)
    val_df = read_data(next_month, next_year)

    X_train, dv = create_X(train_df, None)
    X_val, _ = create_X(val_df, dv)

    target = "duration"
    y_train = train_df[target].values
    y_val = val_df[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)

    return run_id


# define the tasks
read_data_task = PythonOperator(
    task_id='read_data',
    python_callable=read_data,
    dag=dag,
)

create_X_task = PythonOperator(
    task_id='create_X',
    python_callable=create_X,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

run_task = PythonOperator(
    task_id='run',
    python_callable=run,
    dag=dag,
)


# set up the task dependencies
read_data_task >> create_X_task >> train_model_task >> run_task