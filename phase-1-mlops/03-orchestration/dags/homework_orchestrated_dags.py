from datetime import timedelta
from turtle import pd

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG




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