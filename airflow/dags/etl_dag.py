import datetime
import logging

from airflow.decorators import dag
from tasks import etl_tasks

markdown_text = """
### ETL Process for Rain in Australia dataset.
"""

description_text = (
    "ETL process for rain data, separating the dataset into training and testing sets."
)


default_args = {
    "owner": "Juan Cruz Ferreyra",
    "depends_on_past": False,
    "schedule_interval": None,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=15),
}


@dag(
    dag_id="process_etl_rain_data",
    description=description_text,
    doc_md=markdown_text,
    tags=["ETL", "Rain", "Australia", "Dataset"],
    default_args=default_args,
    catchup=False,
)
def etl_rain_australia():

    task_0 = etl_tasks.get_raw_data()

    task_1_1 = etl_tasks.load_raw_data(task_0)
    task_1_2 = etl_tasks.get_location_coords(task_0)

    task_2 = etl_tasks.process_data(task_1_1, task_1_2)
    task_3 = etl_tasks.split_data(task_2)
    task_4 = etl_tasks.reduce_skeweness(task_3)
    task_5 = etl_tasks.cap_outliers(task_4)
    task_6 = etl_tasks.impute_missing(task_5)
    etl_tasks.normalize_data(task_6)


dag = etl_rain_australia()
