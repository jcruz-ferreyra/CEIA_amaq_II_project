import datetime

from airflow.decorators import dag
from tasks import retrain_tasks

markdown_text = """
### Re-train model for Rain in Australia dataset.
"""

description_text = "Retrain model for rain data, evaluates if the new model performs better than existing one."

default_args = {
    "owner": "Juan Cruz Ferreyra",
    "depends_on_past": False,
    "schedule_interval": None,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=15),
}


@dag(
    dag_id="retrain_the_model",
    description=description_text,
    doc_md=markdown_text,
    tags=["Re-Train", "Rain", "Australia", "Dataset"],
    default_args=default_args,
    catchup=False,
)
def retrain_rain_australia():

    task_0 = retrain_tasks.train_the_challenger_model()
    retrain_tasks.evaluate_champion_challenge(task_0)


my_dag = retrain_rain_australia()
