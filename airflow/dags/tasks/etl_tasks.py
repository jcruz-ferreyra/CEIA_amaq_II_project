import datetime
import logging
import os

import numpy as np
import pandas as pd
from airflow.decorators import task

from . import etl_utils


# task.virtualenv makes us to import again the globally imported libraries.
# also it makes hard to access other python files like "utils.py".
@task.virtualenv(
    task_id="get_raw_data",
    requirements=["kagglehub~=0.2"],
    system_site_packages=True,
)
def get_raw_data():
    """
    Download raw data from Kaggle
    """
    import kagglehub
    import pandas as pd

    filepath = kagglehub.dataset_download(
        "jsphyg/weather-dataset-rattle-package/versions/2",
        "weatherAUS.csv",
        force_download=True,
    )

    df = pd.read_csv(filepath, compression="zip")
    df.to_csv(filepath, index=False)

    return filepath


@task(task_id="load_raw_data")
def load_raw_data(filepath):
    """
    Load the raw data to s3 bucket
    """
    import awswrangler as wr
    import pandas as pd

    df = pd.read_csv(filepath)

    filename = "rain_australia.csv"
    new_filepath = f"s3://data/raw/{filename}"

    wr.s3.to_csv(df=df, path=new_filepath, index=False)

    return new_filepath


@task.virtualenv(
    task_id="get_location_coords",
    requirements=["osmnx~=1.9"],
    system_site_packages=True,
)
def get_location_coords(filepath):
    """
    Load the raw data to s3 bucket
    """
    import json
    import logging
    import re

    import awswrangler as wr
    import boto3
    import botocore
    import osmnx as ox
    import pandas as pd

    def check_if_exists_s3(bucket_name, key):
        client = boto3.client("s3")
        try:
            client.head_object(Bucket=bucket_name, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                raise e

    logger = logging.getLogger("get_location_coords")

    bucket_name = "data"
    key = "raw/location_coords.csv"
    loc_filepath = f"s3://{bucket_name}/{key}"

    df = pd.read_csv(filepath)

    mapping_dict = {"Dartmoor": "DartmoorVillage", "Richmond": "RichmondSydney"}
    df["Location"] = df["Location"].map(mapping_dict).fillna(df["Location"])

    locations = df["Location"].unique()
    locations = [re.sub(r"([a-z])([A-Z])", r"\1 \2", l) for l in locations]

    df_existing_locations = pd.DataFrame(columns=["Location", "Lat", "Lon"])
    if check_if_exists_s3(bucket_name, key):
        logger.info("Coords for locations already existing in S3 bucket")
        df_existing_locations = wr.s3.read_csv(path=loc_filepath)
        existing_locations = df_existing_locations["Location"]

        locations = [l for l in locations if l not in existing_locations]

        if len(locations) == 0:
            logger.info(f"{len(locations)} locations to geolocate")
            return loc_filepath

    logger.info(f"{len(locations)} locations to geolocate")

    country = "Australia"

    locs = []
    lats = []
    lons = []
    for location in locations:
        try:
            lat, lon = ox.geocode(location + f", {country}")

            locs.append(location.replace(" ", ""))
            lats.append(lat)
            lons.append(lon)
        except Exception as e:
            logger.info(f"Error retrieving coordinates for {location}: {e}")

    df_locations = pd.DataFrame({"Location": locs, "Lat": lats, "Lon": lons})

    df_locations = pd.concat([df_existing_locations, df_locations])

    wr.s3.to_csv(df=df_locations, path=loc_filepath, index=False)

    # Save information of the dataset
    client = boto3.client("s3")

    info_dict = {}
    try:
        client.head_object(Bucket=bucket_name, Key="info/data_info.json")
        result = client.get_object(Bucket=bucket_name, Key="info/data_info.json")
        text = result["Body"].read().decode()
        info_dict = json.loads(text)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            # Something else has gone wrong.
            raise e

    info_dict["loc_filepath"] = loc_filepath

    info_string = json.dumps(info_dict, indent=2)

    client.put_object(Bucket=bucket_name, Key="info/data_info.json", Body=info_string)

    return loc_filepath


@task(task_id="process_data")
def process_data(filepath, loc_filepath):
    import json
    import os

    import awswrangler as wr
    import boto3
    import botocore
    import numpy as np
    import pandas as pd

    df = wr.s3.read_csv(filepath)

    # Remove data with missing values in target column
    target = ["RainTomorrow"]
    df = df.dropna(subset=target)

    # Encode binary columns to indicators
    binary_columns = ["RainToday"]
    df = etl_utils.encode_binary_columns(df, binary_columns + target)

    # Add location coords as features
    df_locations = wr.s3.read_csv(loc_filepath)
    df = pd.merge(df, df_locations, how="left", on="Location")

    # Encode date columns
    date_columns = ["Date"]
    df = etl_utils.encode_date_columns(df, date_columns)

    # Encode direction columns
    dir_columns = ["WindGustDir", "WindDir9am", "WindDir3pm"]
    df = etl_utils.encode_dir_columns(df, dir_columns)

    filename = os.path.basename(filepath)
    new_filepath = f"s3://data/process/{filename}"

    wr.s3.to_csv(df=df, path=new_filepath, index=False)

    # Save information of the dataset
    client = boto3.client("s3")

    bucket_name = "data"

    info_dict = {}
    try:
        client.head_object(Bucket=bucket_name, Key="info/data_info.json")
        result = client.get_object(Bucket=bucket_name, Key="info/data_info.json")
        text = result["Body"].read().decode()
        info_dict = json.loads(text)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            # Something else has gone wrong.
            raise e

    info_dict["binary_columns"] = binary_columns
    info_dict["date_columns"] = date_columns
    info_dict["dir_columns"] = dir_columns

    info_string = json.dumps(info_dict, indent=2)

    client.put_object(Bucket=bucket_name, Key="info/data_info.json", Body=info_string)
    #

    return new_filepath


@task(task_id="split_data")
def split_data(filepath):
    import os
    import random

    import awswrangler as wr
    import numpy as np
    import pandas as pd

    logger = logging.getLogger("get_location_coords")

    df = wr.s3.read_csv(filepath)
    df_train, df_test = etl_utils.split_by_date(df)

    logger.info(f"Train set observations: {df_train.shape[0]}")
    logger.info(f"Test set observations: {df_test.shape[0]}")

    filename_wo_ext = os.path.splitext(os.path.basename(filepath))[0]

    new_train_filepath = f"s3://data/process/{filename_wo_ext}_train.csv"
    wr.s3.to_csv(df=df_train, path=new_train_filepath, index=False)

    new_test_filepath = f"s3://data/process/{filename_wo_ext}_test.csv"
    wr.s3.to_csv(df=df_test, path=new_test_filepath, index=False)

    new_filepaths = {
        "train_filepath": new_train_filepath,
        "test_filepath": new_test_filepath,
    }

    return new_filepaths


@task(task_id="reduce_skeweness")
def reduce_skeweness(filepaths):
    import json

    import awswrangler as wr
    import boto3
    import botocore
    import numpy as np
    import pandas as pd

    train_filepath = filepaths["train_filepath"]
    test_filepath = filepaths["test_filepath"]

    df_train = wr.s3.read_csv(train_filepath)
    df_test = wr.s3.read_csv(test_filepath)

    right_skewed_columns = ["Rainfall", "Evaporation"]
    for col in right_skewed_columns:
        df_train[f"{col}Log"] = np.log(df_train[col] + 1)
    for col in right_skewed_columns:
        df_test[f"{col}Log"] = np.log(df_test[col] + 1)

    wr.s3.to_csv(df=df_train, path=train_filepath, index=False)
    wr.s3.to_csv(df=df_test, path=test_filepath, index=False)

    filepaths = {
        "train_filepath": train_filepath,
        "test_filepath": test_filepath,
    }

    # Save information of the dataset
    client = boto3.client("s3")

    bucket_name = "data"

    info_dict = {}
    try:
        client.head_object(Bucket=bucket_name, Key="info/data_info.json")
        result = client.get_object(Bucket=bucket_name, Key="info/data_info.json")
        text = result["Body"].read().decode()
        info_dict = json.loads(text)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            # Something else has gone wrong.
            raise e

    info_dict["right_skewed_columns"] = right_skewed_columns

    info_string = json.dumps(info_dict, indent=2)

    client.put_object(Bucket=bucket_name, Key="info/data_info.json", Body=info_string)
    #

    return filepaths


def _get_outlier_thresh(x, q1=0.25, q3=0.75):
    quartile1 = x.quantile(q1)
    quartile3 = x.quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return (low_limit, up_limit)


def _cap_outliers(x, limits):
    low_limit, up_limit = limits

    is_over_limits = x > up_limit
    x[is_over_limits] = up_limit

    is_under_limits = x < low_limit
    x[is_under_limits] = low_limit

    return x


@task(task_id="cap_outliers")
def cap_outliers(filepaths):
    import json

    import awswrangler as wr
    import boto3
    import botocore
    import numpy as np
    import pandas as pd

    train_filepath = filepaths["train_filepath"]
    test_filepath = filepaths["test_filepath"]

    df_train = wr.s3.read_csv(train_filepath)
    df_test = wr.s3.read_csv(test_filepath)

    numeric_columns = [
        "MinTemp",
        "MaxTemp",
        "RainfallLog",
        "EvaporationLog",
        "Sunshine",
        "WindGustSpeed",
        "WindSpeed9am",
        "WindSpeed3pm",
        "Humidity9am",
        "Humidity3pm",
        "Pressure9am",
        "Pressure3pm",
        "Cloud9am",
        "Cloud3pm",
        "Temp9am",
        "Temp3pm",
    ]

    numeric_columns_limits = {}
    for feature in numeric_columns:
        limits = _get_outlier_thresh(df_train[feature])
        df_train[feature] = _cap_outliers(df_train[feature], limits)

        df_test[feature] = _cap_outliers(df_test[feature], limits)

        numeric_columns_limits[feature] = limits

    wr.s3.to_csv(df=df_train, path=train_filepath, index=False)
    wr.s3.to_csv(df=df_test, path=test_filepath, index=False)

    # Save information of the dataset
    client = boto3.client("s3")

    bucket_name = "data"

    info_dict = {}
    try:
        client.head_object(Bucket=bucket_name, Key="info/data_info.json")
        result = client.get_object(Bucket=bucket_name, Key="info/data_info.json")
        text = result["Body"].read().decode()
        info_dict = json.loads(text)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            # Something else has gone wrong.
            raise e

    info_dict["numeric_columns_limits"] = numeric_columns_limits

    info_string = json.dumps(info_dict, indent=2)

    client.put_object(Bucket=bucket_name, Key="info/data_info.json", Body=info_string)
    #

    filepaths = {
        "train_filepath": train_filepath,
        "test_filepath": test_filepath,
    }

    return filepaths


@task(task_id="impute_missing")
def impute_missing(filepaths):
    import json

    import awswrangler as wr
    import boto3
    import botocore
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer

    train_filepath = filepaths["train_filepath"]
    test_filepath = filepaths["test_filepath"]

    df_train = wr.s3.read_csv(train_filepath)
    df_test = wr.s3.read_csv(test_filepath)

    features = [
        "Lat",
        "Lon",
        "DayCos",
        "DaySin",
        "MinTemp",
        "MaxTemp",
        "RainfallLog",
        "EvaporationLog",
        "Sunshine",
        "WindGustSpeed",
        "WindSpeed9am",
        "WindSpeed3pm",
        "WindGustDirCos",
        "WindGustDirSin",
        "WindDir9amCos",
        "WindDir9amSin",
        "WindDir3pmCos",
        "WindDir3pmSin",
        "Humidity9am",
        "Humidity3pm",
        "Pressure9am",
        "Pressure3pm",
        "Cloud9am",
        "Cloud3pm",
        "Temp9am",
        "Temp3pm",
        "RainToday",
    ]

    target = ["RainTomorrow"]

    # Impute numerical columns
    num_imputer = SimpleImputer(strategy="median")
    df_train[features] = num_imputer.fit_transform(df_train[features])
    df_test[features] = num_imputer.fit_transform(df_test[features])

    df_train = df_train[features + target]
    df_test = df_test[features + target]

    wr.s3.to_csv(df=df_train, path=train_filepath, index=False)
    wr.s3.to_csv(df=df_test, path=test_filepath, index=False)

    filepaths = {
        "train_filepath": train_filepath,
        "test_filepath": test_filepath,
    }

    # Save information of the dataset
    client = boto3.client("s3")

    bucket_name = "data"

    info_dict = {}
    try:
        client.head_object(Bucket=bucket_name, Key="info/data_info.json")
        result = client.get_object(Bucket=bucket_name, Key="info/data_info.json")
        text = result["Body"].read().decode()
        info_dict = json.loads(text)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            # Something else has gone wrong.
            raise e

    info_dict["features"] = features

    info_string = json.dumps(info_dict, indent=2)

    client.put_object(Bucket=bucket_name, Key="info/data_info.json", Body=info_string)
    #

    return filepaths


@task(task_id="normalize_data")
def normalize_data(filepaths):
    import json

    import awswrangler as wr
    import boto3
    import botocore
    import mlflow
    from sklearn.preprocessing import StandardScaler

    train_filepath = filepaths["train_filepath"]
    test_filepath = filepaths["test_filepath"]

    df_train = wr.s3.read_csv(train_filepath)
    df_test = wr.s3.read_csv(test_filepath)

    target = ["RainTomorrow"]

    X_train = df_train.drop(columns=target)
    y_train = df_train[target]

    X_test = df_test.drop(columns=target)
    y_test = df_test[target]

    # Scale Xs
    std_scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_arr = std_scaler.fit_transform(X_train)
    X_test_arr = std_scaler.transform(X_test)

    # Save scaled dataframes
    X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_arr, columns=X_test.columns)

    train_filename_wo_ext = os.path.splitext(os.path.basename(train_filepath))[0]
    test_filename_wo_ext = os.path.splitext(os.path.basename(test_filepath))[0]

    X_train_filepath = f"s3://data/final/{train_filename_wo_ext}_X.csv"
    y_train_filepath = f"s3://data/final/{train_filename_wo_ext}_y.csv"
    X_test_filepath = f"s3://data/final/{test_filename_wo_ext}_X.csv"
    y_test_filepath = f"s3://data/final/{test_filename_wo_ext}_y.csv"

    wr.s3.to_csv(df=X_train, path=X_train_filepath, index=False)
    wr.s3.to_csv(df=y_train, path=y_train_filepath, index=False)
    wr.s3.to_csv(df=X_test, path=X_test_filepath, index=False)
    wr.s3.to_csv(df=y_test, path=y_test_filepath, index=False)

    # Log train and test set to mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment = mlflow.set_experiment("Rain in Australia")

    mlflow.start_run(
        run_name="ETL_run_" + datetime.datetime.today().strftime("%Y/%m/%d-%H:%M:%S"),
        experiment_id=experiment.experiment_id,
        tags={"experiment": "etl", "dataset": "Rain in Australia"},
        log_system_metrics=True,
    )

    mlflow_df_train = mlflow.data.from_pandas(
        pd.concat([X_train, y_train], axis=1),
        source="https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package",
        targets=target[0],
        name="rain_australia_train",
    )
    mlflow.log_input(mlflow_df_train, context="Train Data")

    mlflow_df_test = mlflow.data.from_pandas(
        pd.concat([X_test, y_test], axis=1),
        source="https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package",
        targets=target[0],
        name="rain_australia_test",
    )
    mlflow.log_input(mlflow_df_test, context="Test Data")

    mlflow.log_param("Train observations", X_train.shape[0])
    mlflow.log_param("Test observations", X_test.shape[0])

    mlflow.end_run()

    # Save information of the dataset
    client = boto3.client("s3")

    bucket_name = "data"

    info_dict = {}
    try:
        client.head_object(Bucket=bucket_name, Key="info/data_info.json")
        result = client.get_object(Bucket=bucket_name, Key="info/data_info.json")
        text = result["Body"].read().decode()
        info_dict = json.loads(text)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            # Something else has gone wrong.
            raise e
  
    info_dict["std_scaler_scale"] = std_scaler.scale_.tolist()
    info_dict["std_scaler_mean"] = std_scaler.mean_.tolist()
    info_dict["std_scaler_var"] = std_scaler.var_.tolist()

    info_string = json.dumps(info_dict, indent=2)

    client.put_object(Bucket=bucket_name, Key="info/data_info.json", Body=info_string)
    #

    return
