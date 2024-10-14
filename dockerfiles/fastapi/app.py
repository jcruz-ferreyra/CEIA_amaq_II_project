import json
import pickle
from datetime import date
from typing import Literal

import awswrangler as wr
import boto3
import mlflow
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Body, FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler
from typing_extensions import Annotated


def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open("/app/files/model.pkl", "rb")
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client("s3")

        bucket_name = "data"

        s3.head_object(Bucket=bucket_name, Key="info/data_info.json")
        result_s3 = s3.get_object(Bucket=bucket_name, Key="info/data_info.json")
        text_s3 = result_s3["Body"].read().decode()
        info_dict = json.loads(text_s3)
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open("/app/files/data.json", "r")
        info_dict = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, info_dict


def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global info_dict
    global version_model

    try:
        model_name = "rain_australia_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, info_dict = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass


class ModelInput(BaseModel):
    """
    Define data structure for predictions in Rain Australia dataset.
    """

    Date: date = Field(
        description="Fecha de los datos.",
    )
    Location: Literal[
        "Adelaide",
        "Albany",
        "Albury",
        "AliceSprings",
        "BadgerysCreek",
        "Ballarat",
        "Bendigo",
        "Brisbane",
        "Cairns",
        "Canberra",
        "Cobar",
        "CoffsHarbour",
        "Dartmoor",
        "Darwin",
        "GoldCoast",
        "Hobart",
        "Katherine",
        "Launceston",
        "Melbourne",
        "MelbourneAirport",
        "Mildura",
        "Moree",
        "MountGambier",
        "MountGinini",
        "Newcastle",
        "Nhil",
        "NorahHead",
        "NorfolkIsland",
        "Nuriootpa",
        "PearceRAAF",
        "Penrith",
        "Perth",
        "PerthAirport",
        "Portland",
        "Richmond",
        "Sale",
        "SalmonGums",
        "Sydney",
        "SydneyAirport",
        "Townsville",
        "Tuggeranong",
        "Uluru",
        "WaggaWagga",
        "Walpole",
        "Watsonia",
        "Williamtown",
        "Witchcliffe",
        "Wollongong",
        "Woomera",
    ] = Field(
        description="Ubicación de la estación meteorológica.",
    )
    MinTemp: float = Field(
        description="Temperatura mínima de hoy.",
        ge=-10,
        le=50,
    )
    MaxTemp: float = Field(
        description="Temperatura máxima de hoy.",
        ge=-20,
        le=70,
    )
    Rainfall: float = Field(
        description="Cantidad de lluvia caída hoy.",
    )
    Evaporation: float = Field(
        description="Evaporación hoy.",
    )
    Sunshine: float = Field(
        description="Horas de sol hoy.",
        ge=0,
        le=24,
    )
    WindGustDir: Literal[
        "E",
        "ENE",
        "ESE",
        "N",
        "NE",
        "NNE",
        "NNW",
        "NW",
        "S",
        "SE",
        "SSE",
        "SSW",
        "SW",
        "W",
        "WNW",
        "WSW",
    ] = Field(
        description="Dirección de las ráfagas.",
    )
    WindGustSpeed: float = Field(
        description="Velocidad máxima de las ráfagas hoy.",
    )
    WindDir9am: Literal[
        "E",
        "ENE",
        "ESE",
        "N",
        "NE",
        "NNE",
        "NNW",
        "NW",
        "S",
        "SE",
        "SSE",
        "SSW",
        "SW",
        "W",
        "WNW",
        "WSW",
    ] = Field(
        description="Dirección del viento a las 9am.",
    )
    WindDir3pm: Literal[
        "E",
        "ENE",
        "ESE",
        "N",
        "NE",
        "NNE",
        "NNW",
        "NW",
        "S",
        "SE",
        "SSE",
        "SSW",
        "SW",
        "W",
        "WNW",
        "WSW",
    ] = Field(
        description="Dirección del viento a las 3pm.",
    )
    WindSpeed9am: float = Field(
        description="Velocidad del viento a las 9am.",
    )
    WindSpeed3pm: float = Field(
        description="Velocidad del viento a las 3pm.",
    )
    Humidity9am: float = Field(
        description="Humedad a las 9am.",
    )
    Humidity3pm: float = Field(
        description="Humedad a las 3pm.",
    )
    Pressure9am: float = Field(
        description="Presión a las 9am.",
    )
    Pressure3pm: float = Field(
        description="Presión a las 3pm.",
    )
    Cloud9am: float = Field(
        description="Cobertura de nubes a las 9am.",
    )
    Cloud3pm: float = Field(
        description="Cobertura de nubes a las 3pm.",
    )
    Temp9am: float = Field(
        description="Temperatura a las 9am.",
    )
    Temp3pm: float = Field(
        description="Temperatura a las 3pm.",
    )
    RainToday: float = Field(
        description="Llovio hoy? 1: si llovió, 0: no llovió.",
        ge=0,
    )

    model_config = {
        "json_schema_extra": {
            "ejemplos": [
                {
                    "Date": "2021-01-01",
                    "Location": "Sydney",
                    "MinTemp": 15.0,
                    "MaxTemp": 25.0,
                    "Rainfall": 0.0,
                    "Evaporation": 5.0,
                    "Sunshine": 10.0,
                    "WindGustdir": "N",
                    "WindGustSpeed": 30.0,
                    "WindDir9am": "N",
                    "WindDir3pm": "N",
                    "WindSpeed9am": 10.0,
                    "WindSpeed3pm": 15.0,
                    "Humidity9am": 50.0,
                    "Humidity3pm": 60.0,
                    "Pressure9am": 1010.0,
                    "Pressure3pm": 1005.0,
                    "Cloud9am": 5.0,
                    "Cloud3pm": 5.0,
                    "Temp9am": 20.0,
                    "Temp3pm": 23.0,
                    "RainToday": 0,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    API output model.
    """

    int_output: int = Field(description="Output of the model. True if tomorrow will rain.")
    str_output: Literal["Tomorrow will probably rain.", "Tomorrow won't probably rain"] = Field(
        description="Output of the model in string form",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": 1,
                    "str_output": "Tomorrow will probably rain.",
                }
            ]
        }
    }


# Load the model before start
model, version_model, info_dict = load_model("rain_australia_model_prod", "champion")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint of the Rain Autralia Detector API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(
        content=jsonable_encoder(
            {"message": "Welcome to the Rain Autralia Detector API"}
        )
    )


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks,
):
    """
    Endpoint for predicting Rain Autralia.

    This endpoint receives features related to a patient's health and predicts whether tomorrow will rain in some
    australian city or not using a trained model. It returns the prediction result in both integer and string formats.
    """
    import logging

    def encode_date_columns(df, columns):
        for col in columns:
            df[col] = pd.to_datetime(df[col])

            df["DayOfYear"] = df[col].dt.dayofyear
            df["DaysInYear"] = df[col].dt.is_leap_year.apply(
                lambda leap: 366 if leap else 365
            )

            df["Angle"] = 2 * np.pi * (df["DayOfYear"] - 1) / (df["DaysInYear"])

            df["DayCos"] = np.cos(df["Angle"])
            df["DaySin"] = np.sin(df["Angle"])

            df = df.drop(columns=["DayOfYear", "DaysInYear", "Angle"])

        return df

    def encode_dir_columns(df, columns):
        dirs = [
            "E",
            "ENE",
            "NE",
            "NNE",
            "N",
            "NNW",
            "NW",
            "WNW",
            "W",
            "WSW",
            "SW",
            "SSW",
            "S",
            "SSE",
            "SE",
            "ESE",
        ]
        angles = np.radians(np.arange(0, 360, 22.5))
        mapping_dict = {d: a for (d, a) in zip(dirs, angles)}

        for col in columns:
            df[f"{col}Angle"] = df[col].map(mapping_dict)

            df[f"{col}Cos"] = np.cos(df[f"{col}Angle"])
            df[f"{col}Sin"] = np.sin(df[f"{col}Angle"])

            df = df.drop(columns=f"{col}Angle")

        return df

    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(
        np.array(features_list).reshape([1, -1]), columns=features_key
    )

    logger = logging.getLogger()

    logger.info(features_df.columns)

    logger.info(features_df.head(1))

    # Add location coords as features (should improve path hardcoding)
    df_locations = wr.s3.read_csv(info_dict["loc_filepath"])
    features_df = pd.merge(features_df, df_locations, how="left", on="Location")

    # Encode date columns
    date_columns = info_dict["date_columns"]
    features_df = encode_date_columns(features_df, date_columns)

    # Encode direction columns
    dir_columns = info_dict["dir_columns"]
    features_df = encode_dir_columns(features_df, dir_columns)

    # Apply skeweness reducing process
    right_skewed_columns = info_dict["right_skewed_columns"]
    logger.info(right_skewed_columns)
    for col in right_skewed_columns:
        logger.info(features_df[col] + 1)
        features_df[f"{col}Log"] = np.log(features_df[col].astype(float) + 1)

    # Cap outliers
    def _cap_outliers(x, limits):
        low_limit, up_limit = limits

        is_over_limits = x > up_limit
        x[is_over_limits] = up_limit

        is_under_limits = x < low_limit
        x[is_under_limits] = low_limit

        return x

    numeric_columns_limits = info_dict["numeric_columns_limits"]
    for feature, limits in numeric_columns_limits.items():
        features_df[feature] = _cap_outliers(features_df[feature], limits)

    # Normalize data
    features = info_dict["features"]
    features_df = features_df[features]

    # Initialise completely the scaler object
    std_scaler = StandardScaler()

    std_scaler.scale_ = np.array(info_dict["std_scaler_scale"])
    std_scaler.mean_ = np.array(info_dict["std_scaler_mean"])
    std_scaler.var_ = np.array(info_dict["std_scaler_var"])

    features_arr = std_scaler.transform(features_df)
    features_df = pd.DataFrame(features_arr, columns=features_df.columns)

    ##########

    # Make the prediction using the trained model
    prediction = model.predict(features_df)

    # Convert prediction result into string format
    str_pred = "Tomorrow won't probably rain"
    if prediction[0] > 0:
        str_pred = "Tomorrow will probably rain."

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=int(prediction[0].item()), str_output=str_pred)
