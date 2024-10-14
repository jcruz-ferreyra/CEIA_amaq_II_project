from airflow.decorators import task


@task.virtualenv(
    task_id="train_the_challenger_model",
    requirements=["scikit-learn==1.3.2", "mlflow==2.10.2", "awswrangler==3.6.0"],
    system_site_packages=True,
)
def train_the_challenger_model():
    import datetime

    import awswrangler as wr
    import mlflow
    from mlflow.models import infer_signature
    from sklearn.base import clone
    from sklearn.metrics import f1_score

    mlflow.set_tracking_uri("http://mlflow:5000")

    def load_the_champion_model():
        model_name = "rain_australia_model_prod"
        alias = "champion"

        client = mlflow.MlflowClient()
        model_data = client.get_model_version_by_alias(model_name, alias)

        champion_version = mlflow.sklearn.load_model(model_data.source)

        return champion_version

    def load_the_train_test_data(dataset_name):
        X_train = wr.s3.read_csv(f"s3://data/final/{dataset_name}_train_X.csv")
        y_train = wr.s3.read_csv(f"s3://data/final/{dataset_name}_train_y.csv")
        X_test = wr.s3.read_csv(f"s3://data/final/{dataset_name}_test_X.csv")
        y_test = wr.s3.read_csv(f"s3://data/final/{dataset_name}_test_y.csv")

        return X_train, y_train, X_test, y_test

    def mlflow_track_experiment(model, X):
        # Track the experiment
        experiment = mlflow.set_experiment("Rain in Australia")

        mlflow.start_run(
            run_name="Challenger_run_"
            + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
            experiment_id=experiment.experiment_id,
            tags={"experiment": "challenger models", "dataset": "Rain in Australia"},
            log_system_metrics=True,
        )

        params = model.get_params()
        params["model"] = type(model).__name__

        mlflow.log_params(params)

        # Save the artifact of the challenger model
        artifact_path = "model"

        signature = infer_signature(X, model.predict(X))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            serialization_format="cloudpickle",
            registered_model_name="rain_australia_model_dev",
            metadata={"model_data_version": 1},
        )

        # Obtain the model URI
        return mlflow.get_artifact_uri(artifact_path)

    def register_challenger(model, f1_score, model_uri):
        client = mlflow.MlflowClient()
        name = "rain_australia_model_prod"

        # Save the model params as tags
        tags = model.get_params()
        tags["model"] = type(model).__name__
        tags["f1-score"] = f1_score

        # Save the version of the model
        result = client.create_model_version(
            name=name, source=model_uri, run_id=model_uri.split("/")[-3], tags=tags
        )

        # Save the alias as challenger
        client.set_registered_model_alias(name, "challenger", result.version)

    dataset_name = "rain_australia"

    # Load the champion model
    champion_model = load_the_champion_model()

    # Clone the model
    challenger_model = clone(champion_model)

    # Load the dataset
    X_train, y_train, X_test, y_test = load_the_train_test_data(dataset_name)

    # Fit the training model
    challenger_model.fit(X_train, y_train.to_numpy().ravel())

    # Obtain the metric of the model
    y_pred = challenger_model.predict(X_test)
    f1_score = f1_score(y_test.to_numpy().ravel(), y_pred)

    # Track the experiment
    artifact_uri = mlflow_track_experiment(challenger_model, X_train)

    # Record the model
    register_challenger(challenger_model, f1_score, artifact_uri)

    return dataset_name


@task.virtualenv(
    task_id="train_the_challenger_model",
    requirements=["scikit-learn==1.3.2", "mlflow==2.10.2", "awswrangler==3.6.0"],
    system_site_packages=True,
)
def evaluate_champion_challenge(dataset_name):
    import awswrangler as wr
    import mlflow
    from sklearn.metrics import f1_score

    mlflow.set_tracking_uri("http://mlflow:5000")

    def load_the_model(alias):
        model_name = "rain_australia_model_prod"

        client = mlflow.MlflowClient()
        model_data = client.get_model_version_by_alias(model_name, alias)

        model = mlflow.sklearn.load_model(model_data.source)

        return model

    def load_the_test_data(dataset_name):
        X_test = wr.s3.read_csv(f"s3://data/final/{dataset_name}_test_X.csv")
        y_test = wr.s3.read_csv(f"s3://data/final/{dataset_name}_test_y.csv")

        return X_test, y_test

    def promote_challenger(name):
        client = mlflow.MlflowClient()

        # Demote the champion
        client.delete_registered_model_alias(name, "champion")

        # Load the challenger from registry
        challenger_version = client.get_model_version_by_alias(name, "challenger")

        # delete the alias of challenger
        client.delete_registered_model_alias(name, "challenger")

        # Transform in champion
        client.set_registered_model_alias(name, "champion", challenger_version.version)

    def demote_challenger(name):
        client = mlflow.MlflowClient()

        # delete the alias of challenger
        client.delete_registered_model_alias(name, "challenger")

    # Load the champion model
    champion_model = load_the_model("champion")

    # Load the challenger model
    challenger_model = load_the_model("challenger")

    # Load the dataset
    X_test, y_test = load_the_test_data(dataset_name)

    # Obtain the metric of the models
    y_pred_champion = champion_model.predict(X_test)
    f1_score_champion = f1_score(y_test.to_numpy().ravel(), y_pred_champion)

    y_pred_challenger = challenger_model.predict(X_test)
    f1_score_challenger = f1_score(y_test.to_numpy().ravel(), y_pred_challenger)

    experiment = mlflow.set_experiment("Rain in Australia")

    # Obtain the last experiment run_id to log the new information
    list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

    with mlflow.start_run(run_id=list_run[0].info.run_id):
        mlflow.log_metric("test_f1_challenger", f1_score_challenger)
        mlflow.log_metric("test_f1_champion", f1_score_champion)

        if f1_score_challenger > f1_score_champion:
            mlflow.log_param("Winner", "Challenger")
        else:
            mlflow.log_param("Winner", "Champion")

    name = "rain_australia_model_prod"
    if f1_score_challenger > f1_score_champion:
        promote_challenger(name)
    else:
        demote_challenger(name)
