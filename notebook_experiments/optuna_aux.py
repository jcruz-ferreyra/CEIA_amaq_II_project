import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (
                abs(winner - study.best_value) / study.best_value
            ) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(
                f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}"
            )


def objective(trial, X_train, y_train, experiment_id):
    """
    Optimize hyperparameters for a classifier using Optuna.

    Parameters:
    -----------
    trial : optuna.trial.Trial
        A trial is a process of evaluating an objective function.
    X_train : pandas.DataFrame
        Input features for training.
    y_train : pandas.Series
        Target variable for training.
    experiment_id : int
        ID of the MLflow experiment where results will be logged.

    Returns:
    --------
    float
        Mean F1 score of the classifier after cross-validation.
    """

    # Comienza el run de MLflow. Este run debería ser el hijo del run padre,
    # así se anidan los diferentes experimentos.
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"Trial: {trial.number}", nested=True
    ):

        # Parámetros a logguear
        params = {"objective": "clas:f1", "eval_metric": "f1"}

        # Sugiere valores para los hiperparámetros utilizando el objeto trial de optuna.
        classifier_name = trial.suggest_categorical(
            "classifier",
            ["LogisticRegression", "DecisionTreeClassifier", "RandomForest"],
        )
        if classifier_name == "LogisticRegression":
            params["model"] = "LogisticRegression"

            # Regularization parameter C (inverse of regularization strength)
            lr_c = trial.suggest_float("lr_c", 0.01, 100, log=True)

            classifier_obj = LogisticRegression(C=lr_c)

            # Saving parameters for logging
            params["model"] = "LogisticRegression"
            params["lr_c"] = lr_c

        elif classifier_name == "DecisionTreeClassifier":
            # Decision Tree Classifier
            tree_max_depth = trial.suggest_int(
                "tree_max_depth", 2, 32, log=True
            )  # Máxima profundidad del arbol

            classifier_obj = DecisionTreeClassifier(max_depth=tree_max_depth)

            params["model"] = "DecisionTreeClassifier"
            params["tree_max_depth"] = tree_max_depth

        else:
            # Random Forest Classifier
            rf_max_depth = trial.suggest_int(
                "rf_max_depth", 2, 32, log=True
            )  # Máxima profundidad de los arboles
            rf_n_estimators = trial.suggest_int(
                "rf_n_estimators", 2, 10, log=True
            )  # Número de arboles

            classifier_obj = RandomForestClassifier(
                max_depth=rf_max_depth, n_estimators=rf_n_estimators
            )

            params["model"] = "RandomForestClassifier"
            params["rf_max_depth"] = rf_max_depth
            params["rf_n_estimators"] = rf_n_estimators

        # Realizamos validación cruzada y calculamos el score F1
        score = cross_val_score(
            classifier_obj,
            X_train,
            y_train.to_numpy().ravel(),
            n_jobs=-1,
            cv=5,
            scoring="f1",
        )

        # Log los hiperparámetros a MLflow
        mlflow.log_params(params)
        # Y el score f1 medio de la validación cruzada.
        mlflow.log_metric("f1", score.mean())

    return score.mean()
