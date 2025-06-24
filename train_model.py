import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)
    print(f"Accuracy: {acc:.4f}")

    mlflow.log_metric("accuracy",acc)
    mlflow.sklearn.log_model(model, artifact_path="rf_model")

    model_uri =f"runs:/{run.info.run_id}/rf_model"
    result=mlflow.register_model(model_uri,"IrisRFModel")
    print("Model registered:", result.name, "version:", result.version)