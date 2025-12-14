
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def train_model():

    data = load_iris()
    X, y = data.data, data.target
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

    models = {
        "SVM" : SVC(),
        "Tree": DecisionTreeClassifier(),
        "RF"  : RandomForestClassifier(),
        "KNN" : KNeighborsClassifier(),
        "NB"  : GaussianNB()
    }

    acc = {}
    run_ids = {}
    mlflow.set_experiment("Iris_Model")
    with mlflow.start_run(run_name="Model_Selection") as run:
        for model_name, model in models.items():
            score = cross_val_score(model,x_train,y_train,cv=5).mean()
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            test_acc = accuracy_score(y_test,y_pred)
            mlflow.log_metric(f"{model_name}_cv",score)
            mlflow.log_metric(f"{model_name}_test",test_acc)
            mlflow.sklearn.log_model(model,model_name)
            acc[model_name] = score
            run_ids[model_name] = run.info.run_id

    best = max(acc,key=acc.get)
    uri = f"runs:/{run_ids[best]}/{best}"
    return best, uri

if __name__=="__main__":
    value,c=train_model()
    print(value,c)
