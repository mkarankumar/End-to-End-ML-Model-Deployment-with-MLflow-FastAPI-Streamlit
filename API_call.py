from fastapi import FastAPI 
from pydantic import BaseModel
import pandas as pd 
import mlflow.pyfunc 

model = mlflow.pyfunc.load_model("models:/IrisRFModel/1")

app = FastAPI(title="MLflow Model Prediction API")

class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/a")
def working():
    return {"status": "Working Bro"}

@app.post("/predict/")
def predict(data: InputData):
    input_df = pd.DataFrame([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]], columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ])

    pred = model.predict(input_df)
    return {"prediction": int(pred[0])}
