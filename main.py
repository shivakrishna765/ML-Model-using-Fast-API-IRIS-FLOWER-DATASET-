from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


# Load model
model = joblib.load("iris_model.pkl")

# Create app
app = FastAPI()

# Request body structure
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Index route
@app.get("/")
def read_root():
    return {"message": "Welcome to Iris Species Prediction API"}

# Prediction route
@app.post("/predict")
def predict_species(data: IrisRequest):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(features)[0]
    species = ['setosa', 'versicolor', 'virginica']
    return {"prediction": species[prediction]}
