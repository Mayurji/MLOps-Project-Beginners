import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load model during startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Iris Prediction API")


class IrisRequest(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Model API is running"}


@app.post("/predict")
def predict(request: IrisRequest):
    prediction = model.predict([request.features])

    return {
        "prediction": int(prediction[0])
    }