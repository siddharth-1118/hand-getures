from pathlib import Path
import pickle

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Vision2Sense Gesture Recognition API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "gesture_recognizer.pkl"
model = None


# Load model on startup
@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        try:
            with MODEL_PATH.open("rb") as f:
                model = pickle.load(f)
            print("AI Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Error: Model file '{MODEL_PATH}' not found!")


class LandmarkData(BaseModel):
    landmarks: list[float]  # Should be 63 floats (21 x 3D coordinates)


@app.get("/")
def home():
    return {"status": "Vision2Sense API is running", "model_loaded": model is not None}


@app.post("/predict")
async def predict_gesture(data: LandmarkData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    if len(data.landmarks) != 63:
        raise HTTPException(status_code=400, detail="Invalid landmark data. Expected 63 values.")

    try:
        # Convert to numpy array and reshape for prediction
        X_input = np.array([data.landmarks])

        # Get probabilities
        probabilities = model.predict_proba(X_input)[0]
        max_prob_index = np.argmax(probabilities)

        confidence = float(probabilities[max_prob_index])
        prediction = str(model.classes_[max_prob_index])

        return {
            "prediction": prediction,
            "confidence": confidence,
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
