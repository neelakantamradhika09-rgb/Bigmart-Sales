import time
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import (
    BigMartInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)

from api.model_loader import load_model
from api.preprocessing import preprocess_input
from api.monitoring_logger import log_request


app = FastAPI(
    title="BigMart Sales Prediction API",
    description="Machine Learning API for predicting BigMart Item Outlet Sales",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)

MODEL = None
MODEL_NAME = None
MODEL_ERROR = None


@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_NAME, MODEL_ERROR

    try:
        MODEL, MODEL_NAME = load_model()
        MODEL_ERROR = None
        logging.info("Model loaded successfully: %s", MODEL_NAME)

    except Exception as error:
        MODEL_ERROR = str(error)
        logging.error("Model loading failed: %s", error)


@app.get("/")
def home():
    return {
        "message": "Welcome to BigMart Sales Prediction API",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    if MODEL is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "error": MODEL_ERROR
            }
        )

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/metadata")
def metadata():
    return {
        "project": "BigMart Sales Prediction",
        "model_name": MODEL_NAME,
        "problem_type": "Regression",
        "target_column": "Item_Outlet_Sales",
        "framework": "Scikit-learn",
        "api_version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: BigMartInput):
    start = time.perf_counter()

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    try:
        df = preprocess_input([data.model_dump()])
        prediction = float(MODEL.predict(df)[0])

        latency = time.perf_counter() - start

        log_request(
            "/predict",
            "success",
            latency
        )

        return PredictionResponse(
            predicted_sales=round(prediction, 2),
            status="success"
        )

    except Exception as error:
        latency = time.perf_counter() - start

        log_request(
            "/predict",
            "error",
            latency
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(data: BatchPredictionRequest):
    start = time.perf_counter()

    # First validate the records list
    if not data.records:
        raise HTTPException(
            status_code=400,
            detail="At least one record is required"
        )

    # Then check model availability
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    try:
        records = [
            item.model_dump() for item in data.records
        ]

        df = preprocess_input(records)
        predictions = MODEL.predict(df).tolist()

        latency = time.perf_counter() - start

        log_request(
            "/batch_predict",
            "success",
            latency,
            len(records)
        )

        return BatchPredictionResponse(
            predictions=[
                round(float(p), 2) for p in predictions
            ],
            total_records=len(predictions),
            status="success"
        )

    except Exception as error:
        latency = time.perf_counter() - start

        log_request(
            "/batch_predict",
            "error",
            latency,
            len(data.records)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(error)}"
        )