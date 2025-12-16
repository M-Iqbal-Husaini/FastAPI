# ================================================================
# main.py — FastAPI LSTM Service with Laravel callback + progress
# ================================================================

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

import logging
from typing import Optional, List

import requests
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel

# ================================================================
# LOGGING
# ================================================================
log = logging.getLogger("uvicorn.error")

# ================================================================
# CONFIG
# ================================================================
LARAVEL_URL = os.environ.get("LARAVEL_URL", "http://127.0.0.1:8000")
INTERNAL_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "HAIKYU2025")

MODELS_PUBLIC_DIR = os.environ.get(
    "MODELS_PUBLIC_DIR",
    "C:/laragon/www/loginGoogle/public/python/models"
)
os.makedirs(MODELS_PUBLIC_DIR, exist_ok=True)

# ================================================================
# INIT FASTAPI
# ================================================================
app = FastAPI(title="FastAPI LSTM Inference Service")

# ================================================================
# IMPORT SERVICES
# ================================================================
from services.model_loader import ModelWrapper, load_latest_model_from_public_models
from services.dataset_service import (
    fetch_dataset_from_laravel,
    fetch_full_dataset_from_laravel,
)
from services.training_service import train_lstm_on_texts

# ================================================================
# GLOBAL MODEL
# ================================================================
model_wrapper: Optional[ModelWrapper] = None

# ================================================================
# PYDANTIC MODELS
# ================================================================
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: float
    label: Optional[int] = None
    raw: Optional[dict] = None


class BatchPredictItem(BaseModel):
    text: str
    prediction: float
    label: int


class BatchPredictRequest(BaseModel):
    texts: List[str]


class BatchPredictResponse(BaseModel):
    results: List[BatchPredictItem]


class TrainRequest(BaseModel):
    dataset_id: int
    epochs: Optional[int] = 5
    max_words: Optional[int] = 20000
    maxlen: Optional[int] = 200
    model_name_prefix: Optional[str] = "train"
    model_info_id: Optional[int] = None   # dikirim dari Laravel


# ================================================================
# STARTUP EVENT
# ================================================================
@app.on_event("startup")
def startup_event():
    global model_wrapper
    log.info("Loading latest model from %s", MODELS_PUBLIC_DIR)
    model_wrapper = load_latest_model_from_public_models(MODELS_PUBLIC_DIR)
    if model_wrapper is None:
        log.warning("No model loaded at startup.")


# ================================================================
# HEALTH CHECK
# ================================================================
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_wrapper is not None}


# ================================================================
# PREDICT (SINGLE)
# ================================================================
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global model_wrapper

    if not model_wrapper or not getattr(model_wrapper, "model", None):
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        score, label, raw = model_wrapper.predict_text(req.text)
        return PredictResponse(prediction=float(score), label=label, raw=raw)
    except Exception as e:
        log.exception("Predict error")
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# PREDICT (BATCH)
# ================================================================
@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    global model_wrapper

    if not model_wrapper or not getattr(model_wrapper, "model", None):
        raise HTTPException(status_code=503, detail="Model not loaded")

    results: List[BatchPredictItem] = []

    try:
        for text in req.texts:
            score, label, raw = model_wrapper.predict_text(text or "")
            results.append(
                BatchPredictItem(
                    text=text or "",
                    prediction=float(score),
                    label=int(label) if label is not None else 0,
                )
            )
    except Exception as e:
        log.exception("Batch predict error")
        raise HTTPException(status_code=500, detail=str(e))

    return BatchPredictResponse(results=results)


# ================================================================
# INTERNAL: FETCH DATASET (PAGE)
# ================================================================
@app.get("/internal/dataset/{dataset_id}")
def internal_fetch_dataset(
    dataset_id: int,
    page: int = 1,
    per_page: int = 1000,
    x_internal_token: Optional[str] = Header(None),
):
    if x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    data = fetch_dataset_from_laravel(
        base_url=LARAVEL_URL,
        dataset_id=dataset_id,
        page=page,
        per_page=per_page,
        internal_token=INTERNAL_TOKEN,
    )
    if data is None:
        raise HTTPException(status_code=404, detail="Dataset not found or empty")

    return data


# ================================================================
# CALLBACK KE LARAVEL (FINAL)
# ================================================================
def send_callback_to_laravel(model_info_id: int, result: dict):
    """
    Kirim hasil training ke Laravel untuk update tabel model_info.
    """
    history = result.get("history", {})

    payload = {
        "model_info_id": model_info_id,
        "status": "trained",
        "train_accuracy": history.get("accuracy", [None])[-1],
        "val_accuracy": history.get("val_accuracy", [None])[-1],
        "train_loss": history.get("loss", [None])[-1],
        "val_loss": history.get("val_loss", [None])[-1],
        "epochs": result.get("epochs"),
        "model_path": result.get("model_path"),
    }

    callback_url = f"{LARAVEL_URL}/api/internal/train/callback"

    headers = {
        "X-INTERNAL-TOKEN": INTERNAL_TOKEN,
        "Content-Type": "application/json",
    }

    try:
        r = requests.post(callback_url, json=payload, headers=headers, timeout=15)
        log.info("Laravel callback response: %s %s", r.status_code, r.text)
    except Exception as e:
        log.error("Callback to Laravel FAILED: %s", e)


# ================================================================
# CALLBACK PROGRESS KE LARAVEL (PER TAHAP)
# ================================================================
def send_progress_to_laravel(model_info_id: int, progress: dict):
    """
    Kirim progress per tahap ke Laravel.
    progress: dict yang minimal berisi 'step', 'percent', 'message'.
    """
    callback_url = f"{LARAVEL_URL}/api/internal/train/progress"
    headers = {
        "X-INTERNAL-TOKEN": INTERNAL_TOKEN,
        "Content-Type": "application/json",
    }

    payload = {
        "model_info_id": model_info_id,
        "step": progress.get("step"),
        "percent": progress.get("percent"),
        "message": progress.get("message"),
        "extra": progress,
    }

    try:
        r = requests.post(callback_url, json=payload, headers=headers, timeout=10)
        log.info("Laravel progress response: %s %s", r.status_code, r.text)
    except Exception as e:
        log.error("Progress callback to Laravel FAILED: %s", e)


# ================================================================
# INTERNAL: BACKGROUND TRAINING (DIPANGGIL DARI LARAVEL)
# ================================================================
@app.post("/internal/train/background")
def trigger_train_background(
    req: TrainRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: Optional[str] = Header(None),
):
    if x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if req.model_info_id is None:
        raise HTTPException(status_code=400, detail="model_info_id is required")

    def _background_job():
        try:
            # 1. Ambil dataset full dari Laravel
            texts, labels = fetch_full_dataset_from_laravel(
                base_url=LARAVEL_URL,
                dataset_id=req.dataset_id,
                per_page=1000,
                internal_token=INTERNAL_TOKEN,
            )

            if len(texts) == 0:
                log.error("Dataset empty for training (dataset_id=%s)", req.dataset_id)
                send_progress_to_laravel(req.model_info_id, {
                    "step": "data_collection",
                    "percent": 100,
                    "message": "Dataset kosong, training dibatalkan.",
                })
                return

            # progress awal setelah data terkumpul
            send_progress_to_laravel(req.model_info_id, {
                "step": "data_collection",
                "percent": 100,
                "message": f"Dataset diambil dari Laravel: {len(texts)} teks.",
                "total": len(texts),
            })

            # 2. Training LSTM dengan callback progress
            def progress_cb(p: dict):
                send_progress_to_laravel(req.model_info_id, p)

            result = train_lstm_on_texts(
                texts=texts,
                labels=labels,
                output_dir=MODELS_PUBLIC_DIR,
                model_name_prefix=req.model_name_prefix,
                max_words=req.max_words,
                maxlen=req.maxlen,
                epochs=req.epochs,
                progress_cb=progress_cb,
            )

            log.info("Training finished (background).")

            # 3. Kirim callback final ke Laravel supaya tabel model_info diupdate
            send_callback_to_laravel(req.model_info_id, result)

            # 4. Reload model terbaru ke memory FastAPI
            global model_wrapper
            model_wrapper = load_latest_model_from_public_models(MODELS_PUBLIC_DIR)

        except Exception as e:
            log.exception("Training failure in background job: %s", e)
            try:
                send_progress_to_laravel(req.model_info_id, {
                    "step": "model",
                    "percent": 0,
                    "message": f"Training gagal: {e}",
                })
            except Exception:
                pass

    background_tasks.add_task(_background_job)

    return {
        "status": "accepted",
        "dataset_id": req.dataset_id,
        "model_info_id": req.model_info_id,
    }


# ================================================================
# (OPSIONAL) SYNC TRAIN (DEBUG VIA CURL/POSTMAN)
# ================================================================
@app.post("/internal/train")
def trigger_train_sync(
    req: TrainRequest,
    x_internal_token: Optional[str] = Header(None),
):
    if x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    texts, labels = fetch_full_dataset_from_laravel(
        base_url=LARAVEL_URL,
        dataset_id=req.dataset_id,
        per_page=1000,
        internal_token=INTERNAL_TOKEN,
    )

    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="Dataset empty")

    # untuk endpoint sync ini progress_cb cukup log ke console
    def progress_cb(p: dict):
        log.info("TRAIN SYNC PROGRESS: %s", p)

    result = train_lstm_on_texts(
        texts=texts,
        labels=labels,
        output_dir=MODELS_PUBLIC_DIR,
        model_name_prefix=req.model_name_prefix,
        max_words=req.max_words,
        maxlen=req.maxlen,
        epochs=req.epochs,
        progress_cb=progress_cb,
    )

    global model_wrapper
    model_wrapper = load_latest_model_from_public_models(MODELS_PUBLIC_DIR)

    return {
        "status": "training_done",
        "dataset_id": req.dataset_id,
        "history": result.get("history", {}),
        "model_path": result.get("model_path"),
    }
