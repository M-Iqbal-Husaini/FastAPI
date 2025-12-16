# services/model_loader.py
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

log = logging.getLogger("uvicorn.error")


class ModelWrapper:
    def __init__(self, model, tokenizer, metadata: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata or {}
        self.maxlen = int(self.metadata.get("maxlen", 200))

    def predict_text(self, text: str) -> Tuple[float, int, dict]:
        seq = self.tokenizer.texts_to_sequences([text])
        x = pad_sequences(seq, maxlen=self.maxlen, padding="post", truncating="post")
        pred = self.model.predict(x)
        score = float(pred[0][0])
        label = int(score >= 0.5)
        return score, label, {"raw_pred": pred.tolist()}


def _load_tokenizer(tokenizer_path: Path):
    with tokenizer_path.open("r", encoding="utf-8") as f:
        tok_json = f.read()
    tokenizer = tokenizer_from_json(tok_json)
    return tokenizer


def load_model_from_paths(
    model_path: Path, tokenizer_path: Path, metadata_path: Optional[Path]
) -> Optional[ModelWrapper]:
    try:
        log.info("Loading model file: %s", model_path)
        model = load_model(str(model_path))

        metadata = {}
        if metadata_path and metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            log.info("Loaded metadata from %s", metadata_path)

        tokenizer = None
        if tokenizer_path.exists():
            tokenizer = _load_tokenizer(tokenizer_path)
            log.info("Loaded tokenizer from %s", tokenizer_path)
        else:
            log.warning("Tokenizer file not found at %s", tokenizer_path)

        if tokenizer is None:
            log.error("Tokenizer is None — cannot create ModelWrapper")
            return None

        return ModelWrapper(model=model, tokenizer=tokenizer, metadata=metadata)
    except Exception as e:
        log.exception("Failed to load model/tokenizer: %s", e)
        return None


def load_latest_model_from_public_models(models_dir: str) -> Optional[ModelWrapper]:
    """
    Cari *_run_meta.json terbaru, load model/tokenizer berdasar metadata.
    Kalau tidak ada meta, fallback ke *_model.h5 terbaru + tokenizer.json.
    """
    base = Path(models_dir)
    if not base.exists():
        log.warning("Models dir not found: %s", models_dir)
        return None

    meta_files = sorted(
        base.glob("*_run_meta.json"), key=lambda p: p.stat().st_mtime
    )

    if meta_files:
        last_meta = meta_files[-1]
        try:
            with last_meta.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            log.info("Loaded metadata from %s", last_meta)

            model_path = Path(meta["model_path"])
            tokenizer_path = Path(meta["tokenizer_path"])
            return load_model_from_paths(model_path, tokenizer_path, last_meta)
        except Exception as e:
            log.exception("Error reading latest meta file: %s", e)

    # fallback: cari model h5 terbaru
    h5_files = sorted(
        base.glob("*_model.h5"), key=lambda p: p.stat().st_mtime
    )
    if not h5_files:
        log.warning("No model .h5 found in %s", models_dir)
        return None

    model_path = h5_files[-1]
    tokenizer_path = base / "tokenizer.json"
    metadata_path = None

    # kalau ada meta dengan timestamp sama, pakai
    ts_prefix = model_path.name.split("_")[0]
    candidate_meta = list(base.glob(f"{ts_prefix}_*_run_meta.json"))
    if candidate_meta:
        metadata_path = candidate_meta[0]

    return load_model_from_paths(model_path, tokenizer_path, metadata_path)
