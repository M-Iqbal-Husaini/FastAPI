# services/training_service.py
import os
import json
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Optional, Callable

import numpy as np
from sklearn.model_selection import train_test_split

os.environ.setdefault("MPLBACKEND", "Agg")

log = logging.getLogger("uvicorn.error")

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def compute_class_weight(labels: np.ndarray) -> Optional[Dict[int, float]]:
    if labels is None or len(labels) == 0:
        return None
    counts = Counter(labels.tolist())
    if len(counts) <= 1:
        return None
    total = sum(counts.values())
    weights = {int(cls): (total / (len(counts) * cnt)) for cls, cnt in counts.items()}
    return weights


def choose_maxlen(sequences: list, requested_maxlen: int) -> int:
    lengths = np.array([len(s) for s in sequences], dtype=np.int32)
    if lengths.size == 0:
        return requested_maxlen
    p95 = int(np.percentile(lengths, 95))
    chosen = max(32, min(requested_maxlen, max(p95, 32)))
    return chosen


def _make_tf_dataset(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    cache: bool = False,
):
    ds = tf.data.Dataset.from_tensor_slices((x_arr, y_arr))
    if cache:
        try:
            ds = ds.cache()
        except Exception:
            pass
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(x_arr)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _report(progress_cb: Optional[Callable[[Dict[str, Any]], None]], **kwargs):
    """Helper aman untuk kirim progress ke callback (Laravel)."""
    if progress_cb is None:
        return
    try:
        progress_cb(kwargs)
    except Exception:
        log.exception("progress_cb failed")


class ProgressCallback(tf.keras.callbacks.Callback):
    """Callback Keras untuk kirim progress tiap epoch (tahap 'model')."""

    def __init__(self, total_epochs: int, cb: Optional[Callable[[Dict[str, Any]], None]]):
        super().__init__()
        self.total_epochs = max(int(total_epochs), 1)
        self.cb = cb

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.cb is None:
            return
        done = epoch + 1
        percent = int(done / self.total_epochs * 100)

        try:
            msg = (
                f"Epoch {done}/{self.total_epochs} "
                f"loss={logs.get('loss', 0):.4f}, "
                f"val_loss={logs.get('val_loss', 0):.4f}, "
                f"acc={logs.get('accuracy', 0):.4f}, "
                f"val_acc={logs.get('val_accuracy', 0):.4f}"
            )
        except Exception:
            msg = f"Epoch {done}/{self.total_epochs}"

        payload = {
            "step": "model",
            "percent": percent,
            "message": msg,
            "epoch": done,
            "total_epochs": self.total_epochs,
            "logs": {k: float(v) for k, v in logs.items() if v is not None},
        }
        _report(self.cb, **payload)


def train_lstm_on_texts(
    texts: list,
    labels: list,
    output_dir: str,
    model_name_prefix: str = "train",
    max_words: int = 20000,
    maxlen: int = 200,
    embedding_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 128,
    val_split: float = 0.1,
    limit: Optional[int] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Train LSTM dan simpan model + tokenizer + metadata.
    progress_cb: fungsi(dict) yang akan dipanggil di setiap tahap.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not texts or len(texts) == 0:
        raise ValueError("No texts provided for training")
    if labels is None or len(labels) != len(texts):
        raise ValueError("Labels missing or length mismatch")

    if limit is not None and limit > 0 and limit < len(texts):
        texts = texts[:limit]
        labels = labels[:limit]
        log.info("Using subset limit=%s (n=%s)", limit, len(texts))

    # Tahap 1: Pengumpulan Data
    pos = int(sum(labels))
    neg = len(labels) - pos
    _report(
        progress_cb,
        step="data_collection",
        percent=100,
        message=f"Dataset siap: {len(texts)} teks (positif={pos}, negatif={neg})",
        total=len(texts),
        positive=pos,
        negative=neg,
    )

    # Tokenisasi
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    chosen_maxlen = choose_maxlen(sequences, maxlen)
    x = pad_sequences(
        sequences, maxlen=chosen_maxlen, padding="post", truncating="post"
    )
    y = np.array(labels, dtype=np.float32)

    # Tahap 2: Preprocessing
    _report(
        progress_cb,
        step="preprocess",
        percent=100,
        message=f"Tokenisasi & padding selesai. maxlen={chosen_maxlen}, vocab={len(tokenizer.word_index)}",
        maxlen=chosen_maxlen,
        vocab_size=len(tokenizer.word_index),
    )

    # Split train/val
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=val_split, random_state=42, shuffle=True
    )

    # Tahap 3: Split Data
    _report(
        progress_cb,
        step="split",
        percent=100,
        message=f"Split train/val: train={len(x_train)}, val={len(x_val)}",
        train_size=len(x_train),
        val_size=len(x_val),
        val_split=val_split,
    )

    vocab_size = min(max_words, len(tokenizer.word_index) + 1)

    # Bangun model LSTM
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size, output_dim=embedding_dim, input_length=chosen_maxlen
            ),
            LSTM(128, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ProgressCallback(epochs, progress_cb),
    ]

    class_weight = compute_class_weight(y_train)
    if class_weight:
        log.info("Using class_weight: %s", class_weight)

    cache_ok = (x_train.nbytes + x_val.nbytes) < (2 * 1024**3)
    train_ds = _make_tf_dataset(
        x_train, y_train, batch_size=batch_size, shuffle=True, cache=cache_ok
    )
    val_ds = _make_tf_dataset(
        x_val, y_val, batch_size=batch_size, shuffle=False, cache=False
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    trained_epochs = len(history.history.get("loss", []))
    last_val_acc = float(history.history.get("val_accuracy", [0.0])[-1] or 0.0)
    last_val_loss = float(history.history.get("val_loss", [0.0])[-1] or 0.0)

    # Tahap 5: Evaluasi
    _report(
        progress_cb,
        step="evaluation",
        percent=100,
        message=f"Training selesai. val_acc={last_val_acc:.4f}, val_loss={last_val_loss:.4f}",
        val_accuracy=last_val_acc,
        val_loss=last_val_loss,
        epochs_trained=trained_epochs,
    )

    # Simpan model + tokenizer + metadata
    ts = str(int(np.floor(__import__("time").time())))
    model_fname = f"{ts}_{model_name_prefix}_model.h5"
    tok_fname = "tokenizer.json"
    meta_fname = f"{ts}_{model_name_prefix}_run_meta.json"

    model_path = Path(output_dir) / model_fname
    tokenizer_path = Path(output_dir) / tok_fname
    metadata_path = Path(output_dir) / meta_fname

    model.save(str(model_path))
    log.info("Saved model to %s", model_path)

    tok_json = tokenizer.to_json()
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        f.write(tok_json)
    log.info("Saved tokenizer to %s", tokenizer_path)

    meta = {
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": vocab_size,
        "maxlen": chosen_maxlen,
        "max_words": max_words,
        "embedding_dim": embedding_dim,
        "history": {
            "loss": history.history.get("loss", []),
            "val_loss": history.history.get("val_loss", []),
            "accuracy": history.history.get("accuracy", []),
            "val_accuracy": history.history.get("val_accuracy", []),
        },
        "epochs_trained": trained_epochs,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved metadata to %s", metadata_path)

    return {
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "metadata_path": str(metadata_path),
        "history": meta["history"],
        "epochs": trained_epochs,
    }
