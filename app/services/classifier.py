import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
try:
    import tensorflow as tf
except Exception:
    tf = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from ..config import CLASSIFIER_DIR

logger = logging.getLogger(__name__)

CATEGORY_LABELS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Cyber Security",
    "Cloud Computing",
]

VECTORIZER_PATH = Path(CLASSIFIER_DIR) / "vectorizer.pkl"
LABEL_ENCODER_PATH = Path(CLASSIFIER_DIR) / "label_encoder.pkl"
MODEL_PATH = Path(CLASSIFIER_DIR) / "document_classifier.keras"


class ClassifierService:
    def __init__(self):
        os.makedirs(CLASSIFIER_DIR, exist_ok=True)
        self.vectorizer = None
        self.label_encoder = None
        self.model = None
        self._load_artifacts()

    def _load_artifacts(self):
        if not tf:
            logger.debug("TensorFlow not available; skipping classifier artifact load")
            return

        if VECTORIZER_PATH.exists() and LABEL_ENCODER_PATH.exists() and Path(MODEL_PATH).exists():
            with open(VECTORIZER_PATH, "rb") as handle:
                self.vectorizer = pickle.load(handle)
            with open(LABEL_ENCODER_PATH, "rb") as handle:
                self.label_encoder = pickle.load(handle)
            self.model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Loaded classifier artifacts from storage")

    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None and self.label_encoder is not None

    def predict(self, text: str):
        if not self.is_ready():
            raise RuntimeError("Classifier model is not available. Train the classifier first.")
        features = self.vectorizer.transform([text]).toarray()
        logits = self.model.predict(features, verbose=0)
        index = int(np.argmax(logits, axis=1)[0])
        label = self.label_encoder.inverse_transform([index])[0]
        confidence = float(np.max(logits))
        return label, confidence

    def train(self, texts, labels, epochs: int = 10, batch_size: int = 16):
        if not tf:
            raise RuntimeError("TensorFlow is not installed in this environment. Install TensorFlow or pin a Python runtime that supports it to enable training.")
        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
        self.label_encoder = LabelEncoder()
        x = self.vectorizer.fit_transform(texts).toarray()
        y = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(x.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2)
        self.model = model

        with open(VECTORIZER_PATH, "wb") as handle:
            pickle.dump(self.vectorizer, handle)
        with open(LABEL_ENCODER_PATH, "wb") as handle:
            pickle.dump(self.label_encoder, handle)
        model.save(MODEL_PATH)
        return model
