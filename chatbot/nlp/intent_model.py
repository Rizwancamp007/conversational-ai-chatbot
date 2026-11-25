"""Intent classifier: a small Keras text model over the intents dataset.

The model reads the same preprocessed text the rest of the chatbot uses, so
a message is classified exactly as it was written in the dataset. If the
best guess is not confident enough, or the message has no words the model
knows, the answer is the ``fallback`` intent and the caller can ask the
user to rephrase or hand over to a human.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from chatbot.dataset import DEFAULT_PATH as DATASET_PATH
from chatbot.nlp.preprocess import (
    QUESTION_WORDS,
    preprocess,
    preprocess_to_string,
)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "intent_model.keras"
DEFAULT_LABELS_PATH = MODELS_DIR / "labels.json"
DEFAULT_METRICS_PATH = MODELS_DIR / "metrics.json"

DEFAULT_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))
DEFAULT_SEED = 42
MAX_TOKENS = 1000
MAX_LEN = 24


def set_seeds(seed=DEFAULT_SEED):
    """Fix the random seeds so a training run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


FALLBACK_TAG = "fallback"


def load_training_data(path=None):
    """Return (texts, tags, labels) for the dataset at ``path``.

    Texts are preprocessed strings, tags are the intent of each text, and
    labels is the sorted list of intent tags. The ``fallback`` intent is
    left out on purpose: it is not a class the model can learn from, it is
    what the classifier returns when no class wins (see ``predict``).
    """
    from chatbot.dataset import load_intents

    data = load_intents(path)
    texts, tags = [], []
    for intent in data["intents"]:
        if intent["tag"] == FALLBACK_TAG:
            continue
        for pattern in intent["patterns"]:
            text = preprocess_to_string(pattern)
            if text:
                texts.append(text)
                tags.append(intent["tag"])
    labels = sorted(set(tags))
    return texts, tags, labels


def build_model(vocab_size, num_labels, max_tokens=MAX_TOKENS,
                embedding_dim=128, hidden_dim=128, dropout=0.2):
    """Text vectorizer + embedding + pooled dense classifier."""
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=MAX_LEN,
    )
    model = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(hidden_dim, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_labels, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(texts, tags, labels, epochs=300, batch_size=16, seed=DEFAULT_SEED,
          validation_split=0.1, verbose=0):
    """Train a fresh classifier on the given texts. Returns the model."""
    set_seeds(seed)
    label_index = {tag: i for i, tag in enumerate(labels)}
    y = np.array([label_index[t] for t in tags], dtype=np.int32)
    model = build_model(MAX_TOKENS, len(labels))
    model.layers[0].adapt(texts)
    model.fit(
        tf.convert_to_tensor(texts, dtype=tf.string),
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )
    return model


def split_data(texts, tags, test_size=0.2, seed=DEFAULT_SEED):
    """Stratified train/test split, so every intent appears in both halves."""
    from sklearn.model_selection import train_test_split

    return train_test_split(texts, tags, test_size=test_size,
                            random_state=seed, stratify=tags)


def evaluate_model(model, labels, texts, tags):
    """Score a trained model. Returns accuracy plus per-intent numbers."""
    from sklearn.metrics import accuracy_score, classification_report

    probs = model.predict(tf.convert_to_tensor(texts, dtype=tf.string),
                          verbose=0)
    predicted = [labels[i] for i in np.argmax(probs, axis=1)]
    report = classification_report(tags, predicted, labels=labels,
                                   output_dict=True, zero_division=0)
    per_intent = {
        tag: {
            "precision": round(report[tag]["precision"], 3),
            "recall": round(report[tag]["recall"], 3),
            "f1": round(report[tag]["f1-score"], 3),
            "support": int(report[tag]["support"]),
        }
        for tag in labels
    }
    return {
        "accuracy": round(float(accuracy_score(tags, predicted)), 3),
        "samples": len(tags),
        "per_intent": per_intent,
    }


class IntentClassifier:
    """A trained intent model, ready to classify one message at a time."""

    def __init__(self, model, labels, threshold=DEFAULT_THRESHOLD):
        self.model = model
        self.labels = list(labels)
        self.threshold = float(threshold)
        self._vocabulary = set(model.layers[0].get_vocabulary())

    @classmethod
    def load(cls, model_path=None, labels_path=None, threshold=None):
        """Load a model saved by ``scripts/train.py``."""
        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        labels_path = Path(labels_path) if labels_path else DEFAULT_LABELS_PATH
        model = tf.keras.models.load_model(model_path)
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        return cls(model, labels, threshold=threshold)

    def save(self, model_path=None, labels_path=None):
        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        labels_path = Path(labels_path) if labels_path else DEFAULT_LABELS_PATH
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        labels_path.write_text(json.dumps(self.labels), encoding="utf-8")
        return model_path, labels_path

    def predict(self, text):
        """Classify one message.

        Returns ``{"intent": tag, "confidence": float, "probabilities": {...}}``.
        The intent is ``fallback`` when the model is unsure, when the message
        is empty, or when none of its words appear in the training data.
        """
        tokens = preprocess(text)
        probs = self._probabilities(" ".join(tokens)) if tokens else None
        if probs is None:
            return {"intent": "fallback", "confidence": 0.0, "probabilities": {}}
        best = int(np.argmax(probs))
        confidence = float(probs[best])
        # A message whose content words never appeared in training cannot
        # belong to any intent, however sure the model sounds. Messages made
        # of question words only ("how are you") have no content words, so
        # the confidence check decides for those.
        content = [tok for tok in tokens if tok not in QUESTION_WORDS]
        known = True if not content else any(
            tok in self._vocabulary for tok in content
        )
        if not known or confidence < self.threshold:
            intent = "fallback"
        else:
            intent = self.labels[best]
        return {
            "intent": intent,
            "confidence": round(confidence, 3),
            "probabilities": {
                tag: round(float(p), 3)
                for tag, p in zip(self.labels, probs)
            },
        }

    def predict_top(self, text, n=3):
        """The top ``n`` intents for a message, best first."""
        result = self.predict(text)
        ranked = sorted(result["probabilities"].items(),
                        key=lambda item: item[1], reverse=True)
        return ranked[:n]

    def _probabilities(self, text):
        if not text.strip():
            return None
        return self.model.predict(
            tf.convert_to_tensor([text], dtype=tf.string), verbose=0)[0]
