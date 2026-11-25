"""Tests for the intent classifier: training, prediction, fallback."""

import json

import numpy as np
import pytest

from chatbot.nlp.intent_model import (
    IntentClassifier,
    evaluate_model,
    load_training_data,
    split_data,
    train,
)
from chatbot.nlp.preprocess import ensure_nltk_data

EPOCHS = 300
ACCURACY_FLOOR = 0.70


@pytest.fixture(scope="module")
def trained():
    ensure_nltk_data(download=True)
    texts, tags, labels = load_training_data()
    train_texts, test_texts, train_tags, test_tags = split_data(texts, tags)
    model = train(train_texts, train_tags, labels, epochs=EPOCHS)
    classifier = IntentClassifier(model, labels)
    return classifier, labels, test_texts, test_tags


def test_training_produces_all_labels(trained):
    classifier, labels, _, _ = trained
    assert len(labels) == 17
    assert set(classifier.labels) == set(labels)


def test_fallback_is_a_rule_not_a_learned_label(trained):
    classifier, _, _, _ = trained
    assert "fallback" not in classifier.labels


def test_predict_known_pattern(trained):
    classifier, _, _, _ = trained
    result = classifier.predict("how can I get admission")
    assert result["intent"] == "admissions"
    assert result["confidence"] >= classifier.threshold


def test_predict_fees_in_roman_urdu(trained):
    classifier, _, _, _ = trained
    result = classifier.predict("meri fees kitni hai")
    assert result["intent"] == "fees"
    assert result["confidence"] >= classifier.threshold


def test_probabilities_cover_all_intents_and_sum_to_one(trained):
    classifier, labels, _, _ = trained
    result = classifier.predict("what is the timetable for physics")
    assert set(result["probabilities"]) == set(labels)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=0.01)


def test_predict_top_returns_best_first(trained):
    classifier, _, _, _ = trained
    top = classifier.predict_top("hello there", n=3)
    assert len(top) == 3
    assert top[0][0] == "greeting"
    assert top[0][1] >= top[1][1] >= top[2][1]


def test_fallback_on_gibberish(trained):
    classifier, _, _, _ = trained
    result = classifier.predict("zxqv blorp flibberty wobble")
    assert result["intent"] == "fallback"


def test_fallback_on_empty_message(trained):
    classifier, _, _, _ = trained
    result = classifier.predict("")
    assert result["intent"] == "fallback"
    assert result["confidence"] == 0.0


def test_fallback_on_unknown_topic(trained):
    classifier, _, _, _ = trained
    result = classifier.predict("what is the weather in Lahore tomorrow")
    assert result["intent"] == "fallback"


def test_fallback_on_bitcoin_price(trained):
    classifier, _, _, _ = trained
    result = classifier.predict("what is the bitcoin price today")
    assert result["intent"] == "fallback"


def test_save_and_load_roundtrip(trained, tmp_path):
    classifier, _, _, _ = trained
    model_path = tmp_path / "model.keras"
    labels_path = tmp_path / "labels.json"
    classifier.save(model_path, labels_path)
    assert model_path.exists()
    assert json.loads(labels_path.read_text()) == classifier.labels
    loaded = IntentClassifier.load(model_path, labels_path)
    assert loaded.predict("how can I get admission")["intent"] == "admissions"


def test_held_out_accuracy_floor(trained):
    classifier, labels, test_texts, test_tags = trained
    metrics = evaluate_model(classifier.model, labels, test_texts, test_tags)
    assert metrics["accuracy"] >= ACCURACY_FLOOR
    assert metrics["samples"] == len(test_tags)
    assert set(metrics["per_intent"]) == set(labels)


def test_evaluate_reports_every_intent(trained):
    classifier, labels, test_texts, test_tags = trained
    metrics = evaluate_model(classifier.model, labels, test_texts, test_tags)
    for tag in labels:
        scores = metrics["per_intent"][tag]
        assert 0.0 <= scores["precision"] <= 1.0
        assert 0.0 <= scores["recall"] <= 1.0
        assert 0.0 <= scores["f1"] <= 1.0
        assert scores["support"] > 0
