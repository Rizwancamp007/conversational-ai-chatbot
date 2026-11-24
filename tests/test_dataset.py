import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chatbot.dataset import (DatasetError, KNOWN_SLOTS, MIN_PATTERNS, load_and_validate,
                             load_intents, validate_dataset)
from chatbot.nlp.preprocess import ensure_nltk_data, preprocess_to_string

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def data():
    ensure_nltk_data(download=True)
    return load_intents()


def intent(data, tag):
    return next(i for i in data["intents"] if i["tag"] == tag)


def test_shipped_dataset_is_valid(data):
    errors, _ = validate_dataset(data, preprocess=preprocess_to_string)
    assert errors == []


def test_dataset_covers_the_helpdesk_topics(data):
    tags = {i["tag"] for i in data["intents"]}
    expected = {"greeting", "goodbye", "thanks", "admissions", "courses", "fees",
                "timetable", "tests", "results", "online_classes", "contact",
                "teacher_info", "human_handoff", "fallback"}
    assert expected <= tags
    assert 15 <= len(tags) <= 20


def test_every_intent_has_enough_patterns(data):
    for i in data["intents"]:
        assert len(i["patterns"]) >= MIN_PATTERNS, i["tag"]


def test_dataset_includes_roman_urdu_patterns(data):
    text = " ".join(p for i in data["intents"] for p in i["patterns"]).lower()
    for word in ("kab", "kitni", "dakhla", "shukriya", "kahan"):
        assert word in text


def test_slots_are_known_and_have_prompts(data):
    for i in data["intents"]:
        for s in i["required_slots"]:
            assert s in KNOWN_SLOTS
            assert i["slot_prompts"][s]


def test_rejects_missing_intents_list():
    errors, _ = validate_dataset({"foo": []})
    assert errors


def test_rejects_too_few_patterns(data):
    bad = copy.deepcopy(data)
    intent(bad, "greeting")["patterns"] = ["hi", "hello"]
    errors, _ = validate_dataset(bad)
    assert any("greeting" in e and "at least" in e for e in errors)


def test_rejects_duplicate_tags(data):
    bad = copy.deepcopy(data)
    bad["intents"].append(copy.deepcopy(intent(bad, "thanks")))
    errors, _ = validate_dataset(bad)
    assert any("duplicate tag" in e for e in errors)


def test_rejects_duplicate_pattern_in_one_intent(data):
    bad = copy.deepcopy(data)
    intent(bad, "thanks")["patterns"].append("Thank you!")
    errors, _ = validate_dataset(bad)
    assert any("duplicate pattern" in e for e in errors)


def test_rejects_same_pattern_in_two_intents(data):
    bad = copy.deepcopy(data)
    intent(bad, "fees")["patterns"].append("class timings")
    errors, _ = validate_dataset(bad, preprocess=preprocess_to_string)
    assert any("class timings" in e and "also used by" in e for e in errors)


def test_rejects_unknown_slot(data):
    bad = copy.deepcopy(data)
    intent(bad, "fees")["required_slots"].append("shoe_size")
    errors, _ = validate_dataset(bad)
    assert any("unknown slots" in e for e in errors)


def test_rejects_slot_without_prompt(data):
    bad = copy.deepcopy(data)
    intent(bad, "fees")["slot_prompts"] = {}
    errors, _ = validate_dataset(bad)
    assert any("no follow-up prompt" in e for e in errors)


def test_rejects_placeholder_that_is_not_a_slot(data):
    bad = copy.deepcopy(data)
    intent(bad, "contact")["responses"].append("Call {phone} now")
    errors, _ = validate_dataset(bad)
    assert any("not required slots" in e for e in errors)


def test_rejects_missing_fallback(data):
    bad = copy.deepcopy(data)
    bad["intents"] = [i for i in bad["intents"] if i["tag"] != "fallback"]
    errors, _ = validate_dataset(bad)
    assert any("fallback" in e for e in errors)


def test_load_and_validate_raises_on_bad_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"intents": [{"tag": "greeting", "patterns": ["hi"], "responses": ["hey"]}]}))
    with pytest.raises(DatasetError):
        load_and_validate(path)


def test_validate_script_passes_on_shipped_dataset():
    result = subprocess.run([sys.executable, "scripts/validate_dataset.py"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dataset OK" in result.stdout


def test_validate_script_fails_on_bad_dataset(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"intents": []}))
    result = subprocess.run([sys.executable, "scripts/validate_dataset.py", str(path)], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 1
