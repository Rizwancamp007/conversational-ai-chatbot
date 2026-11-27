"""Tests for entity extraction: every entity type, messy/Roman-Urdu input."""

from datetime import date

import pytest

from chatbot.nlp.entities import extract_entities, extract_slots

REF = date(2026, 11, 26)  # a Thursday


def by_type(text, reference_date=REF):
    out = {}
    for entity in extract_entities(text, reference_date):
        out.setdefault(entity["type"], []).append(entity)
    return out


def value(text, type_, reference_date=REF):
    found = by_type(text, reference_date).get(type_)
    return found[0]["value"] if found else None


# --- course / subject -------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("I want admission in FSc", "fsc"),
    ("admission in fsc pre-medical", "fsc pre-medical"),
    ("ICS ki fee kitni hai", "ics"),
    ("pre medical admission", "fsc pre-medical"),
    ("FA admission process", "fa"),
])
def test_course(text, expected):
    assert value(text, "course") == expected


@pytest.mark.parametrize("text,expected", [
    ("who teaches physics", "physics"),
    ("chemistry ki fee", "chemistry"),
    ("maths teacher name", "mathematics"),
    ("computer science class timing", "computer science"),
    ("bio ka test kab hai", "biology"),
    ("pak studies teacher", "pakistan studies"),
])
def test_subject(text, expected):
    assert value(text, "subject") == expected


def test_course_and_subject_together():
    slots = extract_slots("FSc pre-medical mein physics parhate hain?")
    assert slots["course"] == "fsc pre-medical"
    assert slots["subject"] == "physics"


# --- class level ------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("9th class ki fee", "9th"),
    ("class 10 timetable", "10th"),
    ("1st year admission", "1st year"),
    ("2nd year result", "2nd year"),
    ("first year classes", "1st year"),
    ("matric admission", "matric"),
    ("intermediate classes", "intermediate"),
])
def test_class_level(text, expected):
    assert value(text, "class_level") == expected


# --- day / date --------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("kal test hai?", "2026-11-27"),
    ("tomorrow class hai", "2026-11-27"),
    ("aaj ki class", "2026-11-26"),
    ("parso exam hai", "2026-11-28"),
    ("12 March ko test hai", "2026-03-12"),
    ("March 12 test", "2026-03-12"),
    ("12/03 test hai", "2026-03-12"),
])
def test_date(text, expected):
    assert value(text, "date") == expected


@pytest.mark.parametrize("text,expected", [
    ("Monday ki class", "monday"),
    ("next Friday test hai", "friday"),
    ("juma ko class hai", "friday"),
    ("peer ko test", "monday"),
])
def test_day(text, expected):
    assert value(text, "day") == expected


def test_invalid_date_ignored():
    assert value("31 February test", "date") is None


# --- time --------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("class 10 am hai", "10:00"),
    ("test 3:30 pm hai", "15:30"),
    ("10 baje class hai", "10:00"),
    ("shaam 5 baje class", "17:00"),
    ("subah 9 baje test", "09:00"),
    ("12:00 pm class", "12:00"),
    ("12 am class", "00:00"),
])
def test_time(text, expected):
    assert value(text, "time") == expected


# --- phone / email -----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("my number is 0300-1234567", "+923001234567"),
    ("call me 03001234567", "+923001234567"),
    ("+92 300 1234567", "+923001234567"),
    ("contact 0321 9876543", "+923219876543"),
])
def test_phone(text, expected):
    assert value(text, "phone") == expected


def test_email():
    assert value("my email is ahmed.khan@gmail.com",
                 "email") == "ahmed.khan@gmail.com"


def test_phone_not_also_an_amount():
    entities = by_type("call 0300-1234567")
    assert "phone" in entities
    assert "amount" not in entities


# --- student name ------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("my name is Ahmed Khan", "Ahmed Khan"),
    ("mera naam Ali hai", "Ali"),
    ("I am Bilal Ahmed", "Bilal Ahmed"),
    ("my name is ahmed", "Ahmed"),
])
def test_name_by_cue(text, expected):
    assert value(text, "student_name") == expected


def test_name_not_a_sentence():
    assert value("I am a student", "student_name") is None
    assert value("I am looking for admission", "student_name") is None


def test_name_by_chunking():
    # No cue word: NLTK named-entity chunking finds the name.
    # (Multi-word names come from "my name is ..." cues; the chunker
    # sometimes splits them, e.g. "Ahmed Khan" -> PERSON + ORGANIZATION.)
    assert value("Ahmed's result please", "student_name") == "Ahmed"


# --- amount ------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("fee Rs 5000 hai", 5000),
    ("Rs. 12,000 fee", 12000),
    ("5000 rupees fee", 5000),
    ("PKR 2500 per month", 2500),
    ("fee 5k hai", 5000),
    ("2 thousand discount", 2000),
])
def test_amount(text, expected):
    assert value(text, "amount") == expected


# --- spans, slots, empty input ----------------------------------------------

def test_spans_match_the_original_text():
    text = "Physics ki fee Rs 5000 hai, class 10 am hai"
    for entity in extract_entities(text):
        assert text[entity["start"]:entity["end"]] == entity["text"]


def test_entities_sorted_by_position():
    text = "kal 10 am physics ka test hai, fee Rs 5000"
    starts = [e["start"] for e in extract_entities(text)]
    assert starts == sorted(starts)


def test_extract_slots_maps_to_dataset_slots():
    from datetime import timedelta

    slots = extract_slots(
        "my name is Ahmed, 9th class physics ki fee kal kitni hai, "
        "call 03001234567")
    assert slots == {
        "student_name": "Ahmed",
        "class_level": "9th",
        "subject": "physics",
        "date": (date.today() + timedelta(days=1)).isoformat(),
        "phone": "+923001234567",
    }


def test_no_entities_for_plain_messages():
    assert extract_entities("hello") == []
    assert extract_entities("how are you") == []
    assert extract_entities("") == []
    assert extract_entities("asdfgh qwerty") == []


def test_messy_roman_urdu_message():
    from datetime import timedelta

    slots = extract_slots("meri fees kitni hai fsc ki, 9th class, kal test hai?")
    assert slots["course"] == "fsc"
    assert slots["class_level"] == "9th"
    assert slots["date"] == (date.today() + timedelta(days=1)).isoformat()


def test_confidence_is_a_probability():
    for entity in extract_entities(
            "my name is Ahmed, physics 9th class, Rs 5000, 03001234567, "
            "kal 10 am, ahmed@gmail.com"):
        assert 0.0 < entity["confidence"] <= 1.0
