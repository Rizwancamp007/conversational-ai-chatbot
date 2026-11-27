"""Tests for multi-turn dialogue: slot filling, carry-over, switching.

Most tests use a small keyword classifier so the conversation logic is
checked without training a model. One integration test at the end runs a
full three-turn conversation through the real trained model.
"""

import random
from datetime import datetime, timedelta, timezone

import pytest

from chatbot.dialogue.manager import CANCEL_REPLY, DialogueManager
from chatbot.dialogue.state import DialogueState, SessionStore
from chatbot.responses import (
    missing_slots,
    pick_response,
    slot_prompt,
)


# --- a simple stand-in for the intent model --------------------------------

class StubClassifier:
    """Keyword classifier: fast and repeatable, for dialogue logic tests."""

    RULES = [
        ("admission", "admissions"),
        ("fee", "fees"),
        ("timetable", "timetable"),
        ("test", "tests"),
        ("result", "results"),
        ("teacher", "teacher_info"),
        ("human", "human_handoff"),
        ("hello", "greeting"),
        ("hi", "greeting"),
        ("salam", "greeting"),
        ("thanks", "thanks"),
        ("thank you", "thanks"),
        ("bye", "goodbye"),
        ("hafiz", "goodbye"),
    ]

    def predict(self, text):
        low = text.lower()
        for word, tag in self.RULES:
            if word in low:
                return {"intent": tag, "confidence": 0.95,
                        "probabilities": {}}
        return {"intent": "fallback", "confidence": 0.0,
                "probabilities": {}}


@pytest.fixture
def manager():
    return DialogueManager(classifier=StubClassifier(),
                           rng=random.Random(0))


def talk(manager, session, *messages):
    """Run a scripted conversation, return the list of replies."""
    return [manager.handle(session, msg) for msg in messages]


# --- response templates ------------------------------------------------------

def test_pick_response_fills_slots():
    reply = pick_response("admissions",
                          {"course": "fsc", "class_level": "9th"},
                          rng=random.Random(1))
    assert "fsc" in reply
    assert "9th" in reply


def test_pick_response_leaves_unfilled_slot_as_placeholder():
    reply = pick_response("admissions", {"course": "fsc"},
                          rng=random.Random(1))
    assert "{class_level}" in reply


def test_slot_prompt_is_the_dataset_question():
    assert slot_prompt("admissions", "course") == \
        "Which course or program are you interested in?"
    assert slot_prompt("fees", "course") == \
        "Which course do you want the fee for?"
    assert slot_prompt("results", "student_name") == \
        "What is the student's name?"


def test_missing_slots_in_dataset_order():
    assert missing_slots("admissions", {}) == ["course", "class_level"]
    assert missing_slots("admissions", {"course": "fsc"}) == ["class_level"]
    assert missing_slots("admissions",
                         {"course": "fsc", "class_level": "9th"}) == []
    assert missing_slots("fees", {}) == ["course"]


# --- dialogue state ----------------------------------------------------------

def test_state_defaults_to_empty_flow():
    state = DialogueState(session_id="s1")
    assert state.active_intent == ""
    assert state.slots == {}
    assert state.pending_slot == ""
    assert not state.handoff_requested


def test_finish_flow_remembers_last_intent_and_slots():
    state = DialogueState(session_id="s1", active_intent="fees",
                          slots={"course": "ics"})
    state.finish_flow()
    assert state.active_intent == ""
    assert state.slots == {}
    assert state.last_intent == "fees"
    assert state.last_slots == {"course": "ics"}


def test_reset_flow_keeps_history():
    state = DialogueState(session_id="s1", active_intent="fees",
                          slots={"course": "ics"}, pending_slot="course")
    state.add_turn("user", "fees for ics?")
    state.reset_flow()
    assert state.active_intent == ""
    assert state.pending_slot == ""
    assert len(state.history) == 1


def test_state_expires_after_ttl():
    state = DialogueState(session_id="s1")
    old = datetime.now(timezone.utc) - timedelta(minutes=31)
    state.updated_at = old
    assert state.is_expired(ttl_seconds=30 * 60)
    assert not state.is_expired(ttl_seconds=60 * 60)


def test_store_returns_fresh_state_after_expiry():
    store = SessionStore(ttl_seconds=60)
    first = store.get_or_create("s1")
    first.active_intent = "fees"
    first.updated_at -= timedelta(minutes=5)
    second = store.get_or_create("s1")
    assert second is not first
    assert second.active_intent == ""


def test_store_prune_drops_expired_only():
    store = SessionStore(ttl_seconds=60)
    store.get_or_create("old").updated_at -= timedelta(minutes=5)
    store.get_or_create("new")
    assert store.prune() == 1
    assert len(store) == 1


def test_state_dict_roundtrip():
    state = DialogueState(session_id="s1", active_intent="fees",
                          slots={"course": "ics"}, pending_slot="course")
    state.add_turn("user", "fees for ics?")
    restored = DialogueState.from_dict(state.to_dict())
    assert restored.session_id == "s1"
    assert restored.active_intent == "fees"
    assert restored.slots == {"course": "ics"}
    assert restored.pending_slot == "course"
    assert len(restored.history) == 1


# --- slot filling ------------------------------------------------------------

def test_admissions_three_turn_flow(manager):
    replies = talk(manager, "s1",
                   "I want admission",
                   "FSc",
                   "9th class")
    assert replies[0]["intent"] == "admissions"
    assert replies[0]["pending_slot"] == "course"
    assert replies[0]["reply"] == \
        "Which course or program are you interested in?"
    assert replies[1]["pending_slot"] == "class_level"
    assert replies[1]["reply"] == \
        "Which class or level is the student in?"
    assert replies[2]["pending_slot"] == ""
    assert "fsc" in replies[2]["reply"].lower()
    assert "9th" in replies[2]["reply"]


def test_two_slots_in_one_message(manager):
    replies = talk(manager, "s1", "I want admission in FSc, 9th class")
    assert replies[0]["pending_slot"] == ""
    assert "fsc" in replies[0]["reply"].lower()
    assert "9th" in replies[0]["reply"]


def test_fees_answered_in_one_turn_when_course_given(manager):
    replies = talk(manager, "s1", "meri fees kitni hai fsc ki")
    assert replies[0]["intent"] == "fees"
    assert replies[0]["pending_slot"] == ""
    assert "fsc" in replies[0]["reply"].lower()


def test_results_asks_for_name_then_answers(manager):
    replies = talk(manager, "s1",
                   "what is my result",
                   "my name is Ahmed Khan")
    assert replies[0]["pending_slot"] == "student_name"
    assert replies[1]["pending_slot"] == ""
    assert "Ahmed Khan" in replies[1]["reply"]


# --- context carry-over ------------------------------------------------------

def test_follow_up_keeps_topic_and_changes_course(manager):
    replies = talk(manager, "s1",
                   "fees for FSc",
                   "and for ICS?")
    assert replies[0]["intent"] == "fees"
    assert "fsc" in replies[0]["reply"].lower()
    assert replies[1]["intent"] == "fees"
    assert "ics" in replies[1]["reply"].lower()


def test_follow_up_keeps_topic_for_timetable(manager):
    replies = talk(manager, "s1",
                   "timetable for FSc",
                   "what about ICS")
    assert replies[1]["intent"] == "timetable"
    assert "ics" in replies[1]["reply"].lower()


# --- intent switching --------------------------------------------------------

def test_switch_topic_mid_flow(manager):
    replies = talk(manager, "s1",
                   "I want admission",
                   "what is the timetable for ICS")
    assert replies[0]["pending_slot"] == "course"
    assert replies[1]["intent"] == "timetable"
    assert replies[1]["pending_slot"] == ""
    assert "ics" in replies[1]["reply"].lower()


def test_switch_drops_old_slots(manager):
    replies = talk(manager, "s1",
                   "I want admission",
                   "FSc",
                   "what are the fees for ICS")
    assert replies[1]["pending_slot"] == "class_level"
    assert replies[2]["intent"] == "fees"
    assert replies[2]["slots"] == {"course": "ics"}


# --- cancel / reset ----------------------------------------------------------

def test_cancel_mid_flow_starts_fresh(manager):
    replies = talk(manager, "s1",
                   "I want admission",
                   "cancel",
                   "FSc")
    assert replies[0]["pending_slot"] == "course"
    assert replies[1]["reply"] == CANCEL_REPLY
    assert replies[1]["pending_slot"] == ""
    # "FSc" on its own is no longer an admissions answer.
    assert replies[2]["intent"] == "fallback"


def test_manager_reset_forgets_session(manager):
    manager.handle("s1", "I want admission")
    manager.reset("s1")
    reply = manager.handle("s1", "FSc")
    assert reply["intent"] == "fallback"


# --- small talk and handoff --------------------------------------------------

def test_thanks_mid_flow_keeps_the_question(manager):
    replies = talk(manager, "s1",
                   "I want admission",
                   "thanks",
                   "FSc")
    assert replies[1]["intent"] == "thanks"
    assert replies[1]["pending_slot"] == "course"
    assert replies[2]["pending_slot"] == "class_level"


def test_greeting_answered_without_starting_a_flow(manager):
    reply = manager.handle("s1", "hello")
    assert reply["intent"] == "greeting"
    assert reply["pending_slot"] == ""


def test_human_handoff_flagged(manager):
    replies = talk(manager, "s1", "I want to talk to a human")
    assert replies[0]["intent"] == "human_handoff"
    assert replies[0]["handoff_requested"] is True
    assert replies[0]["pending_slot"] == ""


def test_goodbye_ends_the_flow(manager):
    replies = talk(manager, "s1",
                   "I want admission",
                   "Allah Hafiz")
    assert replies[1]["intent"] == "goodbye"
    assert replies[1]["pending_slot"] == ""


# --- fallback and sessions ---------------------------------------------------

def test_gibberish_falls_back(manager):
    reply = manager.handle("s1", "zxqv blorp flibberty")
    assert reply["intent"] == "fallback"
    assert reply["pending_slot"] == ""


def test_empty_message_falls_back(manager):
    reply = manager.handle("s1", "   ")
    assert reply["intent"] == "fallback"


def test_sessions_are_independent(manager):
    manager.handle("s1", "I want admission")
    reply = manager.handle("s2", "FSc")
    assert reply["intent"] == "fallback"
    state = manager.store.get_or_create("s1")
    assert state.pending_slot == "course"


def test_expired_session_forgets_the_flow(manager):
    manager.handle("s1", "I want admission")
    state = manager.store.get_or_create("s1")
    state.updated_at -= timedelta(hours=1)
    reply = manager.handle("s1", "FSc")
    assert reply["intent"] == "fallback"
    assert reply["pending_slot"] == ""


# --- real model integration --------------------------------------------------

@pytest.fixture(scope="module")
def real_manager():
    from chatbot.nlp.intent_model import IntentClassifier

    classifier = IntentClassifier.load()
    return DialogueManager(classifier=classifier, rng=random.Random(0))


def test_real_model_three_turn_admissions(real_manager):
    replies = talk(real_manager, "live",
                   "how can I get admission",
                   "FSc",
                   "9th class")
    assert replies[0]["intent"] == "admissions"
    assert replies[0]["pending_slot"] == "course"
    assert replies[1]["pending_slot"] == "class_level"
    assert replies[2]["pending_slot"] == ""
    assert "fsc" in replies[2]["reply"].lower()
    assert "9th" in replies[2]["reply"]


def test_real_model_follow_up_carry_over(real_manager):
    replies = talk(real_manager, "live2",
                   "meri fees kitni hai fsc ki",
                   "and ICS?")
    assert replies[0]["intent"] == "fees"
    assert replies[1]["intent"] == "fees"
    assert "ics" in replies[1]["reply"].lower()
