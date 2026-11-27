"""Multi-turn dialogue: slot filling, context carry-over, intent switching.

One message at a time, the manager:

1. finds the session's state (or starts a fresh one when it expired),
2. works out the intent and pulls out the details (slots),
3. fills in any question it was waiting on, switches topic when the user
   clearly asks something else, or carries the last topic forward when the
   user just adds a detail ("and for ICS?"),
4. asks the next follow-up question when a detail is still missing, or
   answers once everything is filled in.

The classifier and entity extractor are injected so tests can use a simple
stand-in; by default the real intent model and entity extractor are used.
"""

import random

from chatbot.dialogue.state import SessionStore
from chatbot.nlp.entities import extract_slots
from chatbot.nlp.preprocess import normalize_text, tokenize
from chatbot.responses import (
    load_responses,
    missing_slots,
    pick_response,
    slot_prompt,
)

# Intents that are answered on the spot and never disturb the flow the
# user is already in (a "thanks" mid-form does not lose the form).
SMALL_TALK = {"greeting", "thanks", "how_are_you", "about_bot"}

# Words that drop the current topic and start fresh.
CANCEL_WORDS = {
    "cancel", "stop", "reset", "restart", "start over", "start again",
    "forget it", "never mind", "nevermind", "new chat", "clear",
}

CANCEL_REPLY = ("No problem, let's start fresh. "
                "Ask me about admissions, courses, fees, timetable or tests.")

# Confidence at which a different intent counts as a new topic. Below it,
# the message is treated as part of the topic already in progress.
SWITCH_CONFIDENCE = 0.6

# Words that may surround a slot answer without making it a new question
# ("my name is Ahmed Khan", "I am in 9th class").
SLOT_ANSWER_FILLER = {
    "my", "name", "is", "mera", "naam", "am", "i", "class", "course",
    "ka", "ki", "hai", "for", "the", "in", "and", "yeh", "ye", "sir",
    "please", "plz", "to", "me", "a", "an", "of",
}


class DialogueManager:
    """Runs the conversation for every session."""

    def __init__(self, classifier=None, entity_extractor=None,
                 responses=None, store=None, rng=None):
        self.classifier = classifier  # lazy: real model on first use
        self.entity_extractor = entity_extractor or extract_slots
        self.responses = responses if responses is not None \
            else load_responses()
        self.store = store or SessionStore()
        self.rng = rng or random

    # -- setup ---------------------------------------------------------------

    def _get_classifier(self):
        if self.classifier is None:
            from chatbot.nlp.intent_model import IntentClassifier

            self.classifier = IntentClassifier.load()
        return self.classifier

    # -- helpers -------------------------------------------------------------

    def _is_cancel(self, text):
        clean = normalize_text(text).strip(" .!?,")
        return clean in CANCEL_WORDS

    def _classify(self, text):
        result = self._get_classifier().predict(text)
        return result["intent"], result["confidence"]

    def _slots_for(self, intent):
        entry = self.responses.get(intent) or {}
        return set(entry.get("required_slots") or [])

    def _fill(self, state, intent, found):
        for slot in self._slots_for(intent):
            if slot in found:
                state.slots[slot] = found[slot]

    def _is_slot_answer(self, text, intent, found):
        """True when the message is just the detail the bot asked for.

        The slot values and filler words are stripped out; what is left
        must be at most one word ("FSc", "9th class", "my name is Ahmed
        Khan"). A real question leaves its question words behind ("what
        is the timetable for ICS" leaves "what", "timetable").
        """
        value_words = set()
        for slot in self._slots_for(intent):
            if slot in found:
                value_words.update(
                    str(found[slot]).lower().replace("-", " ").split())
        rest = [w for w in tokenize(text)
                if w not in value_words and w not in SLOT_ANSWER_FILLER]
        return len(rest) <= 1

    def _is_new_topic(self, intent, confidence, active_intent):
        return (intent not in ("fallback", "") and active_intent
                and intent != active_intent
                and (confidence or 0) >= SWITCH_CONFIDENCE)

    def _switch(self, state, intent, found, confidence):
        state.reset_flow()
        state.active_intent = intent
        state.slots = {k: v for k, v in found.items()
                       if k in self._slots_for(intent)}
        return self._continue(state, confidence)

    def _answer(self, state, intent, slots, reply_text=None, confidence=None):
        reply = reply_text if reply_text is not None else pick_response(
            intent, slots, self.responses, self.rng)
        state.add_turn("bot", reply, intent)
        return {
            "session_id": state.session_id,
            "reply": reply,
            "intent": intent,
            "confidence": confidence,
            "slots": dict(slots),
            "pending_slot": state.pending_slot,
            "handoff_requested": state.handoff_requested,
        }

    # -- main entry point ----------------------------------------------------

    def handle(self, session_id, text):
        """Handle one user message. Returns the reply and what happened."""
        state = self.store.get_or_create(session_id)
        text = (text or "").strip()
        state.add_turn("user", text)

        if not text:
            return self._answer(state, "fallback", state.slots)

        if self._is_cancel(text):
            state.reset_flow()
            return self._answer(state, "", state.slots,
                                reply_text=CANCEL_REPLY)

        intent, confidence = self._classify(text)
        found = self.entity_extractor(text) or {}
        found = {k: v for k, v in found.items() if v not in ("", None)}

        # 1. The bot was waiting on a detail and this message gives it.
        #    A bare answer ("FSc", "my name is Ahmed Khan") fills the slot
        #    even when the classifier confidently guesses a different intent
        #    for it — short messages have no intent of their own. A full
        #    question that happens to name a slot ("what is the timetable
        #    for ICS") switches topic instead.
        if state.pending_slot and state.pending_slot in found:
            if self._is_slot_answer(text, state.active_intent, found):
                self._fill(state, state.active_intent, found)
                return self._continue(state, confidence)
            if self._is_new_topic(intent, confidence, state.active_intent):
                return self._switch(state, intent, found, confidence)
            self._fill(state, state.active_intent, found)
            return self._continue(state, confidence)

        # 2. Small talk is answered on the spot; the flow stays as it was.
        if intent in SMALL_TALK:
            return self._answer(state, intent, state.slots,
                                confidence=confidence)

        # 3. A confident new intent replaces the current topic.
        if self._is_new_topic(intent, confidence, state.active_intent):
            return self._switch(state, intent, found, confidence)

        # 4. Same topic, or a follow-up that only adds a detail
        #    ("and for ICS?") — keep the topic, update the details.
        if state.active_intent:
            self._fill(state, state.active_intent, found)
            return self._continue(state, confidence)

        # 5. Nothing in progress: a short follow-up that only adds a detail
        #    to the last topic ("and for ICS?") resumes it; anything else
        #    starts fresh. A longer message naming a new topic ("what
        #    courses do you offer in ICS") is a new question, not a follow-up.
        if state.last_intent:
            needed = self._slots_for(state.last_intent)
            new = {k: v for k, v in found.items() if k in needed}
            short = len(text.split()) <= 5
            if new and (intent in ("fallback", "", state.last_intent)
                        or (short and not self._slots_for(intent))):
                state.active_intent = state.last_intent
                state.slots = {k: v for k, v in state.last_slots.items()
                               if k in needed}
                state.slots.update(new)
                return self._continue(state, confidence)

        if intent in ("fallback", ""):
            return self._answer(state, "fallback", state.slots,
                                confidence=confidence)

        state.active_intent = intent
        state.slots = {k: v for k, v in found.items()
                       if k in self._slots_for(intent)}
        return self._continue(state, confidence)

    def _continue(self, state, confidence):
        """Ask the next follow-up, or answer now that everything is filled."""
        intent = state.active_intent
        if not intent:
            return self._answer(state, "fallback", state.slots,
                                confidence=confidence)

        if intent == "human_handoff":
            state.handoff_requested = True
            state.finish_flow()
            return self._answer(state, "human_handoff", state.slots,
                                confidence=confidence)

        if intent == "goodbye":
            state.finish_flow()
            return self._answer(state, "goodbye", state.slots,
                                confidence=confidence)

        missing = missing_slots(intent, state.slots, self.responses)
        if missing:
            state.pending_slot = missing[0]
            prompt = slot_prompt(intent, missing[0], self.responses)
            reply = prompt or pick_response(intent, state.slots,
                                            self.responses, self.rng)
            state.add_turn("bot", reply, intent)
            return {
                "session_id": state.session_id,
                "reply": reply,
                "intent": intent,
                "confidence": confidence,
                "slots": dict(state.slots),
                "pending_slot": state.pending_slot,
                "handoff_requested": state.handoff_requested,
            }

        state.pending_slot = ""
        reply_intent = intent
        state.finish_flow()
        return self._answer(state, reply_intent, state.last_slots,
                            confidence=confidence)

    def reset(self, session_id):
        """Forget a session entirely (the chat UI's reset button)."""
        self.store.reset(session_id)
