"""Response templates for the chatbot.

Reads the replies and follow-up questions out of ``data/intents.json`` and
fills in the details the conversation has collected so far. A response is
picked at random from the intent's list, so the bot does not repeat the
exact same sentence every time; pass a seeded ``random.Random`` when the
choice has to be repeatable (tests do this).
"""

import random
import string

from chatbot.dataset import DEFAULT_PATH, load_intents

_MISSING = object()


class _SafeSlots(dict):
    """Missing slots stay as ``{slot}`` instead of raising KeyError."""

    def __missing__(self, key):
        return "{" + key + "}"


def load_responses(path=None):
    """Return {tag: {"responses": [...], "slot_prompts": {...},
    "required_slots": [...]}} for every intent in the dataset."""
    data = load_intents(path)
    out = {}
    for intent in data["intents"]:
        out[intent["tag"]] = {
            "responses": list(intent.get("responses", [])),
            "slot_prompts": dict(intent.get("slot_prompts", {}) or {}),
            "required_slots": list(intent.get("required_slots", [])),
        }
    return out


def pick_response(tag, slots=None, responses=None, rng=None):
    """Pick one response template for ``tag`` and fill in ``slots``.

    Returns "" when the intent has no responses (should not happen — the
    dataset checker requires at least one).
    """
    responses = responses if responses is not None else load_responses()
    entry = responses.get(tag) or {}
    templates = entry.get("responses") or []
    if not templates:
        return ""
    rng = rng or random
    template = rng.choice(templates)
    return string.Formatter().vformat(
        template, (), _SafeSlots(slots or {}))


def slot_prompt(tag, slot, responses=None):
    """The follow-up question for a missing slot, or "" when there is none."""
    responses = responses if responses is not None else load_responses()
    entry = responses.get(tag) or {}
    return (entry.get("slot_prompts") or {}).get(slot, "")


def required_slots(tag, responses=None):
    """The slots an intent needs before it can answer."""
    responses = responses if responses is not None else load_responses()
    entry = responses.get(tag) or {}
    return list(entry.get("required_slots") or [])


def missing_slots(tag, slots, responses=None):
    """Required slots for ``tag`` that are not filled yet, in dataset order."""
    filled = slots or {}
    return [s for s in required_slots(tag, responses) if not filled.get(s)]
