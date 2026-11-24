"""Load and check the intents dataset."""

import json
import re
import string
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "intents.json"

KNOWN_SLOTS = {
    "course", "subject", "class_level", "student_name", "phone", "email",
    "date", "day", "time", "amount",
}
REQUIRED_INTENTS = {"greeting", "goodbye", "fallback"}
MIN_PATTERNS = 15
TAG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class DatasetError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(errors))


def load_intents(path=None):
    path = Path(path) if path else DEFAULT_PATH
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _placeholders(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def _norm(text):
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def validate_dataset(data, min_patterns=MIN_PATTERNS, preprocess=None):
    """Return (errors, warnings). Empty errors means the dataset is usable.

    `preprocess` (optional) maps a pattern to the string the model sees; it is
    used to catch the same pattern landing in two different intents.
    """
    errors, warnings = [], []
    if not isinstance(data, dict) or not isinstance(data.get("intents"), list):
        return ["dataset must be an object with an 'intents' list"], warnings
    intents = data["intents"]
    if not intents:
        return ["dataset has no intents"], warnings

    seen_tags = set()
    owner = {}
    for i, intent in enumerate(intents):
        where = f"intent #{i}"
        if not isinstance(intent, dict):
            errors.append(f"{where}: must be an object")
            continue
        tag = intent.get("tag")
        if not isinstance(tag, str) or not TAG_RE.match(tag):
            errors.append(f"{where}: tag must be snake_case text")
            continue
        where = f"'{tag}'"
        if tag in seen_tags:
            errors.append(f"{where}: duplicate tag")
        seen_tags.add(tag)

        patterns = intent.get("patterns")
        if not isinstance(patterns, list) or not all(isinstance(p, str) and p.strip() for p in patterns):
            errors.append(f"{where}: patterns must be a list of non-empty strings")
            patterns = []
        if len(patterns) < min_patterns:
            errors.append(f"{where}: needs at least {min_patterns} patterns, has {len(patterns)}")

        local = set()
        for p in patterns:
            key = _norm(p)
            if key in local:
                errors.append(f"{where}: duplicate pattern '{p}'")
            local.add(key)
            model_key = preprocess(p) if preprocess else key
            if model_key and model_key in owner and owner[model_key] != tag:
                errors.append(f"{where}: pattern '{p}' also used by '{owner[model_key]}'")
            elif model_key:
                owner.setdefault(model_key, tag)

        responses = intent.get("responses")
        if not isinstance(responses, list) or not responses or not all(isinstance(r, str) and r.strip() for r in responses):
            errors.append(f"{where}: needs at least one non-empty response")
            responses = []

        slots = intent.get("required_slots", [])
        if not isinstance(slots, list):
            errors.append(f"{where}: required_slots must be a list")
            slots = []
        unknown = [s for s in slots if s not in KNOWN_SLOTS]
        if unknown:
            errors.append(f"{where}: unknown slots {unknown}")
        prompts = intent.get("slot_prompts", {}) or {}
        for s in slots:
            if not str(prompts.get(s, "")).strip():
                errors.append(f"{where}: slot '{s}' has no follow-up prompt")

        for r in responses:
            extra = _placeholders(r) - set(slots)
            if extra:
                errors.append(f"{where}: response uses {sorted(extra)} which are not required slots")
        if len(responses) == 1:
            warnings.append(f"{where}: only one response, replies will feel repetitive")

    missing = REQUIRED_INTENTS - seen_tags
    if missing:
        errors.append(f"missing required intents: {sorted(missing)}")
    return errors, warnings


def load_and_validate(path=None, **kwargs):
    data = load_intents(path)
    errors, _ = validate_dataset(data, **kwargs)
    if errors:
        raise DatasetError(errors)
    return data
