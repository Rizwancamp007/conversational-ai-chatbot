# Conversational AI Chatbot

An NLP chatbot for an academy helpdesk. It understands what a student or parent is asking (admissions, courses, fees, timetable, tests, results, online classes, contact), pulls out details like the course or class, keeps track of the conversation across turns, and asks a follow-up when something is missing.

Built with Python, NLTK, TensorFlow (Keras) and Flask. The training data is a plain JSON file, so the same bot can be retrained for another domain (a clinic, a shop) by swapping one file.

## Status

Work in progress. Done so far:

- Project layout, Flask app with `/health`
- Academy helpdesk dataset in `data/intents.json` (18 intents, English + common Roman Urdu spellings)
- Dataset checker
- Text preprocessing with NLTK (tokenise, lowercase, lemmatise, keep question words, fix common misspellings and Roman Urdu words)

Intent model, entity extraction, multi-turn dialogue and the chat UI are next.

## Project layout

```
app/                 Flask app
chatbot/nlp/         preprocessing (and the intent model / entity extraction)
chatbot/dialogue/    conversation state and flow
chatbot/dataset.py   loading and checking the dataset
data/intents.json    training data
scripts/             setup, dataset check, training
tests/               pytest
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/setup_nltk.py
cp .env.example .env            # fill in your own values
```

## Run

```bash
python scripts/validate_dataset.py
python wsgi.py                  # http://127.0.0.1:5000/health
pytest
```

## Dataset format

Each intent in `data/intents.json` has:

- `tag` – snake_case name
- `patterns` – example messages (at least 15)
- `responses` – replies; `{slot}` placeholders are filled from the conversation
- `required_slots` – details the bot must have before answering (e.g. `course`)
- `slot_prompts` – the question to ask when a slot is missing

Run `python scripts/validate_dataset.py` after editing it. It catches duplicate tags or patterns, the same pattern under two intents, unknown slots, missing follow-up questions and placeholders that aren't slots.

## Environment variables

See `.env.example`: `FLASK_SECRET_KEY`, `CLIENT_URL`, `ADMIN_TOKEN`, `CONFIDENCE_THRESHOLD`.
