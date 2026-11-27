# Conversational AI Chatbot

An NLP chatbot for an academy helpdesk. It understands what a student or parent is asking (admissions, courses, fees, timetable, tests, results, online classes, contact), pulls out details like the course or class, keeps track of the conversation across turns, and asks a follow-up when something is missing.

Built with Python, NLTK, TensorFlow (Keras) and Flask. The training data is a plain JSON file, so the same bot can be retrained for another domain (a clinic, a shop) by swapping one file.

## Status

Work in progress. Done so far:

- Project layout, Flask app with `/health`
- Academy helpdesk dataset in `data/intents.json` (18 intents, English + common Roman Urdu spellings)
- Dataset checker
- Text preprocessing with NLTK (tokenise, lowercase, lemmatise, keep question words, fix common misspellings and Roman Urdu words)
- Intent classifier (Keras text model, trained on the dataset, with a confidence threshold and a fallback intent for anything outside the domain)
- Entity extraction (course/subject, class level, day/date/time, phone, email, student name, amount — gazetteers + regex + NLTK name chunking, with normalisation like "kal" → a date and "10 baje" → "10:00")
- Multi-turn dialogue: per-session state, slot filling with follow-up questions, context carry-over ("and for ICS?"), topic switching mid-conversation, cancel/reset, session expiry after 30 minutes, small talk that doesn't lose the current question, and human handoff

Intent model, the chat UI and the API are next.

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
python scripts/train.py          # trains the intent model, saves it under models/
python scripts/evaluate.py      # accuracy and per-intent scores on held-out data
python wsgi.py                  # http://127.0.0.1:5000/health
pytest
```

## Train the model

`python scripts/train.py` trains the intent classifier on `data/intents.json`
(80/20 stratified split, fixed seed so runs repeat) and saves the model,
label list and metrics under `models/` — that folder is gitignored, so the
trained files are never committed. `python scripts/evaluate.py` re-scores the
saved model on the same held-out split and prints accuracy, per-intent
precision/recall/F1 and the confusion matrix. Anything the model is unsure
about (confidence below `CONFIDENCE_THRESHOLD`, default 0.6) falls back to
the `fallback` intent.

## Entity extraction

`chatbot/nlp/entities.py` pulls the details out of a message:

```python
from chatbot.nlp.entities import extract_entities, extract_slots

extract_entities("FSc physics ki fee kal kitni hai, Rs 5000?")
# subject: physics, course: fsc, date: <tomorrow's ISO date>, amount: 5000
extract_slots("my name is Ahmed, 9th class, call 03001234567")
# {"student_name": "Ahmed", "class_level": "9th", "phone": "+923001234567"}
```

Each entity carries its span in the original message and a confidence
score. Values are normalised: "kal"/"tomorrow" become an ISO date (counted
from today, or from the `reference_date` you pass in), "10 baje" becomes
"10:00", phone numbers become "+92..." form, "Rs 5,000" becomes `5000`.
Names come from "my name is ..." / "mera naam ..." cues first, with NLTK
named-entity chunking as a fallback. The slot names match the dataset's
`required_slots`, so the dialogue manager can fill them straight from
these results.

## Multi-turn dialogue

`chatbot/dialogue/manager.py` runs one conversation per session id:

```python
from chatbot.dialogue.manager import DialogueManager

bot = DialogueManager()
bot.handle("session-1", "I want admission")
# "Which course or program are you interested in?"
bot.handle("session-1", "FSc")
# "Which class or level is the student in?"
bot.handle("session-1", "9th class")
# "Admissions are open for fsc (9th). You can register at the front desk..."
bot.handle("session-1", "and for ICS?")
# "Admissions are open for ics (9th). ..."
```

It asks a follow-up question for each missing slot (the questions come from
`slot_prompts` in the dataset), keeps the topic when the user just adds a
detail, switches topic when the user clearly asks something else, and
answers small talk ("thanks", "hello") without losing the question it was
waiting on. "cancel" / "start over" drops the current topic; "talk to a
human" flags the session for handoff. Sessions live in memory and expire
after 30 minutes of quiet.

A short answer ("FSc") is read as the answer to the question just asked,
even though the intent model on its own would guess a topic for it — the
manager checks the words left over after the slot value is taken out, so a
full question ("what is the timetable for ICS") still switches topic.

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
