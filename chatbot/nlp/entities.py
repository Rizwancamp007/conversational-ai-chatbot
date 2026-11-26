"""Entity extraction for incoming chat messages.

Pulls the details the bot needs out of a message: the course or subject,
class level, day/date/time, phone, email, student name and amounts. It is
rule based (gazetteers + regex) with NLTK named-entity chunking for names,
so it works without a trained model and stays easy to extend for a new
domain — add the words to the gazetteer below.

Every entity is a dict with the matched text and its span in the original
message, plus a confidence score:

    {"type": "subject", "value": "physics", "text": "Physics",
     "start": 10, "end": 17, "confidence": 0.9}

Values are normalised where that helps the rest of the bot: "kal" and
"tomorrow" become an ISO date, "10 baje" becomes "10:00", "0300-1234567"
becomes "+923001234567", "Rs 5,000" becomes 5000.
"""

import re
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Gazetteers: phrase (as written by students) -> canonical value.
# Longer phrases are matched first, so "computer science" wins over
# "computer" and "fsc pre-medical" wins over "fsc".
# ---------------------------------------------------------------------------

COURSES = {
    "fsc pre-medical": "fsc pre-medical",
    "fsc pre medical": "fsc pre-medical",
    "pre-medical": "fsc pre-medical",
    "pre medical": "fsc pre-medical",
    "fsc pre-engineering": "fsc pre-engineering",
    "fsc pre engineering": "fsc pre-engineering",
    "pre-engineering": "fsc pre-engineering",
    "pre engineering": "fsc pre-engineering",
    "fsc": "fsc",
    "ics": "ics",
    "icom": "icom",
    "fa": "fa",
    "dcom": "dcom",
}

SUBJECTS = {
    "computer science": "computer science",
    "computer": "computer science",
    "comp science": "computer science",
    "pakistan studies": "pakistan studies",
    "pak studies": "pakistan studies",
    "pak study": "pakistan studies",
    "mathematics": "mathematics",
    "maths": "mathematics",
    "math": "mathematics",
    "physics": "physics",
    "chemistry": "chemistry",
    "chem": "chemistry",
    "biology": "biology",
    "bio": "biology",
    "english": "english",
    "urdu": "urdu",
    "islamiat": "islamiat",
    "islamiyat": "islamiat",
    "history": "history",
    "geography": "geography",
    "economics": "economics",
    "statistics": "statistics",
    "accounting": "accounting",
    "business studies": "business studies",
    "business": "business studies",
}

WEEKDAYS = {
    "monday": "monday", "mon": "monday", "peer": "monday",
    "tuesday": "tuesday", "tue": "tuesday", "tues": "tuesday",
    "mangal": "tuesday",
    "wednesday": "wednesday", "wed": "wednesday", "budh": "wednesday",
    "thursday": "thursday", "thu": "thursday", "thur": "thursday",
    "thurs": "thursday", "jumerat": "thursday",
    "friday": "friday", "fri": "friday", "juma": "friday",
    "jumma": "friday",
    "saturday": "saturday", "sat": "saturday", "hafta": "saturday",
    "sunday": "sunday", "sun": "sunday", "itwaar": "sunday",
    "itwar": "sunday",
}

MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Relative day words -> offset in days from the reference date.
RELATIVE_DAYS = {
    "today": 0, "aaj": 0,
    "tomorrow": 1, "kal": 1,
    "day after tomorrow": 2, "parso": 2,
    "yesterday": -1,
}

# Roman Urdu time-of-day hints, used with "baje" ("5 baje" in the evening).
DAY_PART_HOUR = {
    "subah": 0, "morning": 0,
    "dopahar": 12, "afternoon": 12,
    "shaam": 12, "sham": 12, "evening": 12,
    "raat": 12, "night": 12,
}

# Words that can never be part of a name after "my name is" / "I am".
NAME_STOPWORDS = {
    "hai", "hoon", "hun", "from", "and", "the", "a", "an", "student",
    "here", "looking", "asking", "in", "of", "to", "for", "my", "is",
    "am", "i", "me", "sir", "madam", "please", "plz", "kia", "kya",
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?92[ -]?3\d{2}[ -]?\d{7}|03\d{2}[ -]?\d{7})")
_AMOUNT_RE = re.compile(
    r"(?:rs\.?|pkr)\s*(\d[\d,]*(?:\.\d+)?)"
    r"|(\d[\d,]*(?:\.\d+)?)\s*(?:rupees?|rs\.?)"
    r"|(\d[\d,]*(?:\.\d+)?)\s*k\b"
    r"|(\d+(?:\.\d+)?)\s*thousand\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2})\s*([ap])\.?m\.?"
    r"|(\d{1,2})\s*([ap])\.?m\.?"
    r"|(?:(subah|dopahar|shaam|sham|raat|morning|afternoon|evening|night)\s+)?"
    r"(\d{1,2})\s*baje",
    re.IGNORECASE,
)
_DATE_NUM_RE = re.compile(
    r"(\d{4})-(\d{1,2})-(\d{1,2})"
    r"|(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})"
    r"|(\d{1,2})[/-](\d{1,2})"
)
_CLASS_RE = re.compile(
    r"(?:class|grade)\s+(\d+)(?:st|nd|rd|th)?"
    r"|(\d+)(?:st|nd|rd|th)\s*(?:class|year)?"
    r"|(first|second)\s+year"
    r"|(1st|2nd)\s+year"
    r"|(matric|intermediate|inter)\b",
    re.IGNORECASE,
)
_NAME_CUE_RE = re.compile(
    r"(?:my name is|mera naam|mera name|i am|i'm|this is)\s+"
    r"([A-Za-z][A-Za-z' ]{1,40})",
    re.IGNORECASE,
)

_ENTITY_TYPES = (
    "course", "subject", "class_level", "student_name", "phone", "email",
    "date", "day", "time", "amount",
)


def _gazetteer_regex(mapping):
    phrases = sorted(mapping, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(p) for p in phrases) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


_COURSE_RE = _gazetteer_regex(COURSES)
_SUBJECT_RE = _gazetteer_regex(SUBJECTS)
_WEEKDAY_RE = _gazetteer_regex(WEEKDAYS)
_RELATIVE_RE = _gazetteer_regex(RELATIVE_DAYS)
_MONTH_RE = _gazetteer_regex(MONTHS)
_DATE_WORD_RE = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r")\b"
    r"|(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?",
    re.IGNORECASE,
)


def _entity(type_, value, text, start, end, confidence):
    return {
        "type": type_,
        "value": value,
        "text": text,
        "start": start,
        "end": end,
        "confidence": confidence,
    }


def _claimed(spans, start, end):
    return any(start < e and end > s for s, e in spans)


def _claim(spans, start, end):
    spans.append((start, end))


def _to_24h(hour, minute, meridiem):
    hour = int(hour)
    if meridiem in ("a", "am"):
        return f"{0 if hour == 12 else hour:02d}:{int(minute):02d}"
    if meridiem in ("p", "pm"):
        return f"{12 if hour == 12 else hour + 12:02d}:{int(minute):02d}"
    return f"{hour:02d}:{int(minute):02d}"


def _valid_date(year, month, day):
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _extract_email(text, spans):
    out = []
    for m in _EMAIL_RE.finditer(text):
        _claim(spans, m.start(), m.end())
        out.append(_entity("email", m.group(0).lower(), m.group(0),
                           m.start(), m.end(), 0.98))
    return out


def _extract_phone(text, spans):
    out = []
    for m in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if digits.startswith("92"):
            value = "+" + digits
        else:  # starts with 0: 03xx -> +92 3xx
            value = "+92" + digits[1:]
        if len(value) != 13:  # +92 + 10 digits
            continue
        _claim(spans, m.start(), m.end())
        out.append(_entity("phone", value, m.group(0),
                           m.start(), m.end(), 0.98))
    return out


def _extract_amount(text, spans):
    out = []
    for m in _AMOUNT_RE.finditer(text):
        if _claimed(spans, m.start(), m.end()):
            continue
        if m.group(1) is not None or m.group(2) is not None:
            raw = (m.group(1) or m.group(2)).replace(",", "")
            value = float(raw)
        elif m.group(3) is not None:
            value = float(m.group(3).replace(",", "")) * 1000
        else:
            value = float(m.group(4)) * 1000
        value = int(value) if value.is_integer() else value
        _claim(spans, m.start(), m.end())
        out.append(_entity("amount", value, m.group(0),
                           m.start(), m.end(), 0.9))
    return out


def _extract_time(text, spans):
    out = []
    for m in _TIME_RE.finditer(text):
        if _claimed(spans, m.start(), m.end()):
            continue
        if m.group(1) is not None:
            value = _to_24h(m.group(1), m.group(2), m.group(3))
        elif m.group(4) is not None:
            value = _to_24h(m.group(4), 0, m.group(5))
        else:
            hour = int(m.group(7))
            day_part = (m.group(6) or "").lower()
            if day_part and DAY_PART_HOUR[day_part] == 12 and hour < 12:
                hour += 12
            value = f"{hour:02d}:00"
        if not (0 <= int(value[:2]) <= 23):
            continue
        _claim(spans, m.start(), m.end())
        out.append(_entity("time", value, m.group(0),
                           m.start(), m.end(), 0.92))
    return out


def _extract_date(text, spans, reference_date):
    out = []
    reference = reference_date or date.today()
    for m in _RELATIVE_RE.finditer(text):
        if _claimed(spans, m.start(), m.end()):
            continue
        offset = RELATIVE_DAYS[m.group(0).lower()]
        _claim(spans, m.start(), m.end())
        out.append(_entity(
            "date", (reference + timedelta(days=offset)).isoformat(),
            m.group(0), m.start(), m.end(), 0.9))
    for m in _DATE_WORD_RE.finditer(text):
        if _claimed(spans, m.start(), m.end()):
            continue
        if m.group(1) is not None:
            day, month = int(m.group(1)), MONTHS[m.group(2).lower()]
        else:
            month, day = MONTHS[m.group(3).lower()], int(m.group(4))
        value = _valid_date(reference.year, month, day)
        if value is None:
            continue
        _claim(spans, m.start(), m.end())
        out.append(_entity("date", value, m.group(0),
                           m.start(), m.end(), 0.9))
    for m in _DATE_NUM_RE.finditer(text):
        if _claimed(spans, m.start(), m.end()):
            continue
        if m.group(1) is not None:
            value = _valid_date(int(m.group(1)), int(m.group(2)),
                                int(m.group(3)))
        else:
            day = int(m.group(4) or m.group(7))
            month = int(m.group(5) or m.group(8))
            year = int(m.group(6)) if m.group(6) else reference.year
            if year < 100:
                year += 2000
            value = _valid_date(year, month, day)
        if value is None:
            continue
        _claim(spans, m.start(), m.end())
        out.append(_entity("date", value, m.group(0),
                           m.start(), m.end(), 0.9))
    return out


def _extract_day(text, spans):
    out = []
    for m in _WEEKDAY_RE.finditer(text):
        if _claimed(spans, m.start(), m.end()):
            continue
        _claim(spans, m.start(), m.end())
        out.append(_entity("day", WEEKDAYS[m.group(0).lower()], m.group(0),
                           m.start(), m.end(), 0.95))
    return out


def _extract_class_level(text, spans):
    out = []
    for m in _CLASS_RE.finditer(text):
        if _claimed(spans, m.start(), m.end()):
            continue
        if m.group(1) is not None:
            value = f"{int(m.group(1))}th"
        elif m.group(2) is not None:
            number = int(m.group(2))
            if "year" in m.group(0).lower():
                ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(
                    number, f"{number}th")
                value = f"{ordinal} year"
            else:
                value = f"{number}th"
        elif m.group(3) is not None:
            value = "1st year" if m.group(3) == "first" else "2nd year"
        elif m.group(4) is not None:
            value = "1st year" if m.group(4) == "1st" else "2nd year"
        else:
            word = m.group(5).lower()
            value = "intermediate" if word in ("intermediate", "inter") \
                else "matric"
        _claim(spans, m.start(), m.end())
        out.append(_entity("class_level", value, m.group(0),
                           m.start(), m.end(), 0.9))
    return out


def _extract_gazetteer(text, spans, pattern, mapping, type_, confidence=0.9):
    out = []
    for m in pattern.finditer(text):
        if _claimed(spans, m.start(), m.end()):
            continue
        _claim(spans, m.start(), m.end())
        out.append(_entity(type_, mapping[m.group(0).lower()], m.group(0),
                           m.start(), m.end(), confidence))
    return out


def _clean_name(raw):
    """Take the words after a name cue and drop anything that is not a name."""
    words = []
    for word in raw.strip().split():
        word = word.strip(".,!?")
        if not word or word.lower() in NAME_STOPWORDS:
            break
        if not word.replace("'", "").isalpha():
            break
        words.append(word)
        if len(words) == 3:
            break
    if not words:
        return None
    return " ".join(w.capitalize() for w in words)


def _extract_name_by_cue(text, spans):
    out = []
    for m in _NAME_CUE_RE.finditer(text):
        name = _clean_name(m.group(1))
        if not name:
            continue
        start = m.start(1) + (len(m.group(1)) - len(m.group(1).lstrip()))
        end = start + len(name)
        if _claimed(spans, start, end):
            continue
        _claim(spans, start, end)
        out.append(_entity("student_name", name, text[start:end],
                           start, end, 0.9))
    return out


def _chunker_ready():
    try:
        import nltk

        for resource in ("chunkers/maxent_ne_chunker_tab",
                         "corpora/words",
                         "taggers/averaged_perceptron_tagger_eng"):
            nltk.data.find(resource)
        return True
    except (ImportError, LookupError):
        return False


def _extract_name_by_chunking(text, spans):
    """NLTK named-entity chunking, when its data is installed.

    Returns [] when the chunker data is missing, so the bot still works.
    """
    if not _chunker_ready():
        return []
    try:
        import nltk

        from nltk import ne_chunk, pos_tag, word_tokenize

        out = []
        for chunk in ne_chunk(pos_tag(word_tokenize(text))):
            if not hasattr(chunk, "label") or chunk.label() != "PERSON":
                continue
            name = " ".join(token for token, _ in chunk.leaves())
            if not name.replace(" ", "").isalpha():
                continue
            start = text.find(name)
            if start < 0:
                continue
            end = start + len(name)
            if _claimed(spans, start, end):
                continue
            words = name.split()
            if len(words) > 3 or any(
                    w.lower() in NAME_STOPWORDS or
                    w.lower() in SUBJECTS or w.lower() in COURSES
                    for w in words):
                continue
            _claim(spans, start, end)
            out.append(_entity("student_name", name, name,
                               start, end, 0.75))
        return out
    except Exception:
        return []


def extract_entities(text, reference_date=None):
    """Extract all entities from ``text``.

    ``reference_date`` (a ``datetime.date``, defaults to today) is what
    relative words like "tomorrow" / "kal" are counted from. Returns a list
    of entity dicts, sorted by where they appear in the message.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    spans = []
    entities = []
    # Order matters: the most specific patterns claim their spans first, so
    # a phone number is never also read as an amount, and "fsc pre-medical"
    # is never split into "fsc" plus something else.
    entities += _extract_email(text, spans)
    entities += _extract_phone(text, spans)
    entities += _extract_amount(text, spans)
    entities += _extract_time(text, spans)
    entities += _extract_date(text, spans, reference_date)
    entities += _extract_day(text, spans)
    entities += _extract_class_level(text, spans)
    entities += _extract_gazetteer(text, spans, _COURSE_RE, COURSES,
                                   "course")
    entities += _extract_gazetteer(text, spans, _SUBJECT_RE, SUBJECTS,
                                   "subject")
    entities += _extract_name_by_cue(text, spans)
    entities += _extract_name_by_chunking(text, spans)
    entities.sort(key=lambda e: e["start"])
    return entities


def extract_slots(text, reference_date=None):
    """First entity of each type, as a slot dict for the dialogue manager.

    Slot names match the dataset's ``required_slots``: course, subject,
    class_level, student_name, phone, email, date, day, time, amount.
    """
    slots = {}
    for entity in extract_entities(text, reference_date):
        slots.setdefault(entity["type"], entity["value"])
    return slots
