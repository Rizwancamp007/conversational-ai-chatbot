"""Text preprocessing for incoming chat messages.

Tokenises, lowercases, normalises common spellings (including Roman Urdu)
and lemmatises with WordNet. Works without NLTK data too: it falls back to a
regex tokenizer and skips lemmatisation, so the app never crashes on a fresh
machine.
"""

import re
import unicodedata

import nltk

NLTK_PACKAGES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "stopwords": "corpora/stopwords",
    # named-entity chunking for student names (chatbot/nlp/entities.py)
    "maxent_ne_chunker_tab": "chunkers/maxent_ne_chunker_tab",
    "words": "corpora/words",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}

# Words we never drop: they decide what the user is asking.
QUESTION_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "can", "could", "is", "are", "do", "does", "did", "will", "should",
    "not", "no", "yes", "kya", "kab", "kahan", "kaise", "kitni", "kitna",
    "kitne", "kaun", "kyun", "nahi",
}

# Common misspellings and Roman Urdu words mapped to one canonical token.
TOKEN_MAP = {
    # fees
    "fess": "fee", "fees": "fee", "fe": "fee", "fis": "fee", "feez": "fee",
    "charges": "fee", "charge": "fee", "dues": "fee", "paisay": "fee",
    "paise": "fee", "paisa": "fee",
    # admission
    "admision": "admission", "addmission": "admission", "admisson": "admission",
    "dakhla": "admission", "dakhila": "admission", "enroll": "admission",
    "enrol": "admission", "enrollment": "admission", "enrolment": "admission",
    # timetable
    "timetable": "timetable", "timtable": "timetable", "schedual": "schedule",
    "shedule": "schedule", "schedul": "schedule", "timing": "timing",
    "timings": "timing", "waqt": "time", "time": "time",
    # tests / results
    "tst": "test", "tests": "test", "exam": "test", "exams": "test",
    "imtihan": "test", "paper": "test", "papers": "test", "quiz": "test",
    "result": "result", "rezult": "result", "reslt": "result", "marks": "result",
    "nateeja": "result", "natija": "result",
    # classes
    "clas": "class", "classs": "class", "classes": "class", "online": "online",
    "onlin": "online", "zoom": "online", "lecture": "lecture", "lectures": "lecture",
    "lecturer": "teacher",
    # teacher
    "teachr": "teacher", "techer": "teacher", "ustad": "teacher",
    "ustaad": "teacher", "sir": "teacher", "madam": "teacher",
    "faculty": "teacher",
    # course
    "cours": "course", "corse": "course", "courses": "course", "subject": "subject",
    "subjects": "subject", "mazmoon": "subject",
    # contact
    "contect": "contact", "cantact": "contact", "rabta": "contact",
    "number": "number", "nmbr": "number", "address": "address",
    "adress": "address", "pata": "address", "location": "address",
    # chat words
    "u": "you", "ur": "your", "r": "are", "pls": "please", "plz": "please",
    "plzz": "please", "thx": "thanks", "thnx": "thanks", "thanx": "thanks",
    "shukriya": "thanks", "shukria": "thanks", "meherbani": "thanks",
    "hy": "hi", "hii": "hi", "hiii": "hi", "helo": "hello", "hellow": "hello",
    "salam": "salam", "slam": "salam", "aoa": "salam", "assalam": "salam",
    "assalamualaikum": "salam", "asalam": "salam", "khuda": "khuda",
    "allah": "allah", "hafiz": "hafiz", "bye": "bye", "byee": "bye",
    "insan": "human", "banda": "human", "admi": "human", "agent": "human",
    "kb": "kab", "kia": "kya", "kiya": "kya", "kitny": "kitne", "kesay": "kaise",
    "kese": "kaise", "kaisay": "kaise",
}

CANONICAL = set(TOKEN_MAP.values())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_ready = {"tokenizer": None, "lemmatizer": None, "stopwords": None}


def _has(resource):
    # corpora such as wordnet ship as a zip and are only found by that name
    for name in (resource, resource + ".zip"):
        try:
            nltk.data.find(name)
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_data(download=True, quiet=True):
    """Make sure the NLTK data we use is present. Returns the missing names.

    Downloads quietly when allowed; a failed download (no internet) is not an
    error, preprocessing just runs in its simpler mode.
    """
    missing = []
    for name, resource in NLTK_PACKAGES.items():
        if _has(resource):
            continue
        if download:
            try:
                nltk.download(name, quiet=quiet)
            except Exception:  # network errors, permissions, etc.
                pass
        if not _has(resource):
            missing.append(name)
    _ready["tokenizer"] = None
    _ready["lemmatizer"] = None
    _ready["stopwords"] = None
    return missing


def _tokenizer():
    if _ready["tokenizer"] is None:
        if _has("tokenizers/punkt_tab") or _has("tokenizers/punkt"):
            from nltk.tokenize import word_tokenize

            def tok(text):
                try:
                    return word_tokenize(text)
                except LookupError:
                    return _TOKEN_RE.findall(text)

            _ready["tokenizer"] = tok
        else:
            _ready["tokenizer"] = _TOKEN_RE.findall
    return _ready["tokenizer"]


def _lemmatizer():
    if _ready["lemmatizer"] is None:
        if _has("corpora/wordnet"):
            from nltk.stem import WordNetLemmatizer

            lem = WordNetLemmatizer()

            def run(word):
                try:
                    noun = lem.lemmatize(word)
                    return lem.lemmatize(noun, pos="v") if noun == word else noun
                except LookupError:
                    return word

            _ready["lemmatizer"] = run
        else:
            _ready["lemmatizer"] = lambda w: w
    return _ready["lemmatizer"]


def _stopwords():
    if _ready["stopwords"] is None:
        words = set()
        if _has("corpora/stopwords"):
            from nltk.corpus import stopwords

            words = set(stopwords.words("english"))
        _ready["stopwords"] = words - QUESTION_WORDS
    return _ready["stopwords"]


def normalize_text(text):
    """Lowercase, strip accents, collapse repeated letters and whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    # "helloooo" -> "helloo", "pleaseee" -> "pleasee"
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    """Split a message into lowercase word tokens (punctuation removed)."""
    text = normalize_text(text)
    if not text:
        return []
    tokens = []
    for tok in _tokenizer()(text):
        tokens.extend(_TOKEN_RE.findall(tok))
    return tokens


def map_token(token):
    """Map a misspelling or Roman Urdu word to its canonical form."""
    if token in TOKEN_MAP:
        return TOKEN_MAP[token]
    # "helloo" left after collapsing -> try single letter
    squeezed = re.sub(r"(.)\1+", r"\1", token)
    return TOKEN_MAP.get(squeezed, token)


def preprocess(text, remove_stopwords=True, lemmatize=True):
    """Full pipeline: tokens ready for the intent model."""
    lem = _lemmatizer() if lemmatize else (lambda w: w)
    stops = _stopwords() if remove_stopwords else set()
    out = []
    for tok in tokenize(text):
        tok = map_token(tok)
        if tok in stops:
            continue
        if tok not in QUESTION_WORDS and tok not in CANONICAL and not tok.isdigit():
            tok = lem(tok)
            tok = map_token(tok)
        out.append(tok)
    return out


def preprocess_to_string(text, **kwargs):
    return " ".join(preprocess(text, **kwargs))
