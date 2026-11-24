import pytest

from chatbot.nlp import preprocess as pp


@pytest.fixture(scope="module", autouse=True)
def nltk_data():
    pp.ensure_nltk_data(download=True)


def test_tokenize_lowercases_and_drops_punctuation():
    assert pp.tokenize("Hello, WORLD!!") == ["hello", "world"]


def test_tokenize_handles_empty_and_non_text():
    assert pp.tokenize("") == []
    assert pp.tokenize("   ") == []
    assert pp.tokenize(None) == []


def test_normalize_collapses_repeated_letters_and_spaces():
    assert pp.normalize_text("Helloooo    there") == "helloo there"


def test_normalize_strips_urls_and_accents():
    assert pp.normalize_text("Café https://example.com now") == "cafe now"


def test_question_words_are_kept():
    tokens = pp.preprocess("What is the fee and when is the test?")
    assert "what" in tokens and "when" in tokens and "is" in tokens


def test_plain_stopwords_are_removed():
    tokens = pp.preprocess("the fee of the course")
    assert "the" not in tokens and "of" not in tokens


def test_misspellings_map_to_one_token():
    assert pp.preprocess("fess") == pp.preprocess("fees") == pp.preprocess("fee") == ["fee"]
    assert pp.preprocess("addmission") == ["admission"]


def test_roman_urdu_words_map_to_english():
    assert "fee" in pp.preprocess("fees kitni hai")
    assert "kitni" in pp.preprocess("fees kitni hai")
    assert pp.preprocess("dakhla") == ["admission"]
    assert pp.preprocess("shukriya") == ["thanks"]
    assert "test" in pp.preprocess("imtihan kab hai")


def test_lemmatizes_plural_and_verb_forms():
    assert "class" in pp.preprocess("classes")
    assert "teach" in pp.preprocess("who is teaching physics")


def test_chat_shorthand_is_expanded():
    assert pp.preprocess("plz tell ur timings", remove_stopwords=False) == ["please", "tell", "your", "timing"]


def test_numbers_survive():
    assert "10" in pp.preprocess("fee for class 10")


def test_preprocess_to_string_joins_tokens():
    assert pp.preprocess_to_string("Thanks a lot!") == "thanks lot"


def test_works_without_nltk_data(monkeypatch):
    monkeypatch.setattr(pp, "_has", lambda resource: False)
    monkeypatch.setitem(pp._ready, "tokenizer", None)
    monkeypatch.setitem(pp._ready, "lemmatizer", None)
    monkeypatch.setitem(pp._ready, "stopwords", None)
    assert pp.preprocess("Teaching timings?") == ["teaching", "timing"]
    monkeypatch.setitem(pp._ready, "tokenizer", None)
    monkeypatch.setitem(pp._ready, "lemmatizer", None)
    monkeypatch.setitem(pp._ready, "stopwords", None)


def test_ensure_nltk_data_survives_failed_download(monkeypatch):
    monkeypatch.setattr(pp, "_has", lambda resource: False)

    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(pp.nltk, "download", boom)
    missing = pp.ensure_nltk_data(download=True)
    assert set(missing) == set(pp.NLTK_PACKAGES)
