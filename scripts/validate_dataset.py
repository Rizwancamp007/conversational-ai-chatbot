"""Check data/intents.json (or another dataset) before training.

Usage: python scripts/validate_dataset.py [path/to/intents.json]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot.dataset import DEFAULT_PATH, load_intents, validate_dataset  # noqa: E402
from chatbot.nlp.preprocess import ensure_nltk_data, preprocess_to_string  # noqa: E402


def main(argv):
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    ensure_nltk_data()
    try:
        data = load_intents(path)
    except (OSError, ValueError) as exc:
        print(f"Could not read {path}: {exc}")
        return 1
    errors, warnings = validate_dataset(data, preprocess=preprocess_to_string)
    intents = data.get("intents", []) if isinstance(data, dict) else []
    total = sum(len(i.get("patterns", [])) for i in intents if isinstance(i, dict))
    print(f"{path}: {len(intents)} intents, {total} patterns")
    for w in warnings:
        print(f"  warning: {w}")
    for e in errors:
        print(f"  error: {e}")
    if errors:
        print(f"{len(errors)} error(s) found")
        return 1
    print("Dataset OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
