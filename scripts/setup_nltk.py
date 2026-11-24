"""Download the NLTK data the bot uses. Run once after installing requirements."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot.nlp.preprocess import ensure_nltk_data  # noqa: E402

if __name__ == "__main__":
    missing = ensure_nltk_data(download=True, quiet=False)
    if missing:
        print("Could not download:", ", ".join(missing))
        print("The bot still runs, with simpler tokenising and no lemmatising.")
        sys.exit(1)
    print("NLTK data ready")
