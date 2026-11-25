"""Train the intent classifier and save it under models/.

Usage:
    python scripts/train.py [--epochs 80] [--test-size 0.2] [--seed 42]

The training set is an 80/20 stratified split of data/intents.json, so the
held-out numbers printed here are honest: the model never saw those
patterns while training. The model, label list and metrics are written to
models/ (gitignored, never committed).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot.dataset import DEFAULT_PATH as DATASET_PATH  # noqa: E402
from chatbot.nlp.intent_model import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SEED,
    IntentClassifier,
    evaluate_model,
    load_training_data,
    split_data,
    train,
)
from chatbot.nlp.preprocess import ensure_nltk_data  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the intent classifier.")
    parser.add_argument("--data", default=str(DATASET_PATH))
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    ensure_nltk_data()
    texts, tags, labels = load_training_data(args.data)
    train_texts, test_texts, train_tags, test_tags = split_data(
        texts, tags, test_size=args.test_size, seed=args.seed
    )
    print(f"dataset: {len(texts)} patterns, {len(labels)} intents")
    print(f"train: {len(train_texts)} patterns, held out: {len(test_texts)}")

    model = train(train_texts, train_tags, labels,
                  epochs=args.epochs, seed=args.seed)
    metrics = evaluate_model(model, labels, test_texts, test_tags)
    metrics["train_samples"] = len(train_texts)
    metrics["epochs"] = args.epochs
    metrics["seed"] = args.seed
    print(f"held-out accuracy: {metrics['accuracy']}")
    for tag, scores in metrics["per_intent"].items():
        print(f"  {tag:20s} P={scores['precision']:.2f} "
              f"R={scores['recall']:.2f} F1={scores['f1']:.2f} "
              f"(n={scores['support']})")

    classifier = IntentClassifier(model, labels)
    out_path = Path(args.out)
    labels_path = DEFAULT_LABELS_PATH
    classifier.save(out_path, labels_path)
    metrics_path = Path(DEFAULT_METRICS_PATH)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved model to {out_path}")
    print(f"saved labels to {labels_path}")
    print(f"saved metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
