"""Evaluate a trained intent model on the held-out split.

Usage:
    python scripts/evaluate.py [--model models/intent_model.keras]
                               [--test-size 0.2] [--seed 42] [--output FILE]

Rebuilds the same stratified split train.py used (same seed and test size),
so the numbers come from patterns the model never trained on. Prints
accuracy, per-intent precision/recall/F1 and the confusion matrix.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from chatbot.dataset import DEFAULT_PATH as DATASET_PATH  # noqa: E402
from chatbot.nlp.intent_model import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SEED,
    IntentClassifier,
    evaluate_model,
    load_training_data,
    split_data,
)
from chatbot.nlp.preprocess import ensure_nltk_data  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the intent model.")
    parser.add_argument("--data", default=str(DATASET_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    ensure_nltk_data()
    classifier = IntentClassifier.load(args.model, args.labels)
    texts, tags, labels = load_training_data(args.data)
    _, test_texts, _, test_tags = split_data(
        texts, tags, test_size=args.test_size, seed=args.seed
    )
    metrics = evaluate_model(classifier.model, labels, test_texts, test_tags)
    print(f"held-out samples: {metrics['samples']}")
    print(f"accuracy: {metrics['accuracy']}")
    for tag, scores in metrics["per_intent"].items():
        print(f"  {tag:20s} P={scores['precision']:.2f} "
              f"R={scores['recall']:.2f} F1={scores['f1']:.2f} "
              f"(n={scores['support']})")

    probs = classifier.model.predict(
        tf.convert_to_tensor(test_texts, dtype=tf.string), verbose=0)
    predicted = [labels[i] for i in np.argmax(probs, axis=1)]
    matrix = confusion_matrix(test_tags, predicted, labels=labels)
    print("confusion matrix (rows = true, columns = predicted):")
    print("  " + " ".join(f"{t[:6]:>6s}" for t in labels))
    for tag, row in zip(labels, matrix):
        print(f"  {tag[:6]:>6s} " + " ".join(f"{v:6d}" for v in row))
    metrics["confusion_matrix"] = {
        "labels": labels,
        "matrix": [[int(v) for v in row] for row in matrix],
    }
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2),
                                     encoding="utf-8")
        print(f"saved report to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
