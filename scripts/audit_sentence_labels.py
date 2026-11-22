
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/processed")
LABELS = ("evident_conflict", "baseless_info")


def get_sentence_labels(sentence):
    """A sentence is positive if any overlapping span has that label."""
    labels = [0, 0]

    for span in sentence["span_overlaps"]:
        label_type = span["consistent_processed_label"]

        if label_type not in LABELS:
            raise ValueError(f"Unknown span label: {span}")

        labels[LABELS.index(label_type)] = 1

    return tuple(labels)


for split in ("train", "val", "test"):
    path = DATA_DIR / f"qa_{split}_sentences.jsonl"

    counts = Counter()
    response_counts = Counter()
    mismatches = []
    total_sentences = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            sentence_labels = [
                get_sentence_labels(s)
                for s in record["response_sentences"]
            ]

            counts.update(sentence_labels)
            total_sentences += len(sentence_labels)

            # Reconstruct response-level labels using OR
            reconstructed = tuple(
                int(any(labels[k] for labels in sentence_labels))
                for k in range(2)
            )

            original = tuple(record["labels"])
            response_counts[original] += 1

            if reconstructed != original:
                mismatches.append(record["id"])

    print(f"\n{'=' * 45}")
    print(f"Split: {split}")
    print(f"Total sentences: {total_sentences}")
    print(f"Total responses: {sum(response_counts.values())}")

    print("\nSentence-level label distribution:")
    for label in ((0, 0), (1, 0), (0, 1), (1, 1)):
        n = counts[label]
        pct = 100 * n / total_sentences if total_sentences else 0
        print(f"  {label}: {n:6d} ({pct:.2f}%)")

    print("\nResponse-level label distribution:")
    for label in ((0, 0), (1, 0), (0, 1), (1, 1)):
        print(f"  {label}: {response_counts[label]}")

    print(f"\nOR reconstruction mismatches: {len(mismatches)}")
    if mismatches:
        print("Example mismatch IDs:", mismatches[:10])
