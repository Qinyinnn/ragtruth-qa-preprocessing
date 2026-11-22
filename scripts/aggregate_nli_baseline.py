"""Evaluate frozen NLI predictions on RAGTruth QA, without rerunning the model.

Run from the repository root:
    python scripts/aggregate_nli_baseline.py

This evaluates two fixed, heuristic 0.5-threshold aggregation rules. It does
NOT tune thresholds. It detects partial/incomplete validation predictions and
marks their metrics as preliminary. Never use the test set for method selection.
"""
import argparse
import json
from pathlib import Path

LABELS = ("evident_conflict", "baseless_info")
# The inference script saved probabilities in this order (per model.config.id2label).
ENTAILMENT, NEUTRAL, CONTRADICTION = 0, 1, 2


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Bad JSON on {path}:{line_no}; is the file still being written?") from exc


def classify_claim(probabilities, method, threshold):
    """Return (conflict, baseless) for ONE response sentence.

    A max-score rule selects the strongest entailment/contradiction score
    across ALL evidence sentences if it clears threshold. Otherwise neutral.
    Support-first prioritizes strong entailment, then strong contradiction.
    These are heuristics, not probability-calibrated multi-label classifiers.
    """
    if not probabilities:
        raise ValueError("No evidence–claim pairs for a response sentence")
    if any(len(p) != 3 for p in probabilities):
        raise ValueError("Expected entailment/neutral/contradiction triplets")
    max_e = max(p[ENTAILMENT] for p in probabilities)
    max_c = max(p[CONTRADICTION] for p in probabilities)
    if method == "max_score":
        if max_e >= threshold or max_c >= threshold:
            decision = "entailment" if max_e >= max_c else "contradiction"
        else:
            decision = "neutral"
    elif method == "support_first":
        if max_e >= threshold:
            decision = "entailment"
        elif max_c >= threshold:
            decision = "contradiction"
        else:
            decision = "neutral"
    else:
        raise ValueError(f"Unknown aggregation rule: {method}")
    return {"entailment": [0, 0], "contradiction": [1, 0], "neutral": [0, 1]}[decision]


def or_labels(labels):
    return [int(any(row[k] for row in labels)) for k in range(2)]


def evaluate(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Ground truth and predictions have different lengths")
    n = len(y_true)
    out = {"n": n, "by_label": {}}
    for k, name in enumerate(LABELS):
        tp = sum(t[k] == 1 and p[k] == 1 for t, p in zip(y_true, y_pred))
        fp = sum(t[k] == 0 and p[k] == 1 for t, p in zip(y_true, y_pred))
        fn = sum(t[k] == 1 and p[k] == 0 for t, p in zip(y_true, y_pred))
        tn = sum(t[k] == 0 and p[k] == 0 for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        out["by_label"][name] = {
            "support": tp + fn, "predicted_positives": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6),
        }
    out["macro_f1"] = round(sum(out["by_label"][k]["f1"] for k in LABELS) / 2, 6)
    out["exact_match_accuracy"] = round(sum(t == p for t, p in zip(y_true, y_pred)) / n, 6) if n else 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("results/nli_baseline/val_predictions.jsonl"))
    parser.add_argument("--source", type=Path, default=Path("data/processed/qa_val_sentences.jsonl"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/nli_baseline"))
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    source = {str(row["id"]): row for row in read_jsonl(args.source)}
    seen = set()
    rows = []
    for row in read_jsonl(args.predictions):
        rid = str(row["response_id"])
        if rid in seen:
            raise ValueError(f"Duplicate response ID in predictions: {rid}")
        if rid not in source:
            raise ValueError(f"Prediction response ID absent from source: {rid}")
        seen.add(rid)
        if row["true_response_labels"] != source[rid]["labels"]:
            raise ValueError(f"Response label mismatch for ID {rid}")
        if row["n_evidence_sentences"] != len(source[rid]["evidence_sentences"]):
            raise ValueError(f"Evidence count mismatch for ID {rid}")
        if len(row["claims"]) != len(source[rid]["response_sentences"]):
            raise ValueError(f"Claim count mismatch for ID {rid}")
        if not row["claims"]:
            raise ValueError(f"Empty response sentences for ID {rid}")
        for i, claim in enumerate(row["claims"]):
            if claim["sentence_idx"] != i:
                raise ValueError(f"Unexpected sentence index for ID {rid}")
            if len(claim["pair_probabilities"]) != row["n_evidence_sentences"]:
                raise ValueError(f"Pair count mismatch for ID {rid}, sentence {i}")
            src_sentence = source[rid]["response_sentences"][i]
            derived = [int(any(s.get("consistent_processed_label") == label for s in src_sentence["span_overlaps"])) for label in LABELS]
            if claim["true_sentence_labels"] != derived:
                raise ValueError(f"Sentence label mismatch for ID {rid}, sentence {i}")
        if or_labels([c["true_sentence_labels"] for c in row["claims"]]) != row["true_response_labels"]:
            raise ValueError(f"Ground-truth OR reconstruction failed for ID {rid}")
        rows.append(row)

    missing = len(source) - len(seen)
    if missing < 0:
        raise ValueError("More prediction records than source examples")
    print(f"Loaded {len(rows)}/{len(source)} validation responses; missing {missing}.")
    if missing:
        print("WARNING: INCOMPLETE predictions — metrics below are PRELIMINARY, not final validation results.")
    print(f"Fixed threshold: {args.threshold:g}; no threshold tuning performed.")

    report = {
        "split": "validation", "status": "complete" if missing == 0 else "partial_preliminary",
        "threshold": args.threshold, "expected_responses": len(source),
        "evaluated_responses": len(rows), "missing_responses": missing,
        "model_probabilities_order": ["entailment", "neutral", "contradiction"],
        "methods": {},
    }
    for method in ("max_score", "support_first"):
        true_s, pred_s, true_r, pred_r = [], [], [], []
        for row in rows:
            sentence_preds = []
            for c in row["claims"]:
                true_s.append(c["true_sentence_labels"])
                pred = classify_claim(c["pair_probabilities"], method, args.threshold)
                pred_s.append(pred)
                sentence_preds.append(pred)
            true_r.append(row["true_response_labels"])
            pred_r.append(or_labels(sentence_preds))
        report["methods"][method] = {
            "sentence_level": evaluate(true_s, pred_s),
            "response_level": evaluate(true_r, pred_r),
        }
        print(f"\n=== {method} ===")
        for level in ("sentence_level", "response_level"):
            m = report["methods"][method][level]
            print(f"{level}: n={m['n']} macro F1={m['macro_f1']:.4f}, exact match={m['exact_match_accuracy']:.4f}")
            for name in LABELS:
                x = m["by_label"][name]
                print(f"  {name:16s} P={x['precision']:.4f} R={x['recall']:.4f} F1={x['f1']:.4f} support={x['support']} predicted+={x['predicted_positives']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = "val_metrics.json" if missing == 0 else "partial_val_metrics.json"
    output = args.output_dir / filename
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {report['status']} results to {output}")


if __name__ == "__main__":
    main()
