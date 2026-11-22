"""Generate milestone metrics, figures, and error cases for RAGTruth QA.

Run from repository root, after validation NLI inference:
    python scripts/milestone_report.py

Optional response-level baseline prediction files (one JSONL record per QA response):
    python scripts/milestone_report.py \
      --model majority=results/majority/val_predictions.jsonl \
      --model length_logistic=results/length_logistic/val_predictions.jsonl

Each external JSONL line: {"response_id":"11868", "predicted_labels":[0,0]}
(use original response IDs; the source of truth is qa_val.jsonl).

The test set is never read. All reported metrics are validation metrics.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ("evident_conflict", "baseless_info")
BINS = ("short", "medium", "long")
METHODS = ("max_score", "support_first")


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSON in {path}, line {lineno}") from e


def labels_ok(value):
    return isinstance(value, list) and len(value) == 2 and all(type(v) is int and v in (0, 1) for v in value)


def load_unique(path, key):
    out = {}
    for r in read_jsonl(path):
        k = str(r[key])
        if k in out:
            raise ValueError(f"Duplicate ID {k} in {path}")
        out[k] = r
    return out


def classify_claim(probabilities, method, threshold):
    if not probabilities or any(len(p) != 3 for p in probabilities):
        raise ValueError("Bad/empty NLI probability list")
    e = max(p[0] for p in probabilities)
    c = max(p[2] for p in probabilities)
    if method == "support_first":
        if e >= threshold:
            return [0, 0]
        if c >= threshold:
            return [1, 0]
        return [0, 1]
    if method == "max_score":
        if e >= threshold or c >= threshold:
            return [0, 0] if e >= c else [1, 0]
        return [0, 1]
    raise ValueError(method)


def or_labels(rows):
    return [int(any(r[j] for r in rows)) for j in range(2)]


def scores(truth, pred):
    if len(truth) != len(pred):
        raise ValueError("Mismatched truth/prediction lengths")
    if not truth:
        return {"n": 0, "macro_f1": None, "weighted_f1": None, "micro_f1": None,
                "exact_match_accuracy": None, "by_label": {}}
    by = {}
    tot_tp = tot_fp = tot_fn = 0
    for k, label in enumerate(LABELS):
        tp = sum(t[k] == 1 and p[k] == 1 for t, p in zip(truth, pred))
        fp = sum(t[k] == 0 and p[k] == 1 for t, p in zip(truth, pred))
        fn = sum(t[k] == 1 and p[k] == 0 for t, p in zip(truth, pred))
        tn = len(truth) - tp - fp - fn
        tot_tp += tp; tot_fp += fp; tot_fn += fn
        by[label] = {"precision": tp / (tp + fp) if tp + fp else 0.0,
                     "recall": tp / (tp + fn) if tp + fn else 0.0,
                     "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                     "support": tp + fn, "predicted_positives": tp + fp,
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    support = sum(v["support"] for v in by.values())
    return {"n": len(truth), "by_label": by,
            "macro_f1": sum(v["f1"] for v in by.values()) / 2,
            "weighted_f1": sum(v["f1"] * v["support"] for v in by.values()) / support if support else 0.0,
            "micro_f1": 2 * tot_tp / (2 * tot_tp + tot_fp + tot_fn) if 2 * tot_tp + tot_fp + tot_fn else 0.0,
            "exact_match_accuracy": sum(t == p for t, p in zip(truth, pred)) / len(truth)}


def plot_by_length(model, groups, folder):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, constrained_layout=True)
    x = np.arange(3)
    for ax, metric in zip(axes, ("precision", "recall", "f1")):
        for j, name in enumerate(LABELS):
            values = [groups[b]["by_label"].get(name, {}).get(metric, np.nan) for b in BINS]
            values = [np.nan if x is None else x for x in values]
            ax.bar(x + (j - 0.5) * 0.32, values, width=0.32,
                   label=name.replace("_", " "))
        ax.set_title(metric.title())
        ax.set_xticks(x, [f"{b}\n(n={groups[b]['n']})" for b in BINS])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Score (0–1)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{model}: validation metrics by response length", fontsize=12)
    fig.savefig(folder / f"{model}_length_metrics.png", dpi=180)
    plt.close(fig)


def plot_bias(actual, baseline_rates, folder):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    x = np.arange(3)
    for i, (ax, label) in enumerate(zip(axes, LABELS)):
        y = [actual[b][label] for b in BINS]
        ax.plot(x, y, marker="o", linewidth=2, label="Actual positive rate")
        for name, rates in baseline_rates.items():
            z = [rates[b][label] for b in BINS]
            ax.plot(x, z, marker="s", linestyle="--", label=f"{name}: predicted positive rate")
        ax.set_title(label.replace("_", " "))
        ax.set_xticks(x, [f"{b}\n(n={actual[b]['n']})" for b in BINS])
        ax.set_ylim(0, min(1.0, max(0.15, max(y + [v for rr in baseline_rates.values() for v in [rr[b][label] for b in BINS]]) * 1.2)))
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Proportion of responses")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Validation: hallucination prevalence vs. answer length")
    fig.savefig(folder / "length_bias.png", dpi=180)
    plt.close(fig)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path("data/processed/qa_val_sentences.jsonl"))
    ap.add_argument("--full", type=Path, default=Path("data/processed/qa_val.jsonl"))
    ap.add_argument("--audit", type=Path, default=Path("data/processed/audit_report.json"))
    ap.add_argument("--nli", type=Path, default=Path("results/nli_baseline/val_predictions.jsonl"))
    ap.add_argument("--model", action="append", default=[], metavar="NAME=FILE", help="Other response-level baseline JSONL")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("results/milestone"))
    args = ap.parse_args()
    if not 0 <= args.threshold <= 1:
        ap.error("Threshold must lie in [0,1]")
    args.out.mkdir(parents=True, exist_ok=True)
    figs = args.out / "figures"; figs.mkdir(exist_ok=True)

    src = load_unique(args.source, "id")
    full = load_unique(args.full, "id")
    if src.keys() != full.keys():
        raise ValueError("Full-context and sentence-level validation response IDs differ")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    low, high = audit["qa_length_group_cutoffs_from_train"]
    if low > high:
        raise ValueError("Invalid response-length cutoffs")
    def bin_for(rid):
        n = full[rid]["response_tokens"]
        return "short" if n <= low else ("medium" if n <= high else "long")
    truth = {}
    for rid, r in full.items():
        if not labels_ok(r["labels"]) or r["labels"] != src[rid]["labels"]:
            raise ValueError(f"Incorrect source truth labels for {rid}")
        truth[rid] = r["labels"]

    raw_nli = load_unique(args.nli, "response_id")
    if set(raw_nli) != set(src):
        raise ValueError(f"NLI predictions incomplete: {len(raw_nli)}/{len(src)}. Full validation required for milestone.")
    models = {}
    nli_sentence = {}
    for method in METHODS:
        pred_r, true_s, pred_s = {}, [], []
        for rid, r in raw_nli.items():
            s = src[rid]
            if r["true_response_labels"] != truth[rid]:
                raise ValueError(f"Response truth mismatch for {rid}")
            if len(r["claims"]) != len(s["response_sentences"]):
                raise ValueError(f"Sentence count mismatch for {rid}")
            if r["n_evidence_sentences"] != len(s["evidence_sentences"]):
                raise ValueError(f"Evidence count mismatch for {rid}")
            sent_preds = []
            sent_truths = []
            for j, c in enumerate(r["claims"]):
                if c["sentence_idx"] != j or len(c["pair_probabilities"]) != r["n_evidence_sentences"]:
                    raise ValueError(f"Pair alignment mismatch for {rid}:{j}")
                label = [int(any(sp["consistent_processed_label"] == category for sp in s["response_sentences"][j]["span_overlaps"])) for category in LABELS]
                if label != c["true_sentence_labels"]:
                    raise ValueError(f"Sentence label mismatch for {rid}:{j}")
                prediction = classify_claim(c["pair_probabilities"], method, args.threshold)
                sent_truths.append(label); sent_preds.append(prediction)
                true_s.append(label); pred_s.append(prediction)
            if or_labels(sent_truths) != truth[rid]:
                raise ValueError(f"OR truth reconstruction failed for {rid}")
            pred_r[rid] = or_labels(sent_preds)
        name = f"nli_{method}"
        models[name] = pred_r
        nli_sentence[name] = scores(true_s, pred_s)

    for item in args.model:
        if "=" not in item:
            ap.error("--model format: name=path.jsonl")
        name, path = item.split("=", 1)
        if not name or name in models or "/" in name or "\\" in name or ".." in name:
            ap.error(f"Invalid/duplicate model name: {name}")
        rows = load_unique(Path(path), "response_id")
        if set(rows) != set(src):
            raise ValueError(f"External model {name}: response IDs must match entire validation set ({len(rows)}/{len(src)})")
        preds = {}
        for rid, row in rows.items():
            p = row.get("predicted_labels")
            if not labels_ok(p):
                raise ValueError(f"Invalid predicted_labels for {name} response {rid}: {p}")
            preds[rid] = p
        models[name] = preds

    ids = list(src)
    all_true = [truth[rid] for rid in ids]
    actual_rates = {}
    for b in BINS:
        b_ids = [rid for rid in ids if bin_for(rid) == b]
        actual_rates[b] = {"n": len(b_ids), **{label: (sum(truth[rid][k] for rid in b_ids) / len(b_ids) if b_ids else 0.0) for k, label in enumerate(LABELS)}}

    report = {"split": "validation", "threshold": args.threshold,
              "threshold_tuned": False, "length_definition": "standard_deberta response_tokens (from qa_val.jsonl)",
              "length_cutoffs_train_derived": {"short_max": low, "medium_max": high},
              "weighted_f1_definition": "Support-weighted average of two positive-class per-label F1 scores; multilabel samples can contribute to both supports.",
              "actual_positive_rate_by_length": actual_rates, "models": {}}
    err_rows, summary_rows = [], []
    pred_rates_by_model = {}
    for name, predictions in models.items():
        pred = [predictions[rid] for rid in ids]
        overall = scores(all_true, pred)
        per_length = {}
        p_rates = {}
        for b in BINS:
            b_ids = [rid for rid in ids if bin_for(rid) == b]
            per_length[b] = scores([truth[rid] for rid in b_ids], [predictions[rid] for rid in b_ids])
            p_rates[b] = {label: (sum(predictions[rid][k] for rid in b_ids) / len(b_ids) if b_ids else 0.0) for k, label in enumerate(LABELS)}
        pred_rates_by_model[name] = p_rates
        report["models"][name] = {"overall": overall, "by_length": per_length,
                                  "predicted_positive_rate_by_length": p_rates}
        if name in nli_sentence:
            report["models"][name]["sentence_level"] = nli_sentence[name]
        plot_by_length(name, per_length, figs)
        for rid in ids:
            t, p = truth[rid], predictions[rid]
            mismatches = [(LABELS[j], "FP" if p[j] else "FN") for j in range(2) if t[j] != p[j]]
            if mismatches or t == [1, 1]:
                for label, error in (mismatches or [("both_positive", "both_true")]):
                    err_rows.append({"model": name, "response_id": rid, "answer_length_bin": bin_for(rid),
                                     "answer_tokens": full[rid]["response_tokens"],
                                     "error_label": label, "error_type": error,
                                     "both_true": int(t == [1, 1]), "true_labels": str(t), "predicted_labels": str(p),
                                     "query": src[rid]["query"][:240], "response_excerpt": src[rid]["output"][:500],
                                     "evidence_excerpt": src[rid]["context"][:300]})
        summary_rows.append({"model": name, "n": overall["n"], "macro_f1": overall["macro_f1"],
                             "weighted_f1": overall["weighted_f1"], "micro_f1": overall["micro_f1"],
                             "exact_match": overall["exact_match_accuracy"],
                             **{f"{label}_{metric}": overall["by_label"][label][metric] for label in LABELS for metric in ("precision", "recall", "f1")}})

    write_csv(args.out / "metrics_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(args.out / "error_cases.csv", err_rows,
              ["model", "response_id", "answer_length_bin", "answer_tokens", "error_label", "error_type", "both_true", "true_labels", "predicted_labels", "query", "response_excerpt", "evidence_excerpt"])
    # Compare all models in a separate plot; prominently includes length-only logistic if supplied.
    plot_bias(actual_rates, pred_rates_by_model, figs)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation: {len(ids)} responses | train-derived token cutoffs: short <= {low}, medium <= {high}, long > {high}")
    for row in summary_rows:
        print(f"{row['model']:23s} macro-F1={row['macro_f1']:.4f} weighted-F1={row['weighted_f1']:.4f} "
              f"conflict F1={row['evident_conflict_f1']:.4f} baseless F1={row['baseless_info_f1']:.4f}")
    print(f"Saved metrics, figures, and reviewable error cases in {args.out}")
    if not any("logistic" in name for name in models):
        print("NOTE: length-only logistic predictions not supplied; length-bias plot currently shows NLI predictions only.")


if __name__ == "__main__":
    main()
