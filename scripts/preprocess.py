"""Prepare RAGTruth for source-grounded, multi-label QA hallucination research.

Usage: python scripts/preprocess.py
Sentence-level files: python scripts/prepare_sentences.py (run afterwards)
Outputs are reproducible JSONL files and an audit report. No model fine-tuning.
"""
import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoConfig, AutoTokenizer

LABELS = ("evident_conflict", "baseless_info")
MODELS = {
    "standard_deberta": "microsoft/deberta-v3-base",
    "nli_deberta": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
}
# Pinned Hugging Face Hub commits used to generate the committed audit reports.
DATASET_ID = "wandb/RAGTruth-processed"
DATASET_REVISION = "eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b"
MODEL_REVISIONS = {
    "microsoft/deberta-v3-base": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
}


def load_any(value):
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def labels_from(row):
    raw = load_any(row["hallucination_labels_processed"])
    if not isinstance(raw, dict) or any(k not in raw for k in LABELS):
        raise ValueError(f"Unexpected label format for ID {row.get('id')}: {raw!r}")
    return [int(raw[k]) for k in LABELS]


def source_key(row):
    # A shared (question, evidence) pair is a proxy for the original source_id.
    # The processed HF dataset does not expose source_id directly.
    base = json.dumps([row.get("task_type"), row.get("query"), row.get("context")], ensure_ascii=False)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:20]


def encode_lengths(row, tokenizers):
    result = {}
    for short, tok in tokenizers.items():
        # Match the exact input formatting used by the future model.
        # A text pair includes tokenizer's special tokens.
        result[short] = len(tok(
            str(row["context"]), str(row["output"]),
            truncation=False, add_special_tokens=True
        )["input_ids"])
    return result


def build_record(row, split, tokenizers):
    original_spans = load_any(row.get("hallucination_labels"))
    if not isinstance(original_spans, list):
        original_spans = []
    rec = {
        "id": str(row["id"]),
        "source_group": source_key(row),
        "original_split": split,
        "task_type": str(row["task_type"]),
        "query": str(row["query"]),
        "context": str(row["context"]),
        "output": str(row["output"]),
        "quality": str(row.get("quality", "")),
        "model": str(row.get("model", "")),
        "labels": labels_from(row),
        "original_spans": original_spans,
        "response_tokens": len(tokenizers["standard_deberta"](
            str(row["output"]), add_special_tokens=False
        )["input_ids"]),
        "paired_tokens": encode_lengths(row, tokenizers),
    }
    return rec


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(rows, limit):
    if not rows:
        return {"count": 0}
    lens = np.array([r["paired_tokens"]["standard_deberta"] for r in rows])
    answers = np.array([r["response_tokens"] for r in rows])
    total = len(rows)
    b1 = sum(r["labels"][0] for r in rows)
    b2 = sum(r["labels"][1] for r in rows)
    # Length groups are based on response token count, using train-derived cutoffs.
    return {
        "count": total,
        "positive_rate_evident_conflict": round(b1 / total, 5),
        "positive_rate_baseless_info": round(b2 / total, 5),
        "positive_rate_either": round(sum(any(r["labels"]) for r in rows) / total, 5),
        "median_paired_tokens": float(np.median(lens)),
        "p90_paired_tokens": float(np.percentile(lens, 90)),
        "p95_paired_tokens": float(np.percentile(lens, 95)),
        "max_paired_tokens": int(lens.max()),
        "fraction_over_limit": round(float(np.mean(lens > limit)), 5),
        "median_response_tokens": float(np.median(answers)),
        "quality_counts": dict(Counter(r["quality"] for r in rows)),
        "num_source_groups": len({r["source_group"] for r in rows}),
    }


def length_groups(rows, cutoffs):
    groups = defaultdict(list)
    a, b = cutoffs
    for r in rows:
        n = r["response_tokens"]
        bucket = "short" if n <= a else ("medium" if n <= b else "long")
        groups[bucket].append(r)
    return {g: {
        "n": len(xs),
        "rate_conflict": sum(r["labels"][0] for r in xs) / len(xs),
        "rate_baseless": sum(r["labels"][1] for r in xs) / len(xs),
        "rate_either": sum(any(r["labels"]) for r in xs) / len(xs),
    } for g, xs in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=512,
                    help="Token audit limit, not a claim about all checkpoints")
    ap.add_argument("--dataset-revision", default=DATASET_REVISION,
                    help="HF Hub commit of the dataset ('main' = latest, not reproducible)")
    ap.add_argument("--sentences", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.sentences:
        # Removed: it wrote qa_*_sentences.jsonl in an unvalidated schema and would
        # overwrite the output of prepare_sentences.py.
        ap.error("--sentences was removed; run `python scripts/prepare_sentences.py` instead")
    if not 0 < args.val_fraction < 1:
        ap.error("--val-fraction must be between 0 and 1")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_ID, revision=args.dataset_revision)
    print("Available columns:", dataset["train"].column_names)
    tokenizers = {name: AutoTokenizer.from_pretrained(ckpt, revision=MODEL_REVISIONS[ckpt])
                  for name, ckpt in MODELS.items()}
    configs = {name: AutoConfig.from_pretrained(ckpt, revision=MODEL_REVISIONS[ckpt])
               for name, ckpt in MODELS.items()}
    limits = {name: {"tokenizer_model_max_length": str(tok.model_max_length),
                     "max_position_embeddings": getattr(configs[name], "max_position_embeddings", None)}
              for name, tok in tokenizers.items()}
    print("Checkpoint context limits (verify against docs):", limits)

    # Do not modify the official test set. Source grouping is applied to the
    # training set, not a random row-level split.
    train_all = [build_record(r, "train", tokenizers) for r in dataset["train"]]
    test_all = [build_record(r, "test", tokenizers) for r in dataset["test"]]
    all_groups = np.array([r["source_group"] for r in train_all])
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    tr_idx, val_idx = next(splitter.split(np.arange(len(train_all)), groups=all_groups))
    train = [train_all[int(i)] for i in tr_idx]
    val = [train_all[int(i)] for i in val_idx]
    test = test_all

    names = {"all_train": train, "all_val": val, "all_test": test,
             "qa_train": [r for r in train if r["task_type"] == "QA"],
             "qa_val": [r for r in val if r["task_type"] == "QA"],
             "qa_test": [r for r in test if r["task_type"] == "QA"]}

    sets = {k: {r["source_group"] for r in v} for k,v in names.items() if k.startswith("all_")}
    train_val_overlap = len(sets["all_train"] & sets["all_val"])
    train_test_overlap = len((sets["all_train"] | sets["all_val"]) & sets["all_test"])
    if train_val_overlap:
        raise RuntimeError("Source-group leakage between train and validation")
    if train_test_overlap:
        print("WARNING: original train/test splits share", train_test_overlap,
              "source groups; verify against original source_id before final evaluation")

    for name, rows in names.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    qa_train = names["qa_train"]
    if not qa_train:
        raise RuntimeError("No QA training examples found; inspect task_type values")
    q33, q67 = np.percentile([r["response_tokens"] for r in qa_train], [33.33, 66.67])
    cutoffs = [int(q33), int(q67)]

    report = {"source_dataset": "wandb/RAGTruth-processed",
              "model_checkpoints": MODELS, "model_config_lengths": limits,
              "token_audit_limit": args.limit, "seed": args.seed,
              "validation_group_fraction": args.val_fraction,
              "source_group_method": "sha256(task_type + query + context), NOT verified original source_id",
              "train_val_source_overlap": train_val_overlap,
              "original_train_test_source_overlap_proxy": train_test_overlap,
              "qa_length_group_cutoffs_from_train": cutoffs,
              "splits": {k: summarize(v, args.limit) for k, v in names.items()},
              "qa_length_groups": {k: length_groups(names[k], cutoffs)
                                   for k in ("qa_train", "qa_val", "qa_test")}}
    (out / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Output:", out.resolve())
    for name, info in report["splits"].items():
        print(name, "examples:", info["count"], "over", args.limit, "tokens:", info.get("fraction_over_limit"))
    print("QA length cutoffs:", cutoffs)
    print("QA test label rates by response length:", json.dumps(report["qa_length_groups"]["qa_test"], indent=2))


if __name__ == "__main__":
    main()
