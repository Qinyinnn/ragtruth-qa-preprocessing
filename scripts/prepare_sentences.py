"""Sentence-level preprocessing of RAGTruth-processed QA data for a zero-shot NLI baseline.

Reads the existing full-context files produced by ``scripts/preprocess.py``
(``qa_{train,val,test}.jsonl``) and writes, WITHOUT modifying them:

  qa_{train,val,test}_sentences.jsonl   one record per original QA example
  sentence_audit.json                   statistics + validation results

What this script does NOT do: model inference, fine-tuning, NLI aggregation,
truncation, dropping long examples, re-splitting, or creating sentence-level
ground-truth labels. Response sentences only record which ORIGINAL annotated
spans they overlap (with the span's original ``label_type``).

Usage:
  python scripts/prepare_sentences.py                 # build + validate
  python scripts/prepare_sentences.py --validate-only # re-check existing outputs
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
LABELS = ("evident_conflict", "baseless_info")  # order used by preprocess.py
NLI_CHECKPOINT = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"  # pinned HF Hub commit
SPAN_KEYS = {"start", "end", "text", "meta", "label_type", "implicit_true", "due_to_null"}

# Relationship between the ORIGINAL annotation category (span ``label_type``) and
# the PROCESSED binary label it is consistent with. This is NOT used to create
# labels; it is a hypothesis verified per example in ``check_label_consistency``
# (on the current data it holds for 100% of QA examples). Note that "Subtle
# Conflict" spans co-occur with ``evident_conflict = 1``.
LABEL_TYPE_TO_PROCESSED = {
    "Evident Conflict": "evident_conflict",
    "Subtle Conflict": "evident_conflict",
    "Evident Baseless Info": "baseless_info",
    "Subtle Baseless Info": "baseless_info",
}

# Integrity problems that make the output untrustworthy -> non-zero exit code.
FATAL_ISSUES = {
    "sentence_offset_mismatch", "sentence_order_or_overlap", "sentence_not_trimmed",
    "uncovered_non_whitespace", "parent_id_mismatch", "duplicate_id",
    "id_in_multiple_splits", "split_mismatch", "missing_or_extra_example",
    "label_changed", "original_text_changed", "original_spans_changed",
    "token_count_formula_mismatch",
}
LIST_MARKER = re.compile(r"(\(?(\d{1,3}|[A-Za-z]|[ivxIVX]{1,5})[.)]|[-*•–])")
TOKEN_BINS = [0, 64, 128, 256, 384, 512, 768, 1024, 2048, float("inf")]


# --------------------------------------------------------------------------- I/O
def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ segmentation
def build_segmenter(name):
    import spacy
    if name == "sentencizer":
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
    else:  # e.g. en_core_web_sm (must be installed separately)
        nlp = spacy.load(name, exclude=["ner", "lemmatizer", "textcat"])
    nlp.max_length = 10_000_000
    return nlp


def segment(text, nlp):
    """Return [(start, end)] sentence offsets into ``text``.

    Text is first split on newlines so a sentence never crosses a passage/line
    boundary, then each line is segmented by spaCy. Leading/trailing whitespace
    is trimmed, so ``text[start:end]`` is the sentence and only whitespace lies
    between consecutive sentences.
    """
    lines = [(m.start(), m.group()) for m in re.finditer(r"[^\n]+", text) if m.group().strip()]
    spans = []
    for (base, line), doc in zip(lines, nlp.pipe(l for _, l in lines)):
        line_spans = []
        for sent in doc.sents:
            start, end = base + sent.start_char, base + sent.end_char
            raw = text[start:end]
            if not raw.strip():
                continue
            start += len(raw) - len(raw.lstrip())
            end -= len(raw) - len(raw.rstrip())
            # Conservative repairs of known rule-based false splits (same line only):
            #  * a bare list marker ("1.", "b)", "-") is joined to the following text;
            #  * no whitespace between two pieces ("Worksheets.Add", "site.com") means
            #    the boundary is inside a token, not a real sentence break.
            if line_spans and (LIST_MARKER.fullmatch(text[line_spans[-1][0]:line_spans[-1][1]])
                               or line_spans[-1][1] == start):
                line_spans[-1] = (line_spans[-1][0], end)
            else:
                line_spans.append((start, end))
        spans.extend(line_spans)
    return spans


# ------------------------------------------------------------------ span mapping
def map_spans(spans, sentences):
    """Overlap of each original span with each response sentence (char level)."""
    per_sentence = defaultdict(list)
    span_map = []
    for si, sp in enumerate(spans):
        a, b = sp["start"], sp["end"]
        hits = []
        for k, (s, e) in enumerate(sentences):
            lo, hi = max(a, s), min(b, e)
            if lo < hi:
                hits.append(k)
                per_sentence[k].append({
                    "span_index": si,
                    "label_type": sp.get("label_type"),
                    "consistent_processed_label": LABEL_TYPE_TO_PROCESSED.get(sp.get("label_type")),
                    "overlap_start": lo,
                    "overlap_end": hi,
                    "overlap_chars": hi - lo,
                    "fraction_of_span": round((hi - lo) / max(b - a, 1), 4),
                    "fraction_of_sentence": round((hi - lo) / max(e - s, 1), 4),
                })
        span_map.append({"span_index": si, "label_type": sp.get("label_type"),
                         "sentence_indices": hits})
    return per_sentence, span_map


# -------------------------------------------------------------------- build step
def build_record(row, split, nlp):
    context, output = row["context"], row["output"]
    ev = segment(context, nlp)
    rs = segment(output, nlp)
    spans = row["original_spans"]
    # Map only spans whose offsets are valid against the original output.
    valid_idx = [i for i, sp in enumerate(spans) if span_offset_problem(sp, output) is None]
    per_sentence, span_map = map_spans([spans[i] for i in valid_idx], rs)
    for m in span_map:  # restore indices into the full original_spans list
        m["span_index"] = valid_idx[m["span_index"]]
    for overlaps in per_sentence.values():
        for o in overlaps:
            o["span_index"] = valid_idx[o["span_index"]]
    return {
        "id": row["id"],
        "source_group": row["source_group"],
        "original_split": row["original_split"],   # HF split: train / test
        "split": split,                            # project split: train / val / test
        "task_type": row["task_type"],
        "quality": row["quality"],
        "model": row["model"],
        "query": row["query"],
        "context": context,
        "output": output,
        "label_names": list(LABELS),
        "labels": row["labels"],                   # unchanged, response-level
        "response_labels": dict(zip(LABELS, row["labels"])),
        "original_spans": spans,                   # unchanged original annotations
        "evidence_sentences": [
            {"idx": k, "start": s, "end": e, "text": context[s:e]} for k, (s, e) in enumerate(ev)],
        "response_sentences": [
            {"idx": k, "parent_id": row["id"], "start": s, "end": e, "text": output[s:e],
             "span_overlaps": per_sentence.get(k, [])} for k, (s, e) in enumerate(rs)],
        "span_sentence_map": span_map,
    }


def add_token_lengths(records, tok, limit):
    """Per-sentence token counts (no special tokens) and per-example pair stats."""
    n_special = tok.num_special_tokens_to_add(pair=True)
    all_ctx_lens = tok([r["context"] for r in records], add_special_tokens=False)["input_ids"]
    pair_lengths = []
    full_ctx_pairs = []
    for r, ctx_ids in zip(records, all_ctx_lens):
        texts = [s["text"] for s in r["evidence_sentences"] + r["response_sentences"]]
        lens = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]] if texts else []
        ne = len(r["evidence_sentences"])
        for s, n in zip(r["evidence_sentences"] + r["response_sentences"], lens):
            s["n_tokens"] = n
        e = np.array(lens[:ne], dtype=np.int64)
        q = np.array(lens[ne:], dtype=np.int64)
        pairs = (e[:, None] + q[None, :] + n_special) if e.size and q.size else np.zeros((0, 0), np.int64)
        for k, s in enumerate(r["response_sentences"]):
            col = pairs[:, k] if pairs.size else np.array([], np.int64)
            s["max_pair_tokens"] = int(col.max()) if col.size else None
            s["n_pairs_over_limit"] = int((col > limit).sum())
            s["full_context_pair_tokens"] = len(ctx_ids) + s["n_tokens"] + n_special
            full_ctx_pairs.append(s["full_context_pair_tokens"])
        r["pair_token_stats"] = {
            "n_pairs": int(pairs.size),
            "max_pair_tokens": int(pairs.max()) if pairs.size else None,
            "n_pairs_over_limit": int((pairs > limit).sum()),
            "context_tokens": len(ctx_ids),
        }
        pair_lengths.append(pairs.ravel())
    flat = np.concatenate(pair_lengths) if pair_lengths else np.array([], np.int64)
    return flat, np.array(full_ctx_pairs, np.int64), n_special


def verify_pair_formula(records, tok, n_special, n_samples, seed):
    """Check e + r + special == len(tok(e, r)) on a random sample of real pairs."""
    rng = random.Random(seed)
    cands = [r for r in records if r["evidence_sentences"] and r["response_sentences"]]
    issues, checked = [], 0
    for _ in range(min(n_samples, sum(len(r["evidence_sentences"]) * len(r["response_sentences"]) for r in cands))):
        r = rng.choice(cands)
        e = rng.choice(r["evidence_sentences"])
        q = rng.choice(r["response_sentences"])
        actual = len(tok(e["text"], q["text"], truncation=False, add_special_tokens=True)["input_ids"])
        checked += 1
        if actual != e["n_tokens"] + q["n_tokens"] + n_special:
            issues.append({"type": "token_count_formula_mismatch", "id": r["id"],
                           "evidence_idx": e["idx"], "response_idx": q["idx"],
                           "formula": e["n_tokens"] + q["n_tokens"] + n_special, "actual": actual})
    return checked, issues


# ------------------------------------------------------------------- validation
def span_offset_problem(sp, output):
    if not isinstance(sp, dict) or not {"start", "end"} <= set(sp):
        return "span_malformed"
    a, b = sp["start"], sp["end"]
    if not (isinstance(a, int) and isinstance(b, int)) or not (0 <= a < b <= len(output)):
        return "span_out_of_bounds"
    if "text" in sp and output[a:b] != sp["text"]:
        return "span_text_mismatch"
    return None


def check_sentences(rec, field, text_key, issues):
    text = rec[text_key]
    sents = rec[field]
    if text.strip() and not sents:
        issues.append({"type": "empty_segmentation", "id": rec["id"], "field": field})
    covered = np.zeros(len(text), dtype=bool)
    prev_end = 0
    for k, s in enumerate(sents):
        a, b = s["start"], s["end"]
        if s["idx"] != k or a < prev_end or not (0 <= a < b <= len(text)):
            issues.append({"type": "sentence_order_or_overlap", "id": rec["id"], "field": field, "idx": k})
            continue
        if text[a:b] != s["text"]:
            issues.append({"type": "sentence_offset_mismatch", "id": rec["id"], "field": field, "idx": k})
        if s["text"] != s["text"].strip():
            issues.append({"type": "sentence_not_trimmed", "id": rec["id"], "field": field, "idx": k})
        if field == "response_sentences" and s.get("parent_id") != rec["id"]:
            issues.append({"type": "parent_id_mismatch", "id": rec["id"], "idx": k})
        covered[a:b] = True
        prev_end = b
    gaps = "".join(ch for ch, c in zip(text, covered) if not c)
    if gaps.strip():
        issues.append({"type": "uncovered_non_whitespace", "id": rec["id"], "field": field,
                       "chars": len(gaps.strip())})


def check_spans(rec, issues, notes):
    output = rec["output"]
    for si, sp in enumerate(rec["original_spans"]):
        problem = span_offset_problem(sp, output)
        if problem:
            issues.append({"type": problem, "id": rec["id"], "span_index": si})
            continue
        if set(sp) != SPAN_KEYS:
            issues.append({"type": "span_unexpected_keys", "id": rec["id"], "span_index": si,
                           "keys": sorted(sp)})
        if sp.get("label_type") not in LABEL_TYPE_TO_PROCESSED:
            issues.append({"type": "span_unknown_label_type", "id": rec["id"], "span_index": si,
                           "label_type": sp.get("label_type")})
        hits = next((m["sentence_indices"] for m in rec["span_sentence_map"]
                     if m["span_index"] == si), [])
        if not hits:
            issues.append({"type": "span_no_sentence_overlap", "id": rec["id"], "span_index": si})
        elif len(hits) > 1:
            notes["span_crosses_sentence_boundary"] += 1
        # Partial coverage: the span starts/ends inside a sentence (segmentation
        # boundary and annotation boundary disagree). Informational only.
        for k in hits:
            s = rec["response_sentences"][k]
            if sp["start"] > s["start"] or sp["end"] < s["end"]:
                notes["span_partially_covers_sentence"] += 1
                break


def check_label_consistency(rec, issues):
    """Do the processed binary labels agree with the original span categories?

    Reported, never 'fixed': the processed labels are kept as ground truth.
    """
    implied = {k: 0 for k in LABELS}
    for sp in rec["original_spans"]:
        lab = LABEL_TYPE_TO_PROCESSED.get(sp.get("label_type")) if isinstance(sp, dict) else None
        if lab:
            implied[lab] = 1
    for k, v in zip(LABELS, rec["labels"]):
        if implied[k] != v:
            issues.append({"type": "label_span_inconsistency", "id": rec["id"], "label": k,
                           "processed_label": v, "span_implied": implied[k]})


def validate(out_records, src_records, issues, notes):
    """Compare written sentence files against the untouched source files."""
    seen_split = {}
    for split in SPLITS:
        src = {r["id"]: r for r in src_records[split]}
        out = out_records[split]
        ids = [r["id"] for r in out]
        for i, n in Counter(ids).items():
            if n > 1:
                issues.append({"type": "duplicate_id", "id": i, "split": split})
        if set(ids) != set(src) or len(ids) != len(src):
            issues.append({"type": "missing_or_extra_example", "split": split,
                           "missing": len(set(src) - set(ids)), "extra": len(set(ids) - set(src))})
        for rec in out:
            if rec["id"] in seen_split and seen_split[rec["id"]] != split:
                issues.append({"type": "id_in_multiple_splits", "id": rec["id"]})
            seen_split[rec["id"]] = split
            s = src.get(rec["id"])
            if s is None:
                continue
            if rec["split"] != split or rec["original_split"] != s["original_split"]:
                issues.append({"type": "split_mismatch", "id": rec["id"], "split": split})
            if rec["labels"] != s["labels"] or list(rec["response_labels"].values()) != s["labels"]:
                issues.append({"type": "label_changed", "id": rec["id"]})
            if rec["output"] != s["output"] or rec["context"] != s["context"]:
                issues.append({"type": "original_text_changed", "id": rec["id"]})
            if rec["original_spans"] != s["original_spans"]:
                issues.append({"type": "original_spans_changed", "id": rec["id"]})
            check_sentences(rec, "evidence_sentences", "context", issues)
            check_sentences(rec, "response_sentences", "output", issues)
            check_spans(rec, issues, notes)
            check_label_consistency(rec, issues)


# ------------------------------------------------------------------------ audit
def dist(values):
    v = np.asarray(values, dtype=np.int64)
    if v.size == 0:
        return {"count": 0}
    return {"count": int(v.size), "mean": round(float(v.mean()), 2), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "max": int(v.max()), "min": int(v.min())}


def histogram(values):
    counts, _ = np.histogram(np.asarray(values), bins=TOKEN_BINS)
    labels = [f"{int(a)}-{int(b) - 1}" if b != float("inf") else f">={int(a)}"
              for a, b in zip(TOKEN_BINS[:-1], TOKEN_BINS[1:])]
    return dict(zip(labels, map(int, counts)))


def split_summary(records, pairs, full_ctx, limit):
    ev = [len(r["evidence_sentences"]) for r in records]
    rs = [len(r["response_sentences"]) for r in records]
    sent_tok_e = [s["n_tokens"] for r in records for s in r["evidence_sentences"]]
    sent_tok_r = [s["n_tokens"] for r in records for s in r["response_sentences"]]
    resp_sents = [s for r in records for s in r["response_sentences"]]
    span_types = Counter(sp["label_type"] for r in records for sp in r["original_spans"])
    over_by_type = Counter()
    for r in records:
        for m in r["span_sentence_map"]:
            if len(m["sentence_indices"]) > 1:
                over_by_type[m["label_type"]] += 1
    return {
        "examples": len(records),
        "examples_by_quality": dict(Counter(r["quality"] for r in records)),
        "response_label_positive_counts": {k: int(sum(r["labels"][i] for r in records))
                                           for i, k in enumerate(LABELS)},
        "evidence_sentences_per_example": dist(ev),
        "response_sentences_per_example": dist(rs),
        "evidence_sentence_tokens": dist(sent_tok_e),
        "response_sentence_tokens": dist(sent_tok_r),
        "evidence_x_response_pairs": {
            "description": "every (evidence sentence, response sentence) pair; "
                           "length = len(tok(evidence, response)) incl. special tokens",
            "distribution": dist(pairs),
            "histogram": histogram(pairs),
            "n_over_limit": int((pairs > limit).sum()),
            "pct_over_limit": round(100 * float((pairs > limit).mean()), 4) if pairs.size else 0.0,
            "examples_with_any_pair_over_limit": sum(r["pair_token_stats"]["n_pairs_over_limit"] > 0
                                                     for r in records),
        },
        "full_context_x_response_sentence_pairs": {
            "description": "reference only: (full context, one response sentence); shows that "
                           "sentence-splitting the response alone does not remove truncation",
            "distribution": dist(full_ctx),
            "n_over_limit": int((full_ctx > limit).sum()),
            "pct_over_limit": round(100 * float((full_ctx > limit).mean()), 4) if full_ctx.size else 0.0,
        },
        "single_sentences_over_limit": {
            "evidence": int(sum(n + 2 > limit for n in sent_tok_e)),
            "response": int(sum(n + 2 > limit for n in sent_tok_r)),
        },
        "span_mapping": {
            "original_spans": sum(span_types.values()),
            "spans_by_label_type": dict(span_types),
            "spans_overlapping_multiple_sentences": sum(over_by_type.values()),
            "multi_sentence_spans_by_label_type": dict(over_by_type),
            "response_sentences_total": len(resp_sents),
            "response_sentences_with_any_span_overlap": sum(bool(s["span_overlaps"]) for s in resp_sents),
            "response_sentences_in_positive_examples_without_overlap": sum(
                not s["span_overlaps"] for r in records if any(r["labels"]) for s in r["response_sentences"]),
        },
    }


def label_type_cooccurrence(records):
    """P(processed label = 1 | example has >=1 span of this label_type)."""
    table = defaultdict(lambda: {"examples": 0, **{k: 0 for k in LABELS}})
    for split_recs in records.values():
        for r in split_recs:
            for lt in {sp.get("label_type") for sp in r["original_spans"]}:
                table[lt]["examples"] += 1
                for i, k in enumerate(LABELS):
                    table[lt][k] += r["labels"][i]
    return dict(table)


def environment_versions():
    """Library versions that affect segmentation / token counts."""
    import platform
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    for pkg in ("spacy", "transformers", "tokenizers", "sentencepiece", "numpy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/processed", help="directory with qa_{split}.jsonl")
    ap.add_argument("--output-dir", default=None, help="defaults to --data-dir")
    ap.add_argument("--segmenter", default="sentencizer",
                    help="'sentencizer' (spaCy rule-based, default) or an installed spaCy model name")
    ap.add_argument("--tokenizer", default=NLI_CHECKPOINT)
    ap.add_argument("--tokenizer-revision", default=NLI_REVISION,
                    help="HF Hub commit of the tokenizer ('main' = latest, not reproducible)")
    ap.add_argument("--limit", type=int, default=512, help="token audit limit (max_position_embeddings)")
    ap.add_argument("--verify-samples", type=int, default=3000,
                    help="random pairs re-tokenized jointly to verify the pair-length formula")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validate-only", action="store_true",
                    help="skip building; validate existing *_sentences.jsonl against the source files")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir or args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_paths = {s: data_dir / f"qa_{s}.jsonl" for s in SPLITS}
    out_paths = {s: out_dir / f"qa_{s}_sentences.jsonl" for s in SPLITS}
    protected = {p.resolve() for p in data_dir.glob("*.jsonl") if not p.name.endswith("_sentences.jsonl")}
    protected |= {(data_dir / "audit_report.json").resolve()}
    audit_path = out_dir / "sentence_audit.json"
    for p in list(out_paths.values()) + [audit_path]:
        if p.resolve() in protected:
            sys.exit(f"Refusing to overwrite existing processed file: {p}")

    src = {s: read_jsonl(p) for s, p in src_paths.items()}
    for s, rows in src.items():
        print(f"loaded {len(rows):>5} source examples from {src_paths[s]}")

    from transformers import AutoTokenizer
    revision = args.tokenizer_revision if args.tokenizer == NLI_CHECKPOINT else None
    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=revision)

    if args.validate_only:
        out = {s: read_jsonl(p) for s, p in out_paths.items()}
        flat = {}
        n_special = tok.num_special_tokens_to_add(pair=True)
        for s, recs in out.items():
            pairs = [s_e["n_tokens"] + s_r["n_tokens"] + n_special for r in recs
                     for s_e in r["evidence_sentences"] for s_r in r["response_sentences"]]
            full = [s_r["full_context_pair_tokens"] for r in recs for s_r in r["response_sentences"]]
            flat[s] = (np.array(pairs, np.int64), np.array(full, np.int64))
    else:
        nlp = build_segmenter(args.segmenter)
        out, flat = {}, {}
        n_special = tok.num_special_tokens_to_add(pair=True)
        for s in SPLITS:
            recs = [build_record(r, s, nlp) for r in src[s]]
            pairs, full, n_special = add_token_lengths(recs, tok, args.limit)
            write_jsonl(out_paths[s], recs)
            flat[s] = (pairs, full)
            print(f"wrote {len(recs):>5} records -> {out_paths[s]}")
        # Validate what was actually written, not the in-memory objects.
        out = {s: read_jsonl(p) for s, p in out_paths.items()}

    issues, notes = [], Counter()
    validate(out, src, issues, notes)
    checked, formula_issues = verify_pair_formula(
        [r for recs in out.values() for r in recs], tok, n_special, args.verify_samples, args.seed)
    issues += formula_issues

    issue_counts = Counter(i["type"] for i in issues)
    fatal = {k: v for k, v in issue_counts.items() if k in FATAL_ISSUES}
    all_pairs = np.concatenate([flat[s][0] for s in SPLITS])
    report = {
        "source_files": {s: str(p) for s, p in src_paths.items()},
        "output_files": {s: str(p) for s, p in out_paths.items()},
        "tokenizer": args.tokenizer,
        "tokenizer_revision": revision,
        "environment": environment_versions(),
        "segmenter": f"newline pre-split + spaCy {args.segmenter}",
        "token_limit": args.limit,
        "special_tokens_per_pair": n_special,
        "pair_length_formula": "n_tokens(evidence) + n_tokens(response) + special_tokens_per_pair",
        "pair_length_formula_verified_on_random_pairs": checked,
        "seed": args.seed,
        "examples_processed": {s: len(out[s]) for s in SPLITS},
        "annotation_schema": {
            "offset_convention": "0-based, end-exclusive character offsets into `output`",
            "span_keys": sorted(SPAN_KEYS),
            "label_type_to_consistent_processed_label": LABEL_TYPE_TO_PROCESSED,
            "label_type_cooccurrence_with_processed_labels": label_type_cooccurrence(out),
        },
        "splits": {s: split_summary(out[s], flat[s][0], flat[s][1], args.limit) for s in SPLITS},
        "all_splits_pairs_over_limit": {
            "n_pairs": int(all_pairs.size), "n_over_limit": int((all_pairs > args.limit).sum()),
            "pct_over_limit": round(100 * float((all_pairs > args.limit).mean()), 4)},
        "validation": {
            "passed": not fatal,
            "issue_counts": dict(issue_counts),
            "fatal_issue_counts": fatal,
            "informational": dict(notes),
            "issue_examples": issues[:50],
        },
    }
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\naudit -> {audit_path}")
    for s in SPLITS:
        sp = report["splits"][s]
        p = sp["evidence_x_response_pairs"]
        print(f"{s:>5}: {sp['examples']} examples | evidence sents mean "
              f"{sp['evidence_sentences_per_example']['mean']} max {sp['evidence_sentences_per_example']['max']}"
              f" | response sents mean {sp['response_sentences_per_example']['mean']} max "
              f"{sp['response_sentences_per_example']['max']} | pairs {p['distribution']['count']} "
              f"max {p['distribution'].get('max')} tok, >{args.limit}: {p['n_over_limit']} ({p['pct_over_limit']}%)")
    print("validation issues:", dict(issue_counts) or "none")
    print("informational:", dict(notes))
    if fatal:
        print("FAILED integrity checks:", fatal)
        sys.exit(1)
    print("all integrity checks passed")


if __name__ == "__main__":
    main()
