# RAGTruth QA preprocessing

Reproducible data-preparation starter for COMS4705: **evident conflict** and **baseless information** detection.

## Key Findings & Preprocessing Decisions

We made the following preprocessing decisions based on our initial dataset analysis and TA feedback:

| Finding | Decision |
|---|---|
| ~30% of QA training examples exceed DeBERTa's 512-token input budget | Created sentence-level data for the zero-shot NLI baseline |
| Longer responses have higher hallucination-positive rates | Preserved response token lengths for the length-only Logistic Regression baseline |
| Evident conflicts are relatively rare (~7% of QA training examples) | Retained the original label distribution for analysis of class imbalance |
| Original annotations contain four hallucination types | Used the two processed binary labels; confirmed that subtle and evident variants are grouped together |
| Some hallucination spans cross sentence boundaries | Preserved original character offsets and recorded sentence overlaps without creating artificial labels |
| Multiple generated responses share the same source | Used source-grouped validation splitting to reduce leakage |

### Which files should teammates use?

**For majority-class and length-only Logistic Regression baselines:**
- `qa_train.jsonl`
- `qa_val.jsonl`
- `qa_test.jsonl`

Use the response-level labels and response token counts.

**For zero-shot NLI-DeBERTa:**
- `qa_train_sentences.jsonl`
- `qa_val_sentences.jsonl`
- `qa_test_sentences.jsonl`

Use the evidence and response sentence pairs. Response-level prediction aggregation still needs to be implemented.

**For fine-tuning DeBERTa:**
- `qa_train.jsonl`, `qa_val.jsonl`, `qa_test.jsonl` for QA-only training.
- `all_train.jsonl`, `all_val.jsonl` for cross-task training.

The fine-tuning pipeline must explicitly handle inputs exceeding the model's context limit.

**Important:** Sentence-level files are intended for NLI inference, not directly supervised sentence-level training. The original response-level labels remain unchanged.

## Setup

Requires Python 3.11 or later. The first run downloads the dataset and tokenizers (no PyTorch needed); later runs use the HF cache.

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt   # exact versions used for the committed audits
                                       # (or requirements.txt for compatible ranges)
python scripts/preprocess.py           # full-context files + audit_report.json  (~2 min)
python scripts/prepare_sentences.py    # sentence-level files + sentence_audit.json (~15 s)
python scripts/prepare_sentences.py --validate-only   # re-check existing outputs; exit 1 on failure
```

Generated `*.jsonl` files are **not tracked in git** (about 127 MB, see `.gitignore`). Every teammate regenerates them with the commands above. The two audit reports *are* tracked, so you can check your regenerated data against them (see [Reproducibility](#reproducibility)).

## Repository contents

| path | tracked | purpose |
|---|---|---|
| `scripts/preprocess.py` | yes | HF dataset → source-grouped train/val/test, full-context JSONL, `audit_report.json` |
| `scripts/prepare_sentences.py` | yes | QA JSONL → sentence-level JSONL, validation, `sentence_audit.json` |
| `requirements.txt`, `requirements-lock.txt` | yes | dependency ranges / exact pinned versions |
| `data/processed/audit_report.json`, `sentence_audit.json` | yes | reference statistics and validation results |
| `data/processed/*.jsonl` | **no** | generated data; regenerate locally |

## Reproducibility

- **Pinned inputs.** The dataset `wandb/RAGTruth-processed` is pinned at commit `eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b` (change it with `--dataset-revision`). The tokenizers are pinned too: `microsoft/deberta-v3-base` at `8ccc9b6f…` and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` at `6f5cf0a2…`. The full hashes are in the scripts.
- **Deterministic.** The validation split uses `GroupShuffleSplit(seed=42)`, and the pair-length verification sample uses `seed=42`. Running both scripts from scratch into a separate directory reproduced every `*.jsonl` file and `audit_report.json` **byte-for-byte**. `sentence_audit.json` was identical apart from the file paths it records.
- **Check your copy.** Compare your regenerated `audit_report.json` with the committed one (`git diff`), and confirm `prepare_sentences.py` prints `all integrity checks passed`. Segmentation and token counts depend on the spaCy, tokenizers and transformers versions, which are recorded in `sentence_audit.json` → `environment`. Use `requirements-lock.txt` for identical output.

## Split integrity / leakage checks

Results of the checks on the current data, QA subset unless stated otherwise:

- **IDs.** IDs are unique, and the train, val and test sets are pairwise disjoint (all tasks). The sentence files contain exactly the IDs of their source split, in the same order.
- **Splits.** `val` comes only from the original HF `train` split, and `test` equals the original HF `test` split, unchanged.
- **Source groups.** No `source_group` (the source-ID proxy), *normalized `query`* or *normalized `context`* is shared between any two splits. Each source group has exactly 6 responses, one per RAGTruth generator model, and all 6 stay in the same split.
- **Shared outputs.** The only `output` string that appears in more than one split is the refusal `"Unable to answer based on given passages."` (183 train / 26 val / 26 test). That is a template, not leakage.
- **Limitation.** `source_group` is a hash of `(task_type, query, context)`, because the processed dataset has no `source_id`. Near-duplicate passages that differ in wording, or the same document under a different query, would not be detected. Verify against the original RAGTruth `source_id` before reporting final test numbers.

## Output

`data/processed/{qa,all}_{train,val,test}.jsonl`: row-level examples with original source, answer, two binary labels, full original spans, model-specific paired token lengths, response token counts, and split-proxy source IDs.

`data/processed/audit_report.json`: token length percentiles and fractions exceeding a **configurable audit budget** (default 512), label rates, train/validation overlap checks, and QA response-length buckets. Percentile cutoffs use QA **training** data only.

`data/processed/qa_{train,val,test}_sentences.jsonl` and `data/processed/sentence_audit.json`: sentence-level files from `scripts/prepare_sentences.py` (below). **Sentence-level labels are not inferred.**

## Sentence-level preprocessing (zero-shot NLI baseline)

`scripts/prepare_sentences.py` reads the existing `qa_{train,val,test}.jsonl` files. It does not re-download or re-split anything, and it never modifies the full-context files, which stay available for fine-tuning. The script refuses to write to any of the full-context files or to `audit_report.json`.

### What it does

1. **Segmentation.** `context` → evidence sentences and `output` → response sentences. The text is first split on newlines, so a sentence never crosses a passage or list-item line. Each line is then segmented by the spaCy `sentencizer`, with no model download. Whitespace is trimmed. Two conservative repairs are applied, both within a single line:
   - a bare list marker (`1.`, `b)`, `-`) is joined to the text that follows it;
   - two pieces with no whitespace between them (`Worksheets.Add`, `site.com`) are re-joined.

   You can use a different segmenter with `--segmenter en_core_web_sm`, after installing that model.
2. **Span mapping.** Each original span in `hallucination_labels` (stored as `original_spans`) is intersected at the character level with every response sentence. **All** overlaps are recorded, including spans that cover several sentences.
3. **Token lengths.** Lengths use the `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` tokenizer, counted per sentence without special tokens. The length of every evidence×response pair is computed as `n_e + n_r + 3`, which is `[CLS] e [SEP] r [SEP]`. The script checks this formula by jointly re-tokenizing 3,000 random real pairs. Nothing is truncated or dropped.
4. **Validation.** The script re-reads the written files and compares them with the source files (see [Validation](#validation)). It exits with code 1 if any integrity check fails.

### Verified annotation conventions (wandb/RAGTruth-processed, QA)

- `hallucination_labels` is a JSON string holding a list of `{start, end, text, meta, label_type, implicit_true, due_to_null}`. `start` and `end` are **0-based, end-exclusive character offsets into `output`**, and `output[start:end] == text` for all 2,927 QA spans.
- `label_type` takes one of four annotation categories: `Evident Conflict`, `Subtle Conflict`, `Evident Baseless Info`, `Subtle Baseless Info`. **These categories are not our classification labels.** The task labels are the processed binary fields `evident_conflict` and `baseless_info`.
- Empirically, for 100% of QA examples, `evident_conflict = 1` exactly when the example has a span of type `Evident Conflict` *or* `Subtle Conflict`. Likewise, `baseless_info = 1` exactly when it has a span of type `Evident Baseless Info` *or* `Subtle Baseless Info`. So the processed "evident_conflict" label also covers subtle conflicts. The script re-checks this for every example (`label_span_inconsistency`) and never changes the processed labels.

### Output schema (`qa_{split}_sentences.jsonl`, one line per original QA example)

| field | description |
|---|---|
| `id`, `source_group`, `task_type`, `quality`, `model`, `query` | copied unchanged from `qa_{split}.jsonl` |
| `original_split` / `split` | HF split (`train`/`test`) / project split (`train`/`val`/`test`), unchanged |
| `context`, `output` | original strings; all offsets index into these |
| `labels`, `label_names`, `response_labels` | original response-level labels, as `[evident_conflict, baseless_info]` and as a dict |
| `original_spans` | original span annotations, unchanged |
| `evidence_sentences[]` | `{idx, start, end, text, n_tokens}` where `context[start:end] == text` |
| `response_sentences[]` | `{idx, parent_id, start, end, text, n_tokens, max_pair_tokens, n_pairs_over_limit, full_context_pair_tokens, span_overlaps[]}` where `output[start:end] == text` |
| `response_sentences[].span_overlaps[]` | `{span_index, label_type, consistent_processed_label, overlap_start, overlap_end, overlap_chars, fraction_of_span, fraction_of_sentence}`. Empty means the sentence overlaps no annotated span. |
| `span_sentence_map[]` | `{span_index, label_type, sentence_indices[]}`: every response sentence that a span touches |
| `pair_token_stats` | `{n_pairs, max_pair_tokens, n_pairs_over_limit, context_tokens}` for this example |

`n_tokens` excludes special tokens. `consistent_processed_label` names the binary label that the span's `label_type` was verified to co-occur with. It is a property of the **span**, **not** a sentence label.

### How this supports the zero-shot NLI baseline

An MNLI/FEVER/ANLI model scores a (premise, hypothesis) pair. Every (evidence sentence, response sentence) pair is a short premise-hypothesis input in the format the model was trained on. Across all splits there are 628,174 such pairs, with a maximum of 233 tokens, so **none exceed 512**. This avoids truncating the long full-context inputs. `span_overlaps` makes it possible to evaluate sentence-level predictions against the gold spans afterwards. How sentence and pair scores are combined into response-level `evident_conflict` and `baseless_info` predictions (max entailment over evidence, contradiction thresholds, and so on) is left for later work.

### Limitations

- **Multi-sentence evidence.** A claim may need several evidence sentences together, such as a fact split across sentences, a pronoun whose antecedent is in an earlier sentence, or a table or list spread over lines. A single-sentence premise can then look "neutral" or "baseless" even though the full context supports the claim. Response sentences also lose their context, for example "It costs $5." Possible fixes include sliding windows of k sentences, a full-context premise, or adding the question. All of these need a new length audit.
- **Truncation is not solved in general.** Pairing the *full context* with one response sentence still exceeds 512 tokens in 2.0% (train), 1.0% (val) and 2.3% (test) of response sentences (`full_context_x_response_sentence_pairs` in the audit). Multi-sentence windows will push more pairs over the limit.
- **Spans and sentences don't align exactly.** Annotated spans are not sentence-aligned: 519 of 2,927 spans cover more than one sentence, and many cover only part of a sentence. Turning overlaps into sentence labels requires an explicit policy (for example "any overlap" or "≥50% of the sentence"), and that policy should be reported. Most response sentences in positive examples overlap no span (9,018 of them in train). Do not give them the response's positive label.
- **Rule-based segmentation.** The sentencizer is rule-based, so code blocks, tables, abbreviations and unusual punctuation can still be split imperfectly. A handful of `Sure!`, `Pros:` and `` ``` `` lines become standalone sentences. Offsets are always exact, but sentence quality is not guaranteed.
- **Quadratic pair count.** The number of pairs grows as E×R per example (up to 45 evidence × 52 response sentences). Inference cost should be planned with that in mind.

### Validation

Fatal checks, which make the script exit with code 1:
- sentence text equals the original substring;
- sentences are ordered and don't overlap;
- sentences are trimmed;
- no non-whitespace character is left uncovered;
- every response sentence has `parent_id == id`;
- no duplicate IDs;
- the ID set per split matches the source files exactly, and no ID appears in two splits;
- `split` and `original_split` are unchanged;
- labels, `context`/`output` and `original_spans` are unchanged;
- the pair-length formula matches joint tokenization.

Reported but not fatal: span offsets out of bounds, span text mismatches, unexpected span keys, unknown `label_type`, spans that overlap no sentence, and label/span inconsistencies. Spans that cross sentence boundaries or cover only part of a sentence are counted under `validation.informational`. All counts and up to 50 example issues are written to `sentence_audit.json`.

## Methods / important caveats

- Original Hugging Face **test** split is preserved untouched for evaluation. A source-grouped validation set is formed only from the original train split. For strict final evaluation, confirm group identities against `source_id` from the original [RAGTruth repository](https://github.com/ParticleMedia/RAGTruth). The processed dataset lacks an explicit `source_id`, so we hash `(task_type, query, context)` as a source proxy. The script reports possible original train/test proxy collisions but does not silently alter the official split.
- Token counts are measured without truncation using **both** checkpoints: `microsoft/deberta-v3-base` and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. A 512-token threshold is an audit budget; verify the chosen checkpoint's actual supported maximum. Match pair formatting in later modeling code (`tokenizer(context, output, ...)`). A model that also uses the question must re-run the audit for that format.
- Truncation is **not** performed at preprocessing time. Do not assume `truncation=True` is a safe solution to losing evidence or hallucinated answer spans.
- Span indices in `original_spans` refer to the **full original output**, not necessarily token offsets. Editing/stripping answer text can invalidate them. Keep the original output unchanged.
- Records with `quality` of `truncated` or `incorrect_refusal` are retained and flagged. Decide whether/how to handle them with the team, based on the research target, before training.
- Labels are multi-label `labels=[evident_conflict, baseless_info]`.
- By default, outputs do not contain pretrained-model token ID arrays because those should be created by the model trainer with its actual truncation/chunking strategy.

## Source / attribution

- [RAGTruth-processed on Hugging Face](https://huggingface.co/datasets/wandb/RAGTruth-processed)
- [Original RAGTruth data and annotations](https://github.com/ParticleMedia/RAGTruth)
- [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base)
- [NLI-DeBERTa](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)

This repository publishes only scripts and audit statistics. The generated JSONLs contain the dataset's text and are git-ignored. Before redistributing them anywhere, check the license and terms of the Hugging Face dataset and of the original RAGTruth release.
