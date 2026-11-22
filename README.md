# RAGTruth QA Preprocessing

Reproducible data preprocessing for our COMS4705 NLP final project: **Hallucination Detection in Question Answering**.

We use [wandb/RAGTruth-processed](https://huggingface.co/datasets/wandb/RAGTruth-processed) to detect two types of hallucinations in AI-generated responses:

- **Evident conflict:** Information that contradicts the provided evidence.
- **Baseless information:** Information not supported by the provided evidence.

Our task is **multi-label binary classification**, with labels `[evident_conflict, baseless_info]`. Both labels may be positive for the same response.

## 1. Setup

Requires Python 3.11+.

```bash
git clone https://github.com/Qinyinnn/ragtruth-qa-preprocessing.git
cd ragtruth-qa-preprocessing

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt

# Generate full-context datasets
python scripts/preprocess.py

# Generate sentence-level datasets
python scripts/prepare_sentences.py

# Validate generated sentence-level data
python scripts/prepare_sentences.py --validate-only
```

The first run downloads the dataset and tokenizers from Hugging Face. No PyTorch installation is required for preprocessing.

**Note:** Generated JSONL datasets (~127 MB) are excluded from GitHub. Run the scripts above to reproduce them locally. Both audit reports are included in the repository.

### Run the frozen NLI baseline

```bash
python scripts/evaluate_nli_baseline.py
python scripts/aggregate_nli_baseline.py
```

### Generate milestone metrics and figures

```bash
python scripts/metrics_report.py
```

Raw model predictions and generated datasets are excluded from Git. Final metrics and report figures may be committed.

## 2. Key Findings & Preprocessing Decisions

| Finding                                                   | Decision                                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------------------- |
| ~30% of QA training examples exceed the 512-token budget  | Created sentence-level evidence–response pairs for zero-shot NLI          |
| Longer responses have higher hallucination-positive rates | Preserved response lengths for a length-only Logistic Regression baseline |
| Evident conflicts are rare (~7% of QA training examples)  | Preserved original label distributions for class-imbalance analysis       |
| Original annotations contain four hallucination types     | Retained the two processed binary labels and verified their mapping       |
| Some hallucination spans cross sentence boundaries        | Preserved original character offsets and recorded sentence overlaps       |
| Multiple responses share the same source                  | Used source-grouped validation splitting to reduce leakage                |
| Sentence-level labels can be derived from original hallucination spans | Used span overlaps to derive labels without modifying the original ground truth |
| Only 30 of 41,409 QA sentences (0.072%) contain both hallucination types | Retained multi-label ground truth; evaluated three-class NLI as an approximation |
| NLI inference requires combining predictions across evidence sentences | Compared max-score and support-first aggregation on validation data |

For detailed label definitions, aggregation methods, and evaluation
protocols, see [Evaluation Methodology](docs/evaluation_methodology.md).

### Dataset Splits

| Split      | QA examples |
| ---------- | ----------: |
| Train      |       4,260 |
| Validation |         774 |
| Test       |         900 |
| **Total**  |   **5,934** |

- Validation is created from the original training split using `GroupShuffleSplit(seed=42)`.
- The original Hugging Face test split is preserved unchanged.
- Related responses are grouped using a hash of `(task_type, query, context)` to prevent source overlap across splits.
- The processed dataset does not provide the original `source_id`, so near-duplicate sources may remain undetected. Original source IDs should be verified before final evaluation.

## 3. Which Files Should Teammates Use?

All generated files are located in `data/processed/`.

| Model / Experiment              | Files                                 |
| ------------------------------- | ------------------------------------- |
| Majority-class baseline         | `qa_{train,val,test}.jsonl`           |
| Length-only Logistic Regression | `qa_{train,val,test}.jsonl`           |
| Zero-shot NLI-DeBERTa           | `qa_{train,val,test}_sentences.jsonl` |
| QA-only DeBERTa fine-tuning     | `qa_{train,val,test}.jsonl`           |
| All-task fine-tuning            | `all_{train,val,test}.jsonl`          |

**Full-context files** contain original evidence, responses, binary labels, hallucination spans, token counts, and source-group identifiers.

**Sentence-level files** contain:

- `evidence_sentences[]`: sentences extracted from the source context.
- `response_sentences[]`: sentences extracted from the generated response.
- `span_overlaps[]`: original hallucination spans overlapping each response sentence.
- `original_spans`: unchanged hallucination annotations.
- `labels` / `response_labels`: original response-level binary labels.
- `pair_token_stats`: sentence-pair token-length statistics.

**Important:** Sentence-level files do not contain newly inferred sentence-level classification labels. Original response-level labels are preserved.

## 4. Sentence-Level Preprocessing

Implemented in `scripts/prepare_sentences.py` for the **zero-shot NLI-DeBERTa baseline**.

### Method

1. **Sentence segmentation:** Split `context` into evidence sentences and `output` into response sentences using newline boundaries and spaCy's rule-based sentencizer. Minor repairs handle list markers and incorrectly split code-like expressions.
2. **Span mapping:** Preserve original hallucination character offsets and record which response sentences overlap each span.
3. **Token-length analysis:** Measure evidence–response sentence pairs using the `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` tokenizer, including special tokens.
4. **Validation:** Verify offsets, original text, labels, split membership, IDs, and token lengths. No original examples or annotations are modified.

### Verified Annotation Mapping

The original RAGTruth annotations contain four categories:

| Original annotation   | Processed binary label |
| --------------------- | ---------------------- |
| Evident Conflict      | `evident_conflict`     |
| Subtle Conflict       | `evident_conflict`     |
| Evident Baseless Info | `baseless_info`        |
| Subtle Baseless Info  | `baseless_info`        |

This mapping was verified across all QA examples.

**Important:** Despite its name, `evident_conflict` also includes _Subtle Conflict_ annotations.

Original hallucination spans use **0-based, end-exclusive character offsets** into the original `output`. All 2,927 QA spans were verified against their corresponding text.

### Sentence-Level Results

| Metric                     |   Train | Validation |   Test |
| -------------------------- | ------: | ---------: | -----: |
| Evidence–response pairs    | 466,813 |     78,125 | 83,236 |
| Maximum pair tokens        |     208 |        223 |    233 |
| Pairs exceeding 512 tokens |       0 |          0 |      0 |

Across all splits, **628,174 sentence pairs** were generated without exceeding the 512-token budget.

No text is truncated during preprocessing.

## 5. Important Limitations

These issues should be considered when implementing the models:

1. **Incomplete evidence:** Some claims require multiple evidence sentences. A single evidence–response pair may appear neutral even when the full context supports the claim. Consider multiple-sentence windows or evidence aggregation.

2. **Prediction aggregation:** The NLI model produces entailment, neutral, and contradiction scores for sentence pairs. How these scores become the two response-level binary predictions is not yet implemented and requires an explicit strategy.

3. **Sentence-label ambiguity:** 519 hallucination spans cross sentence boundaries, and many cover only part of a sentence. Do not automatically assign the response's positive label to every sentence.

4. **Computational cost:** Comparing every evidence sentence with every response sentence produces ~628K pairs. Batched inference or evidence retrieval may be useful.

5. **Remaining truncation:** Sentence splitting resolves the length issue for individual sentence pairs, **not full-context fine-tuning**. Fine-tuning pipelines must separately handle long inputs without silently removing important evidence.

6. **Segmentation noise:** Rule-based sentence splitting may incorrectly segment code, tables, abbreviations, or unusual punctuation. Original text offsets remain preserved.

7. **Data quality:** Examples marked `truncated` or `incorrect_refusal` are retained. The team should agree on whether to include them during training and evaluation.

## 6. Reproducibility & Validation

- Dataset: `wandb/RAGTruth-processed`, pinned to revision `eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b`.
- Tokenizers: `microsoft/deberta-v3-base` and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, with revisions pinned in the scripts.
- Random seed: `42`.
- Exact dependencies: `requirements-lock.txt`.
- Full-context outputs were reproduced byte-for-byte in an independent run. Sentence audit statistics also matched, apart from recorded file paths.

Run:

```bash
python scripts/prepare_sentences.py --validate-only
```

The validation checks include:

- Sentence offsets matching original text.
- No missing non-whitespace content.
- Unchanged labels, spans, and split assignments.
- No duplicated or cross-split IDs.
- Correct parent IDs and pair-token calculations.

The repository includes two reference reports:

| File                                 | Contents                                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `data/processed/audit_report.json`   | Dataset statistics, label distributions, response-length bias, token limits, and split checks  |
| `data/processed/sentence_audit.json` | Sentence counts, pair-token distributions, hallucination-span overlaps, and validation results |

Compare regenerated reports with these references to confirm reproducibility.


## 7. Evaluation Methodology

All models are evaluated on the same QA validation and test
splits using the original response-level multi-label targets:
`[evident_conflict, baseless_info]`.

The majority-class and length-only logistic regression baselines
operate directly at response level.

The frozen NLI-DeBERTa baseline uses sentence-level inference
and aggregates predictions back to response level for comparison.

See [Evaluation Methodology](docs/evaluation_methodology.md)
for label definitions, aggregation rules, and evaluation details.

## 8. References

- [RAGTruth-processed Dataset](https://huggingface.co/datasets/wandb/RAGTruth-processed)
- [Original RAGTruth Dataset](https://github.com/ParticleMedia/RAGTruth)
- [Microsoft DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base)
- [NLI-DeBERTa (MNLI/FEVER/ANLI)](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)

**Data usage:** Only preprocessing scripts and audit statistics are tracked in Git. Before redistributing generated datasets, check the applicable dataset licenses and terms.
