# RAGTruth: Labeling and Aggregation Decisions

This document records the labeling conventions, aggregation rules, and evaluation protocol for our **NLI-Based Hallucination Detection** project using the QA portion of `wandb/RAGTruth-processed`.

> **Status:** Original ground-truth labels, span-derived sentence labels, response-level evaluation, and sentence-to-response OR aggregation are established. Evidence aggregation for the frozen NLI baseline is **under evaluation**, not yet finalized.

## 1. Original task: Response-level multi-label classification

Each generated QA response has two **independent binary labels**, in this order:

1. `evident_conflict`: Information conflicting with the provided source evidence. In our processed data, this combines original **Evident Conflict** and **Subtle Conflict** spans.
2. `baseless_info`: Information unsupported by the provided source evidence. This combines original **Evident Baseless Info** and **Subtle Baseless Info** spans.

The target is `[evident_conflict, baseless_info]`, and all four combinations are valid: `[0,0]`, `[1,0]`, `[0,1]`, and `[1,1]`.

**Decision:** Preserve the original response-level labels as ground truth. Do not infer, overwrite, or relabel them.

## 2. Sentence-level ground truth: Derived from original spans

RAGTruth includes hallucination annotations with character offsets into the original generated response. Our preprocessing splits each response into sentences and records the original annotations overlapping each sentence in `response_sentences[].span_overlaps`.

For each response sentence and each label:

- Set `evident_conflict = 1` if **any** original conflict span overlaps that sentence.
- Set `baseless_info = 1` if **any** original baseless-information span overlaps that sentence.
- Otherwise set the corresponding label to 0.

These are **span-derived sentence-level ground-truth labels**, **not predictions from the NLI model**. A sentence can have both labels. The definition is *contains at least one annotated span of this type*, not *the entire sentence is hallucinated*.

**Decision:** Compute labels from `span_overlaps` when loading the data, without modifying the original `qa_{train,val,test}_sentences.jsonl` files.

### Sentence-to-response ground-truth consistency

For each label `k`, reconstruct a response label by taking the OR (equivalently, the maximum) across its sentences:

```python
y_response[k] = max(y_sentence_i[k] for y_sentence_i in sentences)
```

The audit found **zero reconstruction mismatches in train and validation**. The test distribution was calculated, but the test reconstruction mismatch count was not included in the shared terminal output and should be confirmed before claiming a zero-mismatch test audit.

### Sentence-level label distribution

| Sentence labels `[conflict, baseless]` | Train | Validation | Test |
|---|---:|---:|---:|
| `[0,0]` Neither | 27,180 | 4,933 | 5,302 |
| `[1,0]` Conflict only | 420 | 74 | 29 |
| `[0,1]` Baseless only | 2,710 | 397 | 334 |
| `[1,1]` Both | 26 | 3 | 1 |
| **Total sentences** | **30,336** | **5,407** | **5,666** |

Only **30 out of 41,409 sentences (~0.072%)** have both positive labels. Thus, a mutually exclusive three-class sentence-level approximation is worth testing for the **frozen** NLI baseline, but the original multi-label labels must still be preserved in evaluation. Class imbalance, particularly for conflicts, is substantial.

## 3. Frozen NLI-DeBERTa baseline: Three prediction stages

**Checkpoint:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`

The frozen checkpoint outputs three probabilities for each premise–hypothesis pair: **entailment, neutral, contradiction**.

### A. Evidence sentence × response sentence: Pair-level inference

To avoid the 512-token input limit, the source context and generated response are segmented into sentences. Treat each source sentence as a **premise** and each response sentence as a **hypothesis**. Run NLI on every evidence–claim pair.

For a response with `E` evidence sentences and `C` response sentences, exhaustive comparison produces `E × C` NLI predictions. The current validation set contains **774 QA responses and 78,125 pairs**.

### B. Evidence → claim: Evidence aggregation (NOT FINALIZED)

The NLI model outputs multiple distributions per claim, one for each evidence sentence. These must be combined into a single assessment of each claim.

Candidate methods to compare on **validation**:

1. **Max-score:** Compare the strongest entailment and contradiction probabilities across evidence sentences; if neither reaches its decision threshold, classify the claim as unsupported.
2. **Support-first:** If strong supporting evidence exists, prioritize entailment; otherwise check for contradiction, then unsupported information.

**Why this remains experimental:** A relevant source sentence can strongly entail a claim while another, unrelated source sentence yields an erroneously high contradiction score. A single maximum contradiction is therefore not a reliable final rule. Thresholds and aggregation strategy should be selected using validation, not a few anecdotes.

### C. NLI assessment → two binary hallucination labels

The **provisional** sentence-level mapping is:

| Aggregated NLI assessment | `evident_conflict` | `baseless_info` |
|---|---:|---:|
| Entailment / supported | 0 | 0 |
| Contradiction | 1 | 0 |
| Neutral / unsupported | 0 | 1 |

**Caveats:** Neutral for one evidence–claim pair does **not** necessarily mean a claim is unsupported by the *full* source. Also, a mutually exclusive three-class assessment cannot output `[1,1]` for one sentence; those rare double-positive ground-truth sentences remain in evaluation rather than being dropped.

### D. Claim → response: OR aggregation

Once each claim sentence has two binary predictions, combine them by taking the OR **independently for each label**:

```python
y_pred_response[k] = max(y_pred_sentence_i[k] for y_pred_sentence_i in sentences)
```

This produces predictions directly comparable with the original response-level RAGTruth labels. Unlike evidence aggregation, this OR step is an established, interpretable rule matching the *any hallucination of this type anywhere in the response* task definition.

## 4. Shared evaluation protocol across models

**Primary evaluation is response-level**, using the same QA train, validation, and test splits and the original response-level two-label targets. This ensures comparability despite models having different internal processing.

**Secondary evaluation is sentence-level**, for models such as the frozen NLI baseline that produce sentence-level predictions. Sentence ground truth comes from span overlaps, and sentence-level errors are analyzed separately.

Report **precision, recall, and F1 per label**, plus macro F1 (and other agreed summary metrics). Do not rely on accuracy alone because the positive classes, especially conflict, are rare.

| Baseline / model | Input unit | Ground-truth labels | Needs aggregation? |
|---|---|---|---|
| Majority-class baseline | Whole response | Original response-level binary labels | No |
| Length-only logistic regression | Response length | Original response-level binary labels | No |
| Frozen NLI-DeBERTa | Evidence–claim sentence pairs | Span-derived sentence labels; original response labels for primary evaluation | **Yes:** evidence → claim, provisional NLI mapping, then sentence → response |
| Fine-tuned DeBERTa models | Per planned training setup | Original response-level binary labels | Not if trained directly for response-level two-label outputs |

For the **majority-class baseline**, choose constant predictions based on training labels. For **length-only logistic regression**, train one binary classifier per label using response length and the original response-level targets. Neither needs sentence segmentation or span-derived labels.

**Data handling:** Use validation for thresholds, aggregation-method selection, and model selection. Keep test held out until the method is fixed. Do not discard the rare `[1,1]` sentences or alter original labels to fit the frozen NLI output space.

## 5. Current implementation status

- [x] Generated QA preprocessing and sentence-level JSONL files.
- [x] Preserved the original response-level labels, spans, offsets, and sentence-overlap mappings.
- [x] Audited derived sentence labels and their distribution.
- [x] Confirmed zero train and validation reconstruction mismatches.
- [x] Loaded frozen DeBERTa on a Mac with PyTorch MPS and tested real sentence pairs.
- [ ] Finish full **validation** inference and cache raw pair-level probabilities in `results/nli_baseline/val_predictions.jsonl` (last reported progress: 25/774 responses).
- [ ] Compare evidence-aggregation rules and select thresholds on validation.
- [ ] Calculate sentence-level and response-level validation metrics.
- [ ] Confirm test label-reconstruction consistency, then run held-out test inference and final evaluation after method selection.

## 6. Relevant files

```text
scripts/
  preprocess.py
  prepare_sentences.py
  audit_sentence_labels.py
  evaluate_nli_baseline.py

data/processed/
  qa_train_sentences.jsonl
  qa_val_sentences.jsonl
  qa_test_sentences.jsonl

results/nli_baseline/
  val_predictions.jsonl   # generated inference cache; may still be incomplete
```

**Methodological distinction:** Deriving ground-truth sentence labels from existing annotated spans and OR-reconstructing response labels is deterministic. **Evidence aggregation of NLI predictions is a modeling decision** whose validity needs empirical testing.
