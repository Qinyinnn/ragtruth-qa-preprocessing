
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MODEL_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16

INPUT_PATH = Path("data/processed/qa_val_sentences.jsonl")
OUTPUT_PATH = Path("results/nli_baseline/val_predictions.jsonl")

LABELS = ("evident_conflict", "baseless_info")


def get_sentence_labels(sentence):
    """Derive labels from original RAGTruth span annotations."""
    return [
        int(any(
            span["consistent_processed_label"] == label
            for span in sentence["span_overlaps"]
        ))
        for label in LABELS
    ]


def load_completed_ids(path):
    """Read IDs of responses already processed."""
    if not path.exists():
        return set()

    completed = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                completed.add(str(json.loads(line)["response_id"]))

    return completed


def run_inference(example, tokenizer, model):
    """Run NLI inference for one complete QA response."""

    evidence = example["evidence_sentences"]
    claims = example["response_sentences"]

    # Generate all evidence-claim pairs.
    # Claim-major order: every evidence sentence for each claim.
    pairs = [
        (ev["text"], claim["text"])
        for claim in claims
        for ev in evidence
    ]

    all_probs = []

    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start:start + BATCH_SIZE]

        inputs = tokenizer(
            [p[0] for p in batch],
            [p[1] for p in batch],
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(DEVICE)

        # Every sentence pair was audited to fit 512 tokens.
        if inputs["input_ids"].shape[1] > 512:
            raise ValueError(
                f"Unexpected overlength pair in response {example['id']}"
            )

        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        all_probs.extend(probs.cpu().tolist())

    # Restore the original claim × evidence grouping.
    n_evidence = len(evidence)
    claim_results = []

    for i, claim in enumerate(claims):
        first = i * n_evidence
        last = first + n_evidence

        claim_results.append({
            "sentence_idx": claim["idx"],
            "true_sentence_labels": get_sentence_labels(claim),
            "pair_probabilities": all_probs[first:last],
        })

    return {
        "response_id": example["id"],
        "true_response_labels": example["labels"],
        "n_evidence_sentences": n_evidence,
        "n_response_sentences": len(claims),
        "claims": claim_results,
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open(encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    completed_ids = load_completed_ids(OUTPUT_PATH)

    print(f"Device: {DEVICE}")
    print(f"Validation responses: {len(examples)}")
    print(f"Already completed: {len(completed_ids)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    ).to(DEVICE)
    model.eval()

    print("NLI labels:", model.config.id2label)

    start_time = time.perf_counter()
    processed = 0
    total_pairs = 0

    # Append one response at a time for resumability.
    with OUTPUT_PATH.open("a", encoding="utf-8") as out:
        for example in examples:
            if str(example["id"]) in completed_ids:
                continue

            result = run_inference(example, tokenizer, model)

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

            processed += 1
            total_pairs += (
                result["n_evidence_sentences"]
                * result["n_response_sentences"]
            )

            if processed % 25 == 0:
                if DEVICE == "mps":
                    torch.mps.synchronize()

                elapsed = time.perf_counter() - start_time
                print(
                    f"Processed {processed} new responses | "
                    f"{total_pairs:,} pairs | "
                    f"{elapsed:.1f}s elapsed",
                    flush=True,
                )

    if DEVICE == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start_time

    print("\nInference complete!")
    print(f"New responses processed: {processed}")
    print(f"New NLI pairs processed: {total_pairs:,}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
