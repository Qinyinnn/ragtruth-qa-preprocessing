import json
import os

# 1. Read JSONL files
def load_data(filename):
    data = []

    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return data

train = load_data("data/processed/qa_train.jsonl")
val = load_data("data/processed/qa_val.jsonl")

# 2. Find the majority class for each label
majority_labels = []

for j in range(2):
    count_1 = 0

    for row in train:
        if row["labels"][j] == 1:
            count_1 += 1

    count_0 = len(train) - count_1

    if count_1 > count_0:
        majority_labels.append(1)
    else:
        majority_labels.append(0)

print("Majority labels:", majority_labels)

# 3. Predict every validation response
os.makedirs("results/majority", exist_ok=True)

with open("results/majority/val_predictions.jsonl", "w") as f:
    for row in val:
        prediction = {
            "response_id": str(row["id"]),
            "predicted_labels": majority_labels
        }

        f.write(json.dumps(prediction) + "\n")

print("Saved", len(val), "validation predictions.")
