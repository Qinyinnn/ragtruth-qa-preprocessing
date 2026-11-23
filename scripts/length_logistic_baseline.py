import json
import os
from sklearn.linear_model import LogisticRegression

# 1. Load data
def load_data(filename):
    data = []

    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return data


train = load_data("data/processed/qa_train.jsonl")
val = load_data("data/processed/qa_val.jsonl")


# 2. Get response lengths
X_train = []
X_val = []

for row in train:
    X_train.append([row["response_tokens"]])

for row in val:
    X_val.append([row["response_tokens"]])


# 3. Train separate logistic regression models
predictions_by_label = []

for j in range(2):
    y_train = []

    for row in train:
        y_train.append(row["labels"][j])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    predictions_by_label.append(predictions)


# 4. Save validation predictions
os.makedirs("results/length_logistic", exist_ok=True)

with open("results/length_logistic/val_predictions.jsonl", "w") as f:
    for i in range(len(val)):
        prediction = {
            "response_id": str(val[i]["id"]),
            "predicted_labels": [
                int(predictions_by_label[0][i]),
                int(predictions_by_label[1][i])
            ]
        }

        f.write(json.dumps(prediction) + "\n")

print("Saved", len(val), "validation predictions.")
