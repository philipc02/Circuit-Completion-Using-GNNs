import json
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


with open("llm_predictions.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# ground truth from file name
def extract_gt(path):
    m = re.search(r"_missing_([A-Za-z]+)\.net$", path)
    return m.group(1) if m else None

df["gt"] = df["file"].apply(extract_gt)

y_true = df["gt"]
y_pred = df["pred"]

weighted_f1 = f1_score(y_true, y_pred, average="weighted")

print("===================================")
print(" LLM Component Classification Stats")
print("===================================\n")
print(f"Weighted F1:      {weighted_f1:.4f}\n")

print("---- Classification Report ----")
print(classification_report(y_true, y_pred))
