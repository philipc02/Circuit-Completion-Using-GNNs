import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Load results
with open("model_complexity_study_MT_1_amsnet/aggregated_results.json", "r") as f:
    results = json.load(f)

# Group results by representation
grouped = defaultdict(list)
for r in results:
    grouped[r["representation"]].append(r)

# Sort by layer count
for rep in grouped:
    grouped[rep] = sorted(grouped[rep], key=lambda x: x["layers"])

plt.figure(figsize=(8, 5))

for rep, values in grouped.items():
    layers = [v["layers"] for v in values]
    mean_f1 = [v["mean_f1"] for v in values]
    std_f1 = [v["std_f1"] for v in values]

    plt.plot(layers, mean_f1, marker="o", label=rep)


plt.xlabel("Number of GNN Layers")
plt.ylabel("Mean F1 Score")
plt.title("Model Complexity Study")
plt.grid(True)

plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

plt.tight_layout()
plt.xticks(range(min(layers), max(layers) + 1))
plt.savefig("model_complexity_plot_MT_1.png", dpi=300)
plt.show()
