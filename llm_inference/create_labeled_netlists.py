import os
import random
import json
import csv
from pathlib import Path

SOURCE_DIR = "../graph_parsers/netlists_ltspice_examples/"
OUT_DIR = "partial_netlists/"
META_FILE = "metadata.csv"

Path(OUT_DIR).mkdir(exist_ok=True)

def extract_component_class(line):
    initial = line.strip()[0].upper()
    if initial == 'R': return "Resistor"
    if initial == 'C': return "Capacitor"
    if initial == 'V': return "VoltageSource"
    if initial == 'X': return "Subcircuit"
    if initial == 'I': return "CurrentSource"
    if initial == 'L': return "Inductor"
    if initial == 'D': return "Diode"
    if initial == 'Q': return "BJT"
    if initial == 'M': return "MOSFET"
    return "Unknown"

metadata = []

for filepath in os.listdir(SOURCE_DIR):
    if not filepath.endswith(".net"):
        continue
    
    with open(os.path.join(SOURCE_DIR, filepath), "r") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]

    # choose component to remove
    removable_lines = [line for line in lines if line[0].isalpha()]
    if not removable_lines:
        continue

    target_line = random.choice(removable_lines)
    target_type = extract_component_class(target_line)
    if target_type == "Unknown":
        continue

    # make partial netlist
    partial = ["? Missing component" if l == target_line else l for l in lines]

    # create output filename
    base = filepath.replace(".net", "")
    out_name = f"{base}_missing_{target_type}.net"

    with open(os.path.join(OUT_DIR, out_name), "w") as f:
        for l in partial:
            f.write(l + "\n")

    metadata.append({
        "file": out_name,
        "ground_truth": target_type,
        "removed_line": target_line
    })

# save metadata for evaluation
with open(META_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "ground_truth", "removed_line"])
    writer.writeheader()
    writer.writerows(metadata)

print(f"Processed {len(metadata)} netlists into {OUT_DIR}")
