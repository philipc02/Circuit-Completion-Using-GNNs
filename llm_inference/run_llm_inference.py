from transformers import pipeline
import torch
import json
import glob
import re, json

def extract_json(s):
    m = re.search(r'\{.*?\}', s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except:
        return None


MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "text-generation",
    model=MODEL,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

component_classes = ["Resistor", "Capacitor", "VoltageSource", "Subcircuit", "CurrentSource", "Inductor", "Diode", "BJT", "MOSFET"]

def build_prompt(partial_netlist):
    return f"""
You are a circuit design assistant. Identify the missing component in a partial circuit.

Partial netlist:
{partial_netlist}

Component type options: {component_classes}

Respond with ONLY this JSON and nothing else:

{{
  "prediction": "<component_type>"
}}
"""

def predict_component(netlist_text):
    prompt = build_prompt(netlist_text)

    out = pipe(
        prompt,
        max_new_tokens=32,
        do_sample=False,
        return_full_text=False
    )

    full_out = out[0]["generated_text"].strip()
    print("\n=== RAW MODEL OUTPUT ===")
    print(full_out)
    print("========================\n")

    pred_json = extract_json(full_out)
    if pred_json:
        return pred_json.get("prediction")
    else:
        return None

    
paths = glob.glob("partial_circuits/*.net")
total = len(paths)

with open("llm_predictions.json", "w") as f:
    f.write("[\n")

    first = True

    for i, path in enumerate(paths):
        try:
            try:
                with open(path, "r", encoding="utf-8") as fr:
                    net_txt = fr.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="latin-1") as fr:
                    net_txt = fr.read()

            lines = [l for l in net_txt.splitlines() if l.strip()]
            cleaned = "\n".join(l for l in lines if l not in {"*", "."})

            pred = predict_component(cleaned)
            entry = {"file": path, "pred": pred}

            if not first:
                f.write(",\n")
            first = False

            json.dump(entry, f)
            f.flush()

            print(f"[{i+1}/{total}] {path} → {pred}")

        except KeyboardInterrupt:
            print("\nStopped by user. JSON so far is valid.")
            break

    f.write("\n]")

print("Done!")