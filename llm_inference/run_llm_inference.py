from transformers import pipeline
import torch
import json
import glob

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

Reply with JSON only in this exact format:
{{
  "prediction": "<component_type>"
}}
"""

def predict_component(netlist_text):
    prompt = build_prompt(netlist_text)

    out = pipe(
        prompt,
        max_new_tokens=128,
        do_sample=False,
        return_full_text=False
    )

    full_out = out[0]["generated_text"].strip()
    print("\n=== RAW MODEL OUTPUT ===")
    print(full_out)
    print("========================\n")

    try:
        pred_json = json.loads(full_out)
        return pred_json.get("prediction")
    except:
        return None

    
results = []

for path in glob.glob("partial_circuits/*.net"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            net_txt = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            net_txt = f.read()
    lines = [l for l in net_txt.splitlines() if l.strip()]
    cleaned_netlist = "\n".join(l for l in lines if l not in {"*", "."})

    pred = predict_component(cleaned_netlist)

    results.append({"file": path, "pred": pred})

with open("llm_predictions.json", "w") as f:
    json.dump(results, f, indent=2)

print("Inference completed!")