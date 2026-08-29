import glob
import json
import os
import random
import re
from datasets import Dataset
import torch
from torch import bfloat16, float16
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_ID = "google/gemma-2-2b-it"
OUTPUT_DIR = "./finetuned_llm_multidataset"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_adapter")
RANDOM_SEED = 42

COMPONENT_CLASSES = [
    "Resistor", "Capacitor", "VoltageSource", "Subcircuit", 
    "CurrentSource", "Inductor", "Diode", "BJT", "MOSFET"
]

DATASET_FOLDERS = [
    "partial_netlists_masala_chai",
    "partial_netlists_analoggenie",
    "partial_netlists_amsnet",
]

def extract_json(s):
    m = re.search(r"\{.*?\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def build_prompt(partial_netlist):
    return f"""You are a circuit design assistant. Identify the missing component in a partial circuit.

Partial netlist:
{partial_netlist}

Component type options: {COMPONENT_CLASSES}

Respond with ONLY this JSON and nothing else:
"""

# ---------------------------------------------------------
# STEP 1: LOAD EVALUATION SAMPLES
# ---------------------------------------------------------
print("1. Collecting test split netlists...")
all_files = []
for folder in DATASET_FOLDERS:
    found = glob.glob(os.path.join(folder, "*.net"))
    all_files.extend(found)

random.seed(RANDOM_SEED)
random.shuffle(all_files)

split_idx = int(len(all_files) * 0.80)
test_files = all_files[split_idx:]

eval_samples = []
for path in test_files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            net_txt = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            net_txt = f.read()

    lines = [l for l in net_txt.splitlines() if l.strip()]
    cleaned = "\n".join(l for l in lines if l not in {"*", "."})

    target_class = next((cls for cls in COMPONENT_CLASSES if cls.lower() in path.lower()), None)
    if target_class:
        eval_samples.append((path, cleaned, target_class))

print(f"Test samples to evaluate: {len(eval_samples)}")

# ---------------------------------------------------------
# STEP 2: LOAD MODEL & LORA ADAPTER FOR INFERENCE ONLY
# ---------------------------------------------------------
print("\n2. Loading model & adapter...")
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = bfloat16 if use_bf16 else float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Correct side for batching / generation

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    torch_dtype=compute_dtype,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

# ---------------------------------------------------------
# STEP 3: FAST EVALUATION LOOP WITH RESUME CAPABILITY
# ---------------------------------------------------------
print("\n3. Running evaluation loop...")
output_json_path = "llm_predictions_lora_combined.json"

# Read already processed files if resuming mid-run
processed_files = set()
if os.path.exists(output_json_path):
    try:
        with open(output_json_path, "r") as f:
            existing_data = json.load(f)
            processed_files = {entry["file"] for entry in existing_data if "file" in entry}
            print(f"Resuming: Found {len(processed_files)} already evaluated netlists.")
    except Exception:
        processed_files = set()

remaining_samples = [s for s in eval_samples if s[0] not in processed_files]

with open(output_json_path, "w") as f:
    f.write("[\n")
    first = True

    pbar = tqdm(eval_samples, desc="Evaluating", unit="sample")

    for path, cleaned_netlist, gt_label in pbar:
        try:
            prompt = build_prompt(cleaned_netlist)
            
            # Enforce max input context truncation (e.g., 1024 tokens)
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    use_cache=True,  # Re-enabled fast caching
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_length = inputs.input_ids.shape[1]
            full_out = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

            pred_json = extract_json(full_out)
            pred = pred_json.get("prediction") if pred_json else None

            entry = {"file": path, "ground_truth": gt_label, "pred": pred}

            if not first:
                f.write(",\n")
            first = False

            json.dump(entry, f)
            f.flush()

            pbar.set_postfix({"File": os.path.basename(path), "GT": gt_label, "Pred": pred})

        except KeyboardInterrupt:
            print("\nInference interrupted by user. Partial progress saved.")
            break

    f.write("\n]")

print(f"\nCompleted evaluation! Results saved to '{output_json_path}'.")