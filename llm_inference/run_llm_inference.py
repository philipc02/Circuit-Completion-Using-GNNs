import glob
import json
import os
import random
import re
from collections import Counter
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
import torch
from torch import bfloat16, float16
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# Environment & Symlink Safety
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_ID = "google/gemma-2-2b-it"
OUTPUT_DIR = "./finetuned_llm_multidataset"
RANDOM_SEED = 42

COMPONENT_CLASSES = [
    "Resistor",
    "Capacitor",
    "VoltageSource",
    "Subcircuit",
    "CurrentSource",
    "Inductor",
    "Diode",
    "BJT",
    "MOSFET",
]

DATASET_FOLDERS = [
    "partial_netlists_masala_chai",
    "partial_netlists_analoggenie",
    "partial_netlists_amsnet",
]


def extract_json(s):
    """Extract JSON block from LLM output."""
    m = re.search(r"\{.*?\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def build_prompt(partial_netlist, target_component=None):
    """Construct prompt for fine-tuning or inference."""
    prompt = f"""You are a circuit design assistant. Identify the missing component in a partial circuit.

Partial netlist:
{partial_netlist}

Component type options: {COMPONENT_CLASSES}

Respond with ONLY this JSON and nothing else:
"""
    if target_component:
        prompt += f'\n{{\n  "prediction": "{target_component}"\n}}'
    return prompt


# ---------------------------------------------------------
# STEP 1: MULTI-DATASET DISCOVERY, SHUFFLING & TRAIN/TEST SPLIT
# ---------------------------------------------------------
def load_all_netlist_entries(test_ratio=0.20):
    all_files = []
    for folder in DATASET_FOLDERS:
        pattern = os.path.join(folder, "*.net")
        found = glob.glob(pattern)
        print(f"Found {len(found)} files in '{folder}'")
        all_files.extend(found)

    print(f"\nTotal netlists collected: {len(all_files)}")

    random.seed(RANDOM_SEED)
    random.shuffle(all_files)

    split_idx = int(len(all_files) * (1 - test_ratio))
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    print(f"Train samples: {len(train_files)} | Test samples: {len(test_files)}")

    training_samples = []
    for path in train_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                net_txt = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                net_txt = f.read()

        lines = [l for l in net_txt.splitlines() if l.strip()]
        cleaned = "\n".join(l for l in lines if l not in {"*", "."})

        target_class = None
        for cls in COMPONENT_CLASSES:
            if cls.lower() in path.lower():
                target_class = cls
                break

        if target_class:
            formatted_prompt = build_prompt(
                cleaned, target_component=target_class
            )
            training_samples.append({"text": formatted_prompt})

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

        target_class = None
        for cls in COMPONENT_CLASSES:
            if cls.lower() in path.lower():
                target_class = cls
                break

        if target_class:
            eval_samples.append((path, cleaned, target_class))

    return training_samples, eval_samples


print("1. Discovering and shuffling datasets...")
train_samples, eval_samples = load_all_netlist_entries()

hf_dataset = Dataset.from_dict({"text": [s["text"] for s in train_samples]})

# ---------------------------------------------------------
# STEP 2: QUANTIZATION & MODEL LOADING
# ---------------------------------------------------------
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
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)

# ---------------------------------------------------------
# STEP 3: LoRA FINE-TUNING
# ---------------------------------------------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    optim="paged_adamw_8bit",
    bf16=use_bf16,
    fp16=not use_bf16,
    save_strategy="epoch",
    report_to="none",
    disable_tqdm=False,
    logging_first_step=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)

print("\n2. Fine-tuning on combined dataset mixture...")
#trainer.train()

#adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
#trainer.model.save_pretrained(adapter_dir)
#tokenizer.save_pretrained(adapter_dir)
#print(f"LoRA Adapter saved to {adapter_dir}")

# ---------------------------------------------------------
# STEP 4: INFERENCE & JSON EXPORT FOR COMBINED EVALUATION
# ---------------------------------------------------------
print("\n3. Running inference across all netlists...")

# Reload trained adapter onto a fresh base model instance to avoid KV cache dtype pollution
from peft import PeftModel

adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")

# Free up memory from trainer instance
del trainer
del model
torch.cuda.empty_cache()

# Reload base model & load fine-tuned adapter
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    torch_dtype=compute_dtype,
)
eval_model = PeftModel.from_pretrained(base_model, adapter_dir)
eval_model.eval()

output_json_path = "llm_predictions_lora_combined.json"

with open(output_json_path, "w") as f:
    f.write("[\n")
    first = True

    pbar = tqdm(eval_samples, desc="Evaluating", unit="sample")

    for path, cleaned_netlist, gt_label in pbar:
        try:
            prompt = build_prompt(cleaned_netlist)
            inputs = tokenizer(prompt, return_tensors="pt").to(eval_model.device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=compute_dtype):
                    outputs = eval_model.generate(
                        **inputs, 
                        max_new_tokens=32, 
                        do_sample=False,
                        use_cache=False  # Prevents KV cache dtype mismatches in Gemma 2
                    )

            full_out = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            ).strip()

            pred_json = extract_json(full_out)
            pred = pred_json.get("prediction") if pred_json else None

            entry = {"file": path, "ground_truth": gt_label, "pred": pred}

            if not first:
                f.write(",\n")
            first = False

            json.dump(entry, f)
            f.flush()

            pbar.set_postfix(
                {"File": os.path.basename(path), "GT": gt_label, "Pred": pred}
            )

        except KeyboardInterrupt:
            print("\nInference interrupted by user. Saved partial JSON.")
            break

    f.write("\n]")

print(f"\nCompleted evaluation! Output exported to '{output_json_path}'.")