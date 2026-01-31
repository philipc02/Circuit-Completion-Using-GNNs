# Circuit-Completion-Using-GNNs

This repository contains the code developed for the Bachelor’s thesis **“A Study on Graph Representations for GNN-based Analog Circuit Completion”**. The project investigates how different circuit graph representations influence the performance of Graph Neural Networks (GNNs) on circuit completion tasks, including component classification and (optionally) link prediction.

The repository builds on and extends an existing GNN-based circuit completion baseline (FEGIN from https://github.com/Anwar-Said/Circuit-Completion-Using-GNNs), introducing multiple graph representations, a multitask learning setup for pin-levle component classification and link prediction, and tools for evaluation.

---

## Repository Structure (High-Level)

- `component_classification_with_link_prediction_head/` – Main training and evaluation code (FEGIN & MultiTaskFEGIN)
- `graph_parsers/` – Netlist parsing, graph construction and graph analysis scripts
- `llm_inference/` – LLM-based circuit completion (zero-shot inference & evaluation)

---

## Data Preparation

### Graph Parsing

Selected netlist datasets (*LTSpice demos*, *LTSpice examples*, upcoming: *AMSNet*) and their parsed graphs are already included in the repository.

To parse additional netlists:

1. Place your netlists (`.net`, `.cir`, or `.sp`) in a new folder under `graph_parsers/`.
2. Choose the appropriate parser script depending on the desired circuit graph representation:
   - `netlist_parser_component_component.py`
   - `netlist_parser_component_net.py`
   - `netlist_parser_component_pin.py`
   - `netlist_parser_component_pin_net.py`
3. In the selected script, edit the `main()` function:
   - Set the **input folder** to your netlist directory
   - Set the **output folder** to `graphs_<dataset_name>/graphs_<representation>`
4. Run the script. Parsed graphs will be stored as NetworkX `.pickle` files in the output directory.

The parser prints basic dataset statistics such as class distribution to stdout.

#### Graph Statistics

For additional statistics, run the analysis script in:
```
graph_parsers/graph_analysis/
```
after updating the graph folder path in the script’s `main()` function.

---

### Dataset Alignment and Splits

To ensure comparability across representations, only circuits common to *all* representations are used.

1. Navigate to:
```
component_classification_with_link_prediction_head/evaluation/FEGIN/
```
2. Edit and run `prepare_all_datasets.py`:
   - Set the base graph folder to `graphs_<dataset_name>`
   - Update the output path for the dataset file
3. The script:
   - Identifies common circuits across all representations
   - Creates an 80:20 train–test split
   - Stores the result as:
     ```
     data/<dataset_name>_dataset.pkl
     ```

---

## Model Training

Model training is handled via:
```
component_classification_with_link_prediction_head/evaluation/FEGIN/main.py
```

### Key Arguments

- `--data` – Dataset name
- `--representation` – Graph representation (default: `component_pin_net`)
- `--model` – Model type:
  - `FEGIN` (default): component classification only (all representations)
  - `MultiTaskFEGIN`: component classification + link prediction (pin-level representations only)

#### Learning Parameters

Applicable to both models:
- `--layers`
- `--hiddens`
- `--emb_size`
- `--epochs`
- `--batch_size`
- `--lr`

Link prediction–specific (MultiTaskFEGIN only):
- `--lambda_node`
- `--lambda_edge`
- `--neg_sampling_ratio`

Other:
- `--reprocess` – Reprocess raw data instead of using cached `.pt` files (cached files can also be manually deleted from `data/processed/`)

### Example Command

```
python3 main.py \
  --model MultiTaskFEGIN \
  --data ltspice_examples \
  --representation component_pin_net \
  --layers 2 \
  --hiddens 16 \
  --emb_size 32 \
  --lr 0.001 \
  --batch_size 8 \
  --epochs 100 \
  --reprocess
```

### Implementation Notes

Important implementation files (in `kernel/`):

- `circuit_datasets.py` – Dataset creation for FEGIN
- `multitask_dataset.py` – Dataset creation for MultiTaskFEGIN
- `FEGIN.py`, `multitask_FEGIN.py` – Model architectures and forward passes
- `train_eval.py`, `multitask_train_eval.py` – Training and evaluation loops

---

## Hyperparameter Search

Hyperparameter search is supported for:
- Component classification only: `hyperparameter_search.py`
- Multitask learning (component classification and link prediction): `multitask_hyperparameter_search.py`

### Search Configuration

- `--search_method`:
  - `grid` – Exhaustive grid search
  - `random` – Random search
- `--n_trials` – Number of random trials (random search only)
- `--reps` – Subset of representations to evaluate (default: all)
- `--output_dir` – Directory for logs and results (created automatically)

### Result Analysis

Use:
- `analyze_results_FEGIN.py`
- `analyze_results_MultiTaskFEGIN.py`

After updating the input/output paths in the script’s `main()` and `load_results()` functions. Generated outputs include:
- Comparison chart: Best weighted F1 score per representation
- Heatmaps for layer/hidden-size combinations
- Hyperparameter sensitivity analysis

### Manual Inference

To inspect a trained model’s predictions, run:
```
inference.py
```
after updating the `model_path` variable in the script.

---

## LLM-Based Circuit Completion

### Partial Netlist Generation

Partial netlists used as LLM prompts are generated via:
```
llm_inference/create_labeled_netlists.py
```

- Currently supports `.net` files
- Update `SOURCE_DIR` to your netlist directory
- Outputs labeled partial netlists to:
  ```
  llm_inference/partial_netlists/
  ```

### LLM Inference

Run:
```
llm_inference/run_llm_inference.py
```

This performs zero-shot circuit completion using **Meta-Llama-3.1-8B-Instruct**. Predictions are stored in:
```
llm_predictions.json
```

Inference can be safely interrupted (`Ctrl+C`) as intermediate results remain valid.

### Evaluation

Evaluate predictions with:
```
llm_inference/evaluate_llm_predictions.py
```

A classification report is printed to stdout.

---

## HeteroConv Experiments

HeteroConv-based experiments are implemented in a separate repository:

https://github.com/philipc02/circuit-gnn

Overview:

1. Parse netlists using the scripts in `netlist_parser/`
2. In `data/dataset_splits.py`, set:
   - `source_folder` to the directory containing `.pickle` graph files
   - `output_root` accordingly
3. Run the script to create dataset splits
4. Train models using `model_heteroconv.py`
   - Uncomment `create_heterodata_obj()` in `main()` (only required once)
   - Adjust input/output paths

An inference script is also provided:
```
gnn_models/component_classification/inference.py
```
This script can be executed after updating the `model_path` variable in `run_inference()`.

---

## Disclaimer

This repository is a fork of:

https://github.com/Anwar-Said/Circuit-Completion-Using-GNNs

Several files and directories, particularly within:
   - `component_classification_with_link_prediction_head/`
   - `link_prediction/`
   - `component_classification_with_link_prediction_head/evaluation/FEGIN/`
   - `kernel/` and `modules/` 
originate from the base repository and are **not directly relevant** to the thesis implementation. They were retained to avoid breaking legacy dependencies and may be cleaned up in future revisions.


