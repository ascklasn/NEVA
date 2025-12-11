# NEVA LoRA: Neuroblastoma Multi-Modal MIL with Adaptive LoRA

This repository contains the training code used for multi-modal weakly supervised learning on whole-slide histopathology. We pair Musk's BEiT3-based image encoder with pathology report embeddings and train a MIL classifier (CLAM) using cross-validation. The core contribution is a configurable LoRA injection scheme (`models/neva.py`) tailored to BEiT3 blocks, enabling lightweight adaptation of the vision backbone while keeping most parameters frozen.

## Highlights
- **Flexible LoRA targeting** – `collect_lora_targets(...)` lets you select attention heads vs feed-forward projections, restrict to the last _N_ transformer blocks, and optionally filter multiway (A/B) heads.
- **MIL + Vision-Language fusion** – Combines Musk features, CLAM MIL pooling, and fusion of pathology reports.
- **Reproducible cross-validation** – `train_mil.py` builds stratified GroupKFold splits per configuration, logs fold distributions, and saves results per seed.
- **Config-driven experiments** – YAML files under `configs/` define dataset paths, model hyperparameters, optimizer settings, and optional held-out test cohorts.

## Repository Layout
- `train_mil.py` – Main training loop handling seed loops, fold generation, trainer setup, and result logging.
- `eval_mil.py` – Evaluation script after training.
- `models/neva.py` – NEVA model definition, LoRA target collection utilities, FiLM fusion, and CLAM integration.
- `models/clam.py`, `models/abmil.py` – MIL backbones.
- `wsi_dataset.py` – Lightning `WSIDataModule` preparing train/val/test splits and loading top-patch bags + reports.
- `workspace/` – Runtime artifacts (splits, models, reports). Generated automatically.
- `configs/*.yaml` – Task-specific training configs (classification/regression).

## Setup
1. **Environment**
   ```bash
   conda create -n neva_lora python=3.10
   conda activate neva_lora
   pip install -r requirements.txt  # create this from environment if needed
   ```
2. **Musk & PEFT dependencies** – Ensure `timm`, `peft`, `huggingface_hub`, `pytorch-lightning`, and `transformers` are installed.
3. **Hugging Face access** – Replace the placeholder token in `models/neva.py` with your own or load it via environment variable:
   ```python
   from huggingface_hub import login
   login(os.environ["HF_TOKEN"])
   ```
4. **Data directories** – Update `configs/*.yaml` (e.g., `Data.image_dir`, `Data.report_dir`) to point to your patch and report locations.

## Data Preparation
- **WSI patches** – Each case requires a folder `<case_id>_top_patches/patch_*.jpg`. The dataset enforces up to 50 patches per bag with zero-padding as needed.
- **Report CSV** – `workspace/data/pathology_report_en.csv` must contain at least `case_id` and `report_en` columns.
- **Splits CSV** – Base CSVs (e.g., `workspace/splits/cmyc.csv`) include `patient_id`, `case_id`, label columns, and `filename` pointing to the `.pt` feature archive or case id.

## Configuring Runs
Key fields in each YAML (see `configs/config_cls_cmyc.yaml`):
- `General.epochs`, `devices`, `metric`, `precision`.
- `Data.dataset_name` (`cls` or `coxreg`), `dataframe`, `label_name`, optional `test_df`.
- `Model.name` (defaults to `models.mmclassifier.MMClassifier`) and its `params`.
- `Optimizer` and `Loss` settings.

Override the seed/fold tuple via `seed&fold: [123, 2]` to resume a particular experiment.

## Training Workflow
Run a single configuration:
```bash
python train_mil.py --csv_list cmyc
```
`train_mil.py` will:
1. Load `configs/config_cls_cmyc.yaml`.
2. Generate grouped 5-fold splits (`outputs_config_cls_cmyc/cmyc_<seed>_folds.csv`).
3. Instantiate `WSIDataModule` with the in-memory fold DataFrame.
4. Build `MILModel` (which wraps `NEVA`) and attach LoRA modules.
5. Train and evaluate each fold, saving checkpoints under `workspace/models/<task>/seed=<seed>_fold=<fold>/`.
6. Append macro metrics to `workspace/results_MultiModal/<config>_results.json`.

To run multiple tasks:
```bash
python train_mil.py --csv_list alk cmyc mki
```

## LoRA Design (models/neva.py)
- `collect_lora_targets` drives target discovery:
  - `mode`: choose among `"qv"`, `"qk"`, `"attn"`, `"ffn"`, `"attn+ffn"`, `"all"`.
  - `num_last_layers`: restricts modules to the last _N_ BEiT3 encoder layers.
  - `include_multiway`: include or drop `.A/.B` heads.
  - `as_regex`: return the compiled regex when integrating with PEFT configs that accept pattern strings.
- `get_musk_lora` creates BEiT3, loads the Musk checkpoint, selects LoRA targets (default: Q & V projections for the last 3 layers), and builds a `peft.LoraConfig` (`task_type=FEATURE_EXTRACTION`).
- `NEVA.__init__` loads the CLAM MIL module, applies LoRA-augmented Musk features, encodes reports, applies FiLM fusion, and exposes per-modality classifiers.
- During training only LoRA parameters and head layers remain trainable, enabling efficient finetuning on small cohorts.

### Customizing LoRA
1. Change `mode`/`num_last_layers` in `get_musk_lora` to trade capacity vs efficiency.
2. Switch to regex-based targeting by passing `as_regex=True` and setting `target_modules=re.compile(...)` support in PEFT.
3. Increase `r`/`lora_alpha` for higher-rank adapters; ensure GPU memory is sufficient.
4. Set `include_multiway=False` to prune mirrored heads when sharing weights across `.A`/`.B` branches.

## Extending / Debugging
- **Logging** – Training logs stream to stdout; adjust the `logging` level in `train_mil.py` if needed.
- **Lightning Trainer tweaks** – Update callbacks or precision in `prepare_trainer` to match your hardware.
- **Experiments** – Duplicate a YAML file, change data paths or LoRA settings, and rerun `train_mil.py` with the new name.

## Troubleshooting
- `pandas.errors.EmptyDataError`: Ensure fold CSVs are generated and not overwritten mid-run. `train_mil.py` now keeps fold DataFrames in memory to avoid race conditions.
- Missing Hugging Face token: export `HF_TOKEN` before running training or comment out the direct `login` call and handle it externally.
- CUDA OOM: Lower `batch_size`, reduce `lora_alpha`, or restrict LoRA to fewer layers.
