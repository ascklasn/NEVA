
![header](https://capsule-render.vercel.app/api?type=soft&height=80&color=gradient&text=NEVA:%20A%20Unified%20Vision-Language%20Model%20for%20Precision%20Neuroblastoma%20Care&section=header&reversal=false&textBg=false&fontSize=23&fontAlign=50&animation=fadeIn)

### 👉[Interactive Demo](https://nevademo.netlify.app/)
---
### 👥 Authors

<details>
<summary>Click to expand author list</summary>

Jin Zhu¹⁰‡, Ruizhen Hu²‡, Sen Yang¹³‡, Qing Sun⁵‡, Zhenzhen Zhao¹, Juan Cao⁶, Peiying Pan⁷, Kun Wang⁷, Liyan Cui⁸,
Hongping Tang⁹, Qianqian Fang¹, Sijin Jiang¹, Linli Lei¹, Wenjian Zhang¹, Jiajun Xie¹, Shuo Kang¹, Dongyuan Xiao¹,
Ming Xiao¹², Xuan Zhai¹, Yuntao Jia¹, Junyang Chen², Wei Yuan⁴, Xiao Han⁴, Junhan Zhao¹¹, Xiyue Wang³‡
Yi Li²\*, [Jinxi Xiang](https://jinxixiang.com/)³\*, Biyue Zhu¹\*

> ‡ Equal contribution · *Corresponding authors: [Biyue Zhu](mailto:biyuezhu@hospital.cqmu.edu.cn), [Jinxi Xiang](mailto:xiangjx@stanford.edu), [Yi Li](mailto:liyi@sz.tsinghua.edu.cn)*

</details>

---

### 🏥 Affiliations

<details>
<summary>Click to expand affiliations</summary>

1. Children’s Hospital of Chongqing Medical University, Chongqing, China
2. Shenzhen International Graduate School, Tsinghua University, Shenzhen, China
3. Department of Radiation Oncology, Stanford University School of Medicine, Palo Alto, USA
4. College of Biomedical Engineering, Sichuan University, Chengdu, China
5. Peking University First Hospital, Beijing, China
6. Shenzhen Children’s Hospital, Shenzhen, China
7. Guiyang Maternal and Child Health Care Hospital, Guiyang, China
8. Inner Mongolia Maternity and Child Health Care Hospital, Hohhot, China
9. Shenzhen Maternity and Child Healthcare Hospital, Southern Medical University, Shenzhen, China
10. Department of Pathology, Chongqing Medical University, China
11. Department of Biomedical Informatics, Harvard Medical School, Boston, USA
12. Molecular Medicine Diagnostic and Testing Center, Chongqing Medical University, China
13. Ant Group

</details>



---

## Overview

<img align="right" valign="top" src="./NEVA_logo.png" width="250px" />

**NEVA** (**NE**uroblastoma **V**ision--language **A**I) is a unified foundation model trained on whole-slide histopathology images and pathology reports to perform 11 clinically essential tasks related to neuroblastoma diagnosis, biomarker prediction, and prognosis.

NEVA was trained and validated using the largest known multi-institutional neuroblastoma dataset to date:  

- **1,238 patients**  
- **1,419 pathology reports**  
- **3,593 whole-slide images (WSIs)**

It achieves robust performance over unimodal and conventional models in:  

- Classifying tumor **Risk Groups**, histologic **Subtypes**, **Mitotic-Karyorrhectic Index (MKI)**, and **Shimada classification**  
- Predicting molecular markers: **ALK**, **NMYC**, **CMYC**, **1p36**, and **11q23**  
- Forecasting **Progression-Free Survival (PFS)** and **Overall Survival (OS)**  

> NEVA is scalable, interpretable, and data-efficient — a promising AI framework for precision neuroblastoma care across diverse clinical settings.

![NEVA pipeline](NEVA_pipeline.png)  
*NEVA pipeline*

---

## 🔧 Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/ascklasn/NEVA.git
    cd NEVA
    ```

2. Create and activate the conda environment:

    ```bash
    conda env create -n NEVA -f environment.yml
    conda activate NEVA
    ```

---

## 🧪 WSI Preprocessing
The WSI preprocessing pipeline is adapted from [TRIDENT](https://github.com/mahmoodlab/TRIDENT).

We recommend directly using the official TRIDENT repository for tile extraction and feature extraction with Patch-Level and Slide-Level foundation models (e.g., UNI, CONCH, Virchow, CHIEF, MUSK, etc.).

Please refer to the official repository for setup and usage instructions.

1. Navigate to the preprocessing directory:

    ```bash
    cd ./1_process_wsi
    ```

2. Extract features using [FEATHER](https://github.com/mahmoodlab/MIL-Lab) (refer to the official repository of Trident)

   ```python
   python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./trident_processed --slide_encoder feather --mag 10 --patch_size 384
   ```

3. Select TOP-200 Patches 
   ```python
   python extract_patch_top.py --feat-root ./trident_processed/10x_384px_0px_overlap/slide_features_feather --output-dir ./top200_patches --wsi-root /mnt/hdd2_4t/hrz/神经母细胞瘤/test
   ```
   
4. Output contents:
    - `top200_patches/<case_id>_top_patches/`: A folder containing up to 200 JPG images of TOP patches.
---

<!-- ## 🔍 Feature Extraction

### 1. Patch-Level Feature Extraction (WSIs)

First, set paths:

```python
h5_root = "../1_process_wsi/outputs/patches"
slide_root = "../1_process_wsi/WSIs_source"
```

Then run:

```bash
python extract_image.py
```

> Output: `./outputs/images`

--- -->

<!-- ### 2. Textual Feature Extraction (Pathology Reports)

1. Place pathology reports in `pathology_report_en_eval.csv`.

2. Authenticate Hugging Face:

```python
my_token = 'hf_xxxxx'  # Replace with your token
login(my_token)
```

3. Run:

```bash
python extract_report.py
```

> Output: `./outputs/report_large` -->

---

## 🏋️ Training NEVA  

#### Repository Layout
- `train_mil.py` – Main training loop handling seed loops, fold generation, trainer setup, and result logging.
- `eval_mil.py` – Evaluation script after training.
- `models/neva.py` – NEVA model definition, LoRA target collection utilities, FiLM fusion, and CLAM integration.
- `models/clam.py`, `models/abmil.py` – MIL backbones.
- `wsi_dataset.py` – Lightning `WSIDataModule` preparing train/val/test splits and loading top-patch bags + reports.
- `workspace/` – Runtime artifacts (splits, models, reports). Generated automatically.
- `configs/*.yaml` – Task-specific training configs (classification/regression).

1. Navigate to `2_train_NEVA` directory:

   ```bash
   cd ./2_train_NEVA
   ```

2. Modify the `./configs/*yaml` files to update the following fields under `Data`:
    - `Data.dataframe`: Path to the base CSV file (e.g., `workspace/splits/cmyc.csv`), which must contain `patient_id`, `case_id`, label columns, and a `filename` column pointing to the `.pt` feature archive or case identifier.
    - `Data.image_dir`: Path to a directory containing image patches structured as `<case_id>_top_patches/patch_*.jpg`.
    - `Data.report_dir`: Path to the directory containing the English pathology report CSV file (e.g., `workspace/data/pathology_report_en.csv`). This file must include at least the `case_id` and `report_en` columns.

3. Configure Your Training/Inference Runs  
    Key fields in each YAML config file (see `configs/config_cls_cmyc.yaml` for an example):
    - `General.epochs`, `devices`, `metric`, and `precision`.
    - `Data.dataset_name` (`cls` for classification or `coxreg` for Cox regression), `dataframe`, `label_name`, and optionally `test_df`.
    - `Model.name` (defaults to `models.mmclassifier.MMClassifier`) along with its `params`.
    - Settings for `Optimizer` and `Loss`.
  
4. Training Workflow:
    Run a single configuration:
    ```bash
    python train_mil.py --csv_list cmyc
    ```
    `train_mil.py` will:
    -. Load `configs/config_cls_cmyc.yaml`.
    -. Generate grouped 5-fold splits (`outputs_config_cls_cmyc/cmyc_<seed>_folds.csv`).
    -. Instantiate `WSIDataModule` with the in-memory fold DataFrame.
    -. Build `MILModel` (which wraps `NEVA`) and attach LoRA modules.
    -. Train and evaluate each fold, saving checkpoints under `workspace/models/<task>/seed=<seed>_fold=<fold>/`.
    -. Append macro metrics to `workspace/results_MultiModal/<config>_results.json`.

    To run multiple tasks:
    ```bash
    python train_mil.py --csv_list alk cmyc mki
    ```

> Output: model weights saved in `./workspace/models/`

---

## 💾 Download Pretrained Model Weights and Perform Inference

1. Download the model weights and place them in:
`./3_evaluation/model_weights/`

    | Task             | Performance     | Weights                                                                                        |
    | ---------------- | --------------- | ---------------------------------------------------------------------------------------------- |
    | Risk Group       | AUROC = 0.806   | [Download](https://drive.google.com/file/d/1g75iUeCsTXae_J3csr2O7UUUXRTaDZ2P/view?usp=sharing) |
    | Subtype          | AUROC = 0.916   | [Download](https://drive.google.com/file/d/1rCxmsO5RNk-q8KLe4mlUW5dKhBpLqufJ/view?usp=sharing) |
    | MKI              | AUROC = 0.791   | [Download](https://drive.google.com/file/d/11aHCpRlqcdt2WQ4peiPxywIiOL7cjH-P/view?usp=sharing) |
    | Shimada          | AUROC = 0.823   | [Download](https://drive.google.com/file/d/1upOALcXuY6JYkdyPFydZwKTV6VLaog8D/view?usp=sharing) |
    | ALK              | AUROC = 0.764   | [Download](https://drive.google.com/file/d/1g3uDkHVAUFzW5657grXOmgDiD5D0gtWA/view?usp=sharing) |
    | NMYC             | AUROC = 0.924   | [Download](https://drive.google.com/file/d/1EU3C7845uZAbmcen4eqAqoqX8kuZhYin/view?usp=sharing) |
    | CMYC             | AUROC = 0.703   | [Download](https://drive.google.com/file/d/18QO3jdP9jcWsRfwGkj1DKnwOhTrzypM-/view?usp=sharing) |
    | 1p36 Deletion    | AUROC = 0.830   | [Download](https://drive.google.com/file/d/1v0G2Ytz_l9HmpiDzKtztfiTdkqcHkUN1/view?usp=sharing) |
    | 11q23 Deletion   | AUROC = 0.776   | [Download](https://drive.google.com/file/d/1dnPoJxA2LaZGQG0D6esHeLHaDwa1kPZK/view?usp=sharing) |
    | Overall Survival | C-index = 0.717 | [Download](https://drive.google.com/file/d/1Z_sPkAMqHHL6QGL5Bgmal7rVou2xZaDW/view?usp=sharing) |
    | PFS              | C-index = 0.645 | [Download](https://drive.google.com/file/d/14UsRMndaSZSnVJ7nmiYJdB2Jpvm4eza7/view?usp=sharing) |

2. Perform inference on your custom dataset using the pretrained weights.  

   1. Navigate to `./4_evaluation` directory:

        ```bash
        cd ./3_evaluation
        ```

   2. Run inference script:

       ```bash
       python inference.py --csv_list nmyc --dataset_type cls --image_dir ../1_process_wsi/top200_patches --eval_df path/to/your/eval_csv   
       ```

    > - `--dataset_type`: `cls` for classification, `reg` for Cox regression
    > - `--csv_list`: '1p36', '11q23', 'alk', 'cmyc', 'risk_group', 'mki', 'nmyc', 'shimada', 'subtype', 'os', 'pfs'

    > Output: The result of the model inference will be saved in `./eval_results/`

---

## 📊 Evaluation

Run model evaluation in:

```bash
./3_evaluation/Model_Evaluation.ipynb
```

---

## 👁️ Visualization

The `configs` directory contains two files:  

- A *CSV* file with metadata for the samples  
- A *YAML* configuration file for visualization settings

### CSV File

The CSV file should include the following columns:

- `patient_id`
- `case_id`
- `label`
- `filename`
- `resolution`

You can refer to examples like:   `./configs/PUFH/pfs.csv`  

### YAML File

The YAML file specifies general visualization parameters. Example files:  `./configs/PUFH/pfs.yaml`  

Important fields under the `General` section include:

- `proj_name`: Task name, e.g., `pfs`, `os`, `shimada`, `subtype`, `alk`, `1p36`, `11q23`
- `proj_type`: `cls` for classification, `reg` for Cox regression
- `df_path`: Path to the corresponding CSV file containing sample metadata
- `pretrained_dir`: Path to the pretrained model weights for visualization, you can get pretrained weights for visualization from [Google Drive](https://drive.google.com/drive/folders/14UIlOTtGQ7EsgDU0qnI7JkN6jvtCHGlB?usp=drive_link)
- `patch_risk_heatmap`: Enable patch-level risk heatmap visualization
- `patch_attention_heatmap`: Enable patch-level attention heatmap visualization

### Running Visualization

```bash
cd ./4_visualization
python draw.py
```

---
