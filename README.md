# CosmoFormer: Multi-Foreground Transformer

This repository contains the training and evaluation code for the CosmoFormer model, updated to handle multiple foreground signals.

## Running the Multi-Foreground Model: Step-by-Step

### Step 1: Place Raw Data
Create a directory for your raw data (by default `data/raw/10k_diff/`) and place the raw numpy dataset arrays inside it. If you use the default config, name your files as follows:
- `mixed_cls_64_10k_30GHz.npy` (Mixed 1 Array: True + FG1)
- `mixed_cls_64_10k_60GHz.npy` (Mixed 2 Array: True + FG2)
- `true_64_10k.npy` (Clean target array - true cmb)
- `fg1_64_10k.npy` (Foreground 1 target array)
- `fg2_64_10k.npy` (Foreground 2 target array)

*Note: If you choose to use different names or directories, you must update corresponding fields in the config file.*

### Step 2: Update Configuration
Before running the pipelines, update `configs/config_2layer.yaml` according to your dataset size and parameters. 

**Data Configuration (`data` section):**
- `raw_dir`: Path to the raw dataset directory (matches Step 1)
- `processed_dir`: Path where processed sequence data will be saved
- `mixed1_name`: Filename for the first mixed signal array
- `mixed2_name`: Filename for the second mixed signal array
- `true_name`: Filename for the clean target array
- `fg1_name`: Filename for the first foreground target array
- `fg2_name`: Filename for the second foreground target array

**Logging & Checkpoints:**
- `run_name`: Name for the MLflow run
- `checkpoint_dir`: Directory to save the best model checkpoints

### Step 3: Set up Environment
Ensure your conda environment is activated:
```bash
conda activate CosmoFormer
```

### Step 4: Prepare Data
Process the raw dataset arrays into structured `.npy` files suitable for training:
```bash
python -m src.data.prepare_data 
```

### Step 5: Train the Model
Start the training loop (which uses `config_2layer.yaml` by default). The model expects two mixed input arrays and predicts three targets.
```bash
python -m src.training.train
```

### Step 6: Evaluate the Model
Run evaluations on your trained model:
```bash
python -m src.eval.evaluate
```

---

## Model Inputs & Targets

**Inputs to Training (2 Arrays)**
- Mixed 1 Array (True + Foreground 1)
- Mixed 2 Array (True + Foreground 2)

**Predicted Targets (3 Arrays)**
- Clean Array (true cmb) `[Mean, Logvar]`
- Foreground 1 Array `[Mean, Logvar]`
- Foreground 2 Array `[Mean, Logvar]`

