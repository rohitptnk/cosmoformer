# CosmoFormer

CosmoFormer is a 1D Transformer Autoencoder model designed for predicting clean signal and noise components from mixed signals.

## 1. Setup and Environment
First, ensure you are in an appropriate environment. For example:
```bash
conda create -n CosmoFormer python=3.10
conda activate CosmoFormer
pip install -r requirements.txt
```

## 2. Directory Structure & Raw Data
Your raw data must be placed inside the `data/raw/<dataset_name>/` folder. For instance, if your dataset is named `10k_diff`, you would put it under `./data/raw/10k_diff/`.

You must provide **three** `.npy` files (Numpy arrays) with the shape `(N, seq_len)`:
1. **Mixed Data (Input X):** The combined signal.
2. **True Signal (Target Y_true):** The clean/true signal you want to recover.
3. **Noise Signal (Target Y_noise):** The isolated noise.

**Example names:**
- `mixed_cls_64_10k.npy`
- `true_cls_64_10k.npy`
- `noise_cls_64_10k.npy`

## 3. Configuration
All parameters are managed via a configuration file located in the `configs/` directory (e.g., `configs/config_2layer.yaml`).

Before running any script, open your `config.yaml` file and ensure the `data` section matches your raw data directory and filenames:
```yaml
data:
  raw_dir: ./data/raw/10k_diff
  processed_dir: ./data/processed/10k_diff
  mixed_cls_name: mixed_cls_64_10k.npy
  true_cls_name: true_cls_64_10k.npy
  noise_cls_name: noise_cls_64_10k.npy
  seq_len: 127 # Make sure this matches the 2nd dimension of your arrays

logging:
  mlflow:
    run_name: cosmoformer_2layer_10k_diff_50epochs

checkpoint:
    checkpoint_dir: ./experiments/cosmoformer_2layer_10k_diff_50epochs
```

*Note: The scripts below currently expect the config file path to be hardcoded. Before running, update the `config_path` variable at the bottom (in the `if __name__ == "__main__":` block) of `prepare_data.py`, `train.py`, and `evaluate.py` to point to your specific config file.*

## 4. Execution Workflow
Run the following commands in this exact order from the root directory of the repository:

### Step 1: Prepare Data
```bash
python -m src.data.prepare_data
```
**What it does:** Reads the raw data, creates an 80/20 train-validation split, standardizes the scales, and saves the formatted files into your `processed_dir`.

### Step 2: Train Model
```bash
python -m src.training.train
```
**What it does:** Loads the processed data, initializes the model, and runs the training loop. It saves the best model checkpoint to your `checkpoint_dir` and logs training metrics locally via MLflow (stored in `./mlruns`).

### Step 3: Evaluate Model
```bash
python -m src.eval.evaluate
```
**What it does:** Loads the best model from your `checkpoint_dir` and evaluates it on the validation set. It prints out the MSE and generates comparative plots (saving them to `<checkpoint_dir>/eval_outputs/`).