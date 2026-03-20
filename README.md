# CosmoFormer: Multi-Foreground Transformer

This repository contains the training and evaluation code for the CosmoFormer model, updated to handle multiple foreground signals.

## Running the Multi-Foreground Model: Step-by-Step

### Step 1: Place Raw Data
Create a directory for your raw data (by default `data/raw/10k_diff/`) and place the raw numpy dataset arrays inside it. If you use the default config, name your files as follows:
- `mixed_cls_64_10k_30GHz.npy` (Mixed 1 Array: True + Freq 1)
- `mixed_cls_64_10k_100GHz.npy` (Mixed 2 Array: True + Freq 2)
- `mixed_cls_64_10k_freq3.npy` (Mixed 3 Array: True + Freq 3)
- `mixed_cls_64_10k_freq4.npy` (Mixed 4 Array: True + Freq 4)
- `true_64_10k.npy` (Clean target array - true cmb)
- `freq1_64_10k.npy` (Frequency 1 target array)
- `freq2_64_10k.npy` (Frequency 2 target array)
- `freq3_64_10k.npy` (Frequency 3 target array)
- `freq4_64_10k.npy` (Frequency 4 target array)

*Note: If you choose to use different names or directories, you must update corresponding fields in the config file.*

### Step 2: Update Configuration
Before running the pipelines, update `configs/config_2layer.yaml` according to your dataset size and parameters. 

**Data Configuration (`data` section):**
- `raw_dir`: Path to the raw dataset directory (matches Step 1)
- `processed_dir`: Path where processed sequence data will be saved
- `mixed1_name` to `mixed4_name`: Filenames for the four mixed signal arrays
- `true_name`: Filename for the clean target array
- `freq1_name` to `freq4_name`: Filenames for the four frequency target arrays

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
Start the training loop (which uses `config_2layer.yaml` by default). The model expects four mixed input arrays and predicts five targets.
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

**Inputs to Training (4 Arrays)**
- Mixed 1 Array (True + Foreground at freq1)
- Mixed 2 Array (True + Foreground at freq2)
- Mixed 3 Array (True + Foreground at freq3)
- Mixed 4 Array (True + Foreground at freq4)

**Predicted Targets (5 Arrays)**
- Clean Array (true cmb) `[Mean, Logvar]`
- Net fg at freq 1 Array `[Mean, Logvar]`
- Net fg at freq 2 Array `[Mean, Logvar]`
- Net fg at freq 3 Array `[Mean, Logvar]`
- Net fg at freq 4 Array `[Mean, Logvar]`

To do

- [ ] Find freqency of lite bird and echo
- [ ] Find out how to add Detector noise
- [ ] Dusts to be added
    - [*] synchrotron 
    - [ ] thermal dust
    - [ ] free free
- [ ] Train and evaluate for 100 Ghz single head

- [ ] Errors to be added to evaluation
    - [ ] r2 score
    - [ ] mse
    - [ ] correlation
    - [ ] structural similarity index


- [ ] Within 30-150GHZ choose 5 freq with low detector noise from echo or litebird and 
    - [ ] Make the model for 3fg + det noise
