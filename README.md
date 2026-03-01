CosmoFormer Training Details

### Change config.yaml
Change these file names in config.yaml according to dataset_size:
- raw_dirprocessed dir
- mixed_cls_name
- true_cls_name
- noise_cls_name
- run_name
- checkpoint_dir


Run 
```
conda activate CosmoFormer
python -m src.data.prepare_data 
python -m src.training.train
python -m src.eval.evaluate
```


Inputs to training
- Mixed 1 Array (True + Foreground 1)
- Mixed 2 Array (True + Foreground 2)
Target
- Clean Array (true cmb) [Mean, Logvar]
- Foreground 1 Array [Mean, Logvar]
- Foreground 2 Array [Mean, Logvar]

