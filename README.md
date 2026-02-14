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