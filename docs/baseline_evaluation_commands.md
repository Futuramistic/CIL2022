# Commands for Baseline Evaluation

## CRANet

VRAM usage ca. 7GB with BS 4
One run takes ca. 3h with the specified number of epochs
-> 20 runs: 60 hours
reduce or increase if seen fit (try not to reduce)

`python3 main_hyperopt.py -s cranet_baseline_eval -n 20`

## DeepLabV3

VRAM usage ca. 7.5GB with BS 4
One run takes ca. 3h with the specified number of epochs

### Split 1 (`new_original` / `original_split_1`)
```
python3 main.py --model=deeplabv3 --dataset=original_split_1 --batch_size=4 --split=0.827 -E=DeepLabV3 --optimizer_or_lr=1e-4 --num_epochs=500 --checkpoint_interval=250 --hyper_seg_threshold=True '--run_name=Baseline evaluation; BS 4' --use_geometric_augmentation=True --use_color_augmentation=True
```

### Split 2 (`original_split_2`)
- same, but `--dataset=original_split_2` instead of `--dataset=original_split_1`

### Split 3 (`original_split_3`)
- same, but `--dataset=original_split_3` instead of `--dataset=original_split_1`