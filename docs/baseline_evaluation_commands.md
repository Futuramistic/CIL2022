# Commands for Baseline Evaluation

## CRANet

VRAM usage ca. 7GB with BS 4

### Split 1 (`new_original` / `original_split_1`)
```
python3 main.py --model=cranet --dataset=original_split_1 --batch_size=4 --split=0.827 -E=CRA_Net --num_epochs=300 --checkpoint_interval=250 --hyper_seg_threshold=True '--run_name=Baseline evaluation; BS 4; LR 5e-3; LR scheduler step size adapted to account for smaller DS' --use_geometric_augmentation=True --use_color_augmentation=True
```

### Split 2 (`original_split_2`)
- same, but `--dataset=original_split_2` instead of `--dataset=original_split_1`

### Split 3 (`original_split_3`)
- same, but `--dataset=original_split_3` instead of `--dataset=original_split_1`


## DeepLabV3

VRAM usage ca. 7.5GB with BS 4

### Split 1 (`new_original` / `original_split_1`)
```
python3 main.py --model=deeplabv3 --dataset=original_split_1 --batch_size=4 --split=0.827 -E=DeepLabV3 --optimizer_or_lr=1e-4 --num_epochs=500 --checkpoint_interval=250 --hyper_seg_threshold=True '--run_name=Baseline evaluation; BS 4' '--run_name=Baseline evaluation; BS 4; LR 1e-4; ReduceLROnPlateau' --use_geometric_augmentation=True --use_color_augmentation=True
```

### Split 2 (`original_split_2`)
- same, but `--dataset=original_split_2` instead of `--dataset=original_split_1`

### Split 3 (`original_split_3`)
- same, but `--dataset=original_split_3` instead of `--dataset=original_split_1`