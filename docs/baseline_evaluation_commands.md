# Commands for Baseline Evaluation

## CRANet

VRAM usage ca. 7GB with BS 4
One run takes ca. 3h with the specified number of epochs
-> e.g. 20 runs: 60 hours
reduce or increase if seen fit (try not to reduce)

`python3 main_hyperopt.py -s cranet_baseline_eval -n 100`

## DeepLabV3

VRAM usage ca. 7.5GB with BS 4
One run takes ca. 3h with the specified number of epochs

`python3 main_hyperopt.py -s deeplabv3_baseline_eval -n 100`

## SegFormer

VRAM usage ca. 21GB with BS 4, hence small batch sizes used in search space (currently only 2 and 4 can be sampled; add more if GPU has more VRAM)

One run takes ca. 3h with the specified number of epochs

`python3 main_hyperopt.py -s segformer_baseline_eval -n 100`