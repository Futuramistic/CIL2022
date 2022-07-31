# SimpleRLCNN:
`python main.py -m=simplerlcnn -E=debugging -R=1 -s=0.5 -e=2 -b=2 -i=1 -v=8 -c=1 -d=original --patch_size=[100,100] --history_size=5 --max_rollout_len=1e4 --std=1e-3 --reward_discount_factor=0.99 --num_policy_epochs=4 --policy_batch_size=10 --sample_from_action_distributions=False --visualization_interval=1`

# SimpleRLCNNMinimal:

Without supervision:
`python3 main.py --model=simplerlcnnminimal --experiment_name=debuggingMinimal --run_name=Test --split=0.98 --evaluation_interval=4 --checkpoint_interval=100000 --dataset=new_original --patch_size=[100,100] --rollout_len=200 --std=[0.01,0.1] --reward_discount_factor=0.99 --num_policy_epochs=4 --policy_batch_size=10 --sample_from_action_distributions=True --visualization_interval=1 --batch_size=2`
With supervision:
`python3 main.py --model=simplerlcnnminimalsupervised --experiment_name=debuggingMinimal --run_name='Supervision test' --dataset=new_original --split=0.827 --num_epochs=2 --batch_size=2 --evaluation_interval=4 --num_samples_to_visualize=9 --checkpoint_interval=1000 --dataset=new_original --patch_size=[400,400] --rollout_len=1000 --std=[0.01,0.1] --reward_discount_factor=0.99 --num_policy_epochs=4 --policy_batch_size=10 --sample_from_action_distributions=False --visualization_interval=1 --blobs_removal_threshold=0 --use_supervision=True`



# SimpleRLCNN with hyperopt:
`python main_hyperopt.py -s simple_cnn_1 -n 100`

# testing weighted samples (torch) via cranet
`python main.py -m cranet -d original -E TestingWeightedSamplingTorch -R run1 -s 0.01 -e 3 -b 2 -i 2 -w True`
Debugging:
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Weighted Sampling Debugging",
            "type": "python",
            "request": "launch",
            "program": "main.py",
            "args": ["-m=cranet", "-E=TestingWeightedSamplingTorch", "-R=run1", "-s=0.03", "-e=3", "-b=2", "-i=2", "-w=True", "-d=original"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}

# testing weightes samples (tf) via unetexp
`python main.py --model=unetexp --dataset=new_original_aug_6 -E=TestingWeightedSamplingTf -R=run1 --split=0.03 --batch_size=2 --blobs_removal_threshold=0 -i=1 --use_geometric_augmentation=True --use_color_augmentation=False --dropout=0.0 --num_epochs=3 --optimizer_or_lr=0.001 --hyper_seg_threshold=False -w=True`

# Debugging with launch.json
{
    // Verwendet IntelliSense zum Ermitteln möglicher Attribute.
    // Zeigen Sie auf vorhandene Attribute, um die zugehörigen Beschreibungen anzuzeigen.
    // Weitere Informationen finden Sie unter https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Reinforcement Learning Debugging",
            "type": "python",
            "request": "launch",
            "program": "main.py",
            "args": ["-m=simplerlcnn", "-E=debugging", "-R=1", "-s=0.2", "-e=2", "-b=2", "-i=1", "-v=8", "-c=1", "-d=original", "--patch_size=[100,100]", "--history_size=5", "--max_rollout_len=10", "--std=1e-3", "--reward_discount_factor=0.99", "--num_policy_epochs=4", "--policy_batch_size=10", "--sample_from_action_distributions=False", "--visualization_interval=1"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}
# Deeplabv3 adaboost
`python main.py --model=deeplabv3 --dataset=original --batch_size=2 --split=0.03 --evaluation_interval=1 -E=DeepLabV3Adaboost --optimizer_or_lr=1e-4 --num_epochs=2 --checkpoint_interval=1 --hyper_seg_threshold=True --use_adaboost=True --adaboost_runs=2 --evaluate --apply_sigmoid=False --blob_threshold=0.0`

# Debugging adaboost with launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Adaboost Debugging",
            "type": "python",
            "request": "launch",
            "program": "main.py",
            "args": ["--model=unet", "--dataset=original", "--batch_size=2", "--split=0.02", "--evaluation_interval=1", "-E=AdaTestingUnet", "--use_adaboost=True", "--adaboost_runs=2", "--evaluate"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}

# Debugging torch predictor via launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Adaboost Debugging",
            "type": "python",
            "request": "launch",
            "program": "main.py",
            "args": ["--model=unet", "--dataset=original", "--batch_size=2", "--split=0.02", "--evaluation_interval=1", "-E=AdaTestingUnet", "--use_adaboost=True", "--adaboost_runs=2", "--evaluate"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}


# DeepLabV3 Run on Euler with Adaboost
`bsub -n 1 -W 75:00 -R "rusage[ngpus_excl_p=1, mem=10000]" -R "select[gpu_model0==NVIDIAGeForceGTX1080]" "python main.py -m=deeplabv3 --use_adaboost=True --adaboost_runs=10 --dataset=original_split_1 -E DeepLabV3AdaboostFinal --split=0.827 --num_epochs=75 --checkpoint_interval=25 --hyper_seg_threshold=True --optimizer_or_lr=0.0002 --use_geometric_augmentation=True --use_color_augmentation=True"`

# SegFormer Run on Euler with Adaboost
`bsub -n 1 -W 75:00 -R "rusage[ngpus_excl_p=1, mem=25000]" -R "select[gpu_model0==NVIDIAGeForceGTX1080]" "python main.py -m=segformer --use_adaboost=True --adaboost_runs=10 --dataset=original_split_1 -E SegFormerAdaboost12345 --split=0.827 --num_epochs=260 --checkpoint_interval=100000 --hyper_seg_threshold=True --optimizer_or_lr=0.0005 --batch_size=2 --use_geometric_augmentation=True --use_color_augmentation=True --blobs_removal_threshold=0 --hyper_seg_threshold=True"`

# UnetExp on Euler with Adaboost (sadly only bs=1 possible)
`bsub -n 1 -W 75:00 -R "rusage[ngpus_excl_p=1, mem=28000]" -R "select[gpu_model0==NVIDIAGeForceGTX1080]" "python main.py --model=unetexp --dataset=new_original_aug_6 --split=0.971 --batch_size=2 --blobs_removal_threshold=0 --use_geometric_augmentation=True --use_color_augmentation=False --dropout=0.0 --num_epochs=100 --optimizer_or_lr=0.001 --hyper_seg_threshold=False --input_shape=[400,400,3] --architecture='vgg' --experiment_name=UnetExpAdaboost --normalize=True --dropout=0.0 --kernel_regularizer=\(None\) --load_checkpoint_path=sftp://mlflow_user:waiMohu749@algvrithm.com:22/mlruns/169/37ab12b7fd784ab2931a9010e082027d/artifacts/checkpoints/cp_best_f1.ckpt --use_adaboost=True --adaboost_runs=10"`

# print Adaboost checkpoints
go to checkpoint dir
python:
import pickle
checkpoint_paths = pickle.load(open("checkpoints.pkl", "rb"))
print(checkpoint_paths)