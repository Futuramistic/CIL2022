# SimpleRLCNN:
`python main.py -m=simplerlcnn -E=debugging -R=1 -s=0.5 -e=2 -b=2 -i=1 -v=8 -c=1 -d=original --patch_size=[100,100] --history_size=5 --max_rollout_len=1e4 --std=1e-3 --reward_discount_factor=0.99 --num_policy_epochs=4 --policy_batch_size=10 --sample_from_action_distributions=False --visualization_interval=1`

# SimpleRLCNNMinimal:
Old:
`python main.py -m=simplerlcnnminimal -E=debuggingMinimal -R=1 -s=0.01 -e=2 -b=2 -i=1 -v=8 -c=1 -d=original --patch_size=[100,100]  --rollout_len=1e3 --std=1e-3 --reward_discount_factor=0.99 --num_policy_epochs=4 --policy_batch_size=10 --sample_from_action_distributions=False --visualization_interval=1`
Updated:
`python3 main.py -m=simplerlcnnminimal -E=debuggingMinimal -R=1 -s=0.5 -e=2 -b=2 -i=4 -v=8 -c=1 -d=original --patch_size=[100,100] --rollout_len=160000 --std=[0.01,0.1] --reward_discount_factor=0.99 --num_policy_epochs=4 --policy_batch_size=10 --sample_from_action_distributions=True --visualization_interval=1`


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

# testing weightes samples (tf) via cranet
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
            "args": ["--model=deeplabv3", "--dataset=original", "--batch_size=2", "--split=0.03", "--evaluation_interval=1", "-E=DeepLabV3Adaboost", "--optimizer_or_lr=1e-4", "num_epochs=2", "--checkpoint_interval=1", "--hyper_seg_threshold=True", "--use_adaboost=True", "--adaboost_runs=2"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}

# Debugging tensorflow adaboost
`python main.py --model=attunet --dataset=original --batch_size=2 --split=0.03 --evaluation_interval=1 -E=UnetPPAdaboost --optimizer_or_lr=1e-4 --num_epochs=1 --checkpoint_interval=1 --use_adaboost=True --adaboost_runs=2`