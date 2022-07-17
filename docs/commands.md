# SimpleRLCNN:
`python main.py -m=simplerlcnn -E=debugging -R=1 -s=0.5 -e=2 -b=2 -i=1 -v=8 -c=1 -d=original --patch_size=[100,100] --history_size=5 --max_rollout_len=1e4 --std=1e-3 --reward_discount_factor=0.99 --num_policy_epochs=4 --policy_batch_size=10 --sample_from_action_distributions=False --visualization_interval=1`

# SimpleRLCNNMinimal:
Old:
`python main.py -m=simplerlcnnminimal -E=debuggingMinimal -R=1 -s=0.01 -e=2 -b=2 -i=1 -v=8 -c=1 -d=original --patch_size=[100,100]  --rollout_len=1e3 --std=1e-3 --reward_discount_factor=0.99 --num_policy_epochs=4 --policy_batch_size=10 --sample_from_action_distributions=False --visualization_interval=1`
Updated:
`python3 main.py -m=simplerlcnnminimal -E=debuggingMinimal -R=1 -s=0.5 -e=2 -b=2 -i=4 -v=8 -c=1 -d=original --patch_size=[100,100] --rollout_len=160000 --std=[0.01,0.1] --reward_discount_factor=0.99 --num_policy_epochs=4 --policy_batch_size=10 --sample_from_action_distributions=True --visualization_interval=1`
With supervision:
`python3 main.py --model=simplerlcnnminimalsupervised --experiment_name=debuggingMinimal --run_name='Supervision test' --dataset=new_original --split=0.827 --num_epochs=2 --batch_size=2 --evaluation_interval=4 --num_samples_to_visualize=9 --checkpoint_interval=1000 --dataset=new_original --patch_size=[400,400] --rollout_len=1000 --std=[0.01,0.1] --reward_discount_factor=0.99 --num_policy_epochs=4 --policy_batch_size=10 --sample_from_action_distributions=False --visualization_interval=1 --blobs_removal_threshold=0 --use_supervision=True`

# SimpleRLCNN with hyperopt:
`python main_hyperopt.py -s simple_cnn_1 -n 100`

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