# SimpleRLCNN:
`python main.py -m=simplerlcnn -E=debugging -R=1 -s=0.5 -e=2 -b=2 -i=1 -v=8 -c=1 -d=original --patch_size=[100,100] --history_size=5 --max_rollout_len=1e4 --std=1e-3 --reward_discount_factor=0.99 --num_policy_epochs=4 --policy_batch_size=10 --sample_from_action_distributions=False --visualization_interval=1`

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