# SimpleRLCNN:
`python main.py -m simplerlcnn -E debugging -R 1 -s 0.5 -e 2 -b 2 -i 1 -v 8 -c 1 -d original -p --patch_size=[100,100] --history_size=5 --max_rollout_len=1e6 --std=1e-3 --reward_discount_factor=0.99 --num_policy_epochs=4 --policy_batch_size=10 --sample_from_action_distributions=False`


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
            "args": ["-m=simplerlcnn", "-E=debugging", "-R=1", "-s=0.5", "-e=2", "-b=2", "-i=1", "-v=8", "-c=1", "-d=original", "--patch_size=[100,100]", "--history_size=5", "--max_rollout_len=1e6", "--std=1e-3", "--reward_discount_factor=0.99", "--num_policy_epochs=4", "--policy_batch_size=10", "--sample_from_action_distributions=False"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}