Evaluation wrapper for results/PPO_anchored_20261124_a03/model_iter050.pt.

The fixed-route and diverse start-lane evaluators read a BC training folder
(config.json, configs, dataset path). The PPO actor has the same architecture
and observation configuration as the BC model it was fine-tuned from, so this
folder holds that BC run's configuration files with the PPO weights as model.pt.
