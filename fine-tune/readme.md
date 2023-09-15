git clone stanford-alpaca: https://github.com/tatsu-lab/stanford_alpaca

move the scripts to the main folder of stanford-alpaca

if you get OOM when saving the model, change 
envs/your-env-name/lib/python3.10/site-packages/torch/distributed/fsdp/_state_dict_utils.py

from state_dict[fqn] = state_dict[fqn].clone().detach() to state_dict[fqn] = state_dict[fqn].cpu().clone().detach()