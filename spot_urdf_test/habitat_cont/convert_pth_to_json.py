import torch
from collections import OrderedDict
import json

print('Loading...')
state_dict = torch.load('/home/joanne/repos/dynamics_aware_navigation_fair/pytorch_sac_private/ddppo_policies/ckpt.12.pth', map_location='cuda')
print('Done loading model.')
print('Converting...')
actual_dict = OrderedDict()
for k, v in state_dict["state_dict"].items():
	actual_dict[k] = v.tolist()
with open('spot_sliding_12.json', 'w') as f:
	json.dump(actual_dict, f)
print('Done.')