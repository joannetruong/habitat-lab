import matplotlib.pyplot as plt
import argparse
import os
from collections import defaultdict

import habitat
import numpy as np
import quaternion
import torch
from evaluate_coda import load_model
from gym.spaces import Dict as SpaceDict
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations.utils import (images_to_video,
                                                observations_to_image)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.utils.common import action_to_velocity_control
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.config.default import get_config
from habitat_sim import geo
from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
from PIL import Image

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--cfg-path", type=str, required=True)
    args = parser.parse_args()
    # args.cfg_path = '/coc/testnvme/jtruong33/habitat-cont-v2/habitat-lab/habitat_baselines/config/pointnav/ddppo_pointnav_spot_coda.yaml'
    # args.cfg_path = '/coc/pskynet3/jtruong33/develop/flash_results/cont_ctrl_results_v2/spot_collision_0.1_nosliding_visual_encoder_nccl/ddppo_pointnav_spot_eval.yaml'
    # args.cfg_path = '/Users/joanne/repos/habitat_spot_v2/habitat-lab/habitat_baselines/config/pointnav/ddppo_pointnav_spot.yaml'
    config = get_config(args.cfg_path)
    # config.defrost()
    # config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
    # config.TASK_CONFIG.TASK.MEASUREMENTS.append("SOFT_SPL")
    config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    # config.TASK_CONFIG.TASK.MEASUREMENTS.append("EPISODE_DISTANCE")
    # config.freeze()
    envs = construct_envs(config, get_env_class(config.ENV_NAME))
    sensors_obs = envs.observation_spaces[0]

    device = torch.device("cpu")
    # args.model_path = '/Users/joanne/repos/habitat_spot_v2/spot_urdf_test/ddppo_policies/ckpt.11.pth'
    # args.model_path = '/coc/pskynet3/jtruong33/develop/flash_results/cont_ctrl_results_v2/spot_collision_0.1_nosliding_visual_encoder_nccl/checkpoints/ckpt.11.pth'
    dim_actions = 2
    model = load_model(args.model_path, args.cfg_path, dim_actions)

    model.eval()
    observations = envs.reset()
    batch = batch_obs(observations, device)

    current_episode_reward = torch.zeros(envs.num_envs, 1, device=device)
    num_processes=config.NUM_ENVIRONMENTS
    test_recurrent_hidden_states = torch.zeros(
        num_processes, 
        model.net.num_recurrent_layers,
        512,
        device=device,
    )
    prev_actions = torch.zeros(num_processes, dim_actions, device=device)
    not_done_masks = torch.zeros(num_processes, 1, dtype=torch.bool, device=device)

    stats_episodes = dict()  # dict of dicts that stores stats per episode

    stats_actions = defaultdict(int)

    rgb_frames = [
        [] for _ in range(num_processes)
    ]

    if len(config.VIDEO_OPTION) > 0:
        os.makedirs(config.VIDEO_DIR, exist_ok=True)

    ## not logging collisions
    num_actions = 0
    called_stop = False
    # frame = observations_to_image(observations[0], {})
    # rgb_frames[0].append(frame)
    while (
        len(stats_episodes) < 994 and envs.num_envs > 0
    ):
        current_episodes = envs.current_episodes()
        with torch.no_grad():
            _, actions, _, test_recurrent_hidden_states = model.act(
                batch,
                test_recurrent_hidden_states,
                prev_actions,
                not_done_masks,
                deterministic=False,
            )

            prev_actions.copy_(actions)
        step_data = [
                    action_to_velocity_control(a)
                    for a in actions.to(device="cpu")
                    ]
        outputs = envs.step(step_data)
        num_actions +=1

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        batch = batch_obs(observations, device)
        not_done_masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.bool,
            device=device,
        )

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=device
        ).unsqueeze(1)
        current_episode_reward += rewards
        next_episodes = envs.current_episodes()
        envs_to_pause = []
        n_envs = envs.num_envs
        for i in range(n_envs):
            if (
                next_episodes[i].scene_id,
                next_episodes[i].episode_id,
            ) in stats_episodes:
                envs_to_pause.append(i)
            # episode ended
            if not_done_masks[i].item() == 0:
                episode_stats = dict()
                episode_stats["spl"] = infos[i]["spl"]
                episode_stats["success"] = int(infos[i]["spl"] > 0)
                episode_stats["reward"] = current_episode_reward[i].item()
                if actions[i][0].cpu().detach().tolist() == 0:
                    called_stop = True

                print("# Actions: {}".format(num_actions))
                print("# Collisions: {}".format(infos[i]["collisions"]["count"]))
                print("Success: {}".format(episode_stats["success"]))
                print("SPL: {}".format(episode_stats["spl"]))
                print("Called Stop: {}".format(called_stop))

                current_episode_reward[i] = 0
                ## not logging collisions
                prev_base_state = [0, 0, 0]
                num_actions = 0
                print_once = True
                called_stop = False

                # use scene_id + episode_id as unique id for storing stats
                stats_episodes[
                    (
                        current_episodes[i].scene_id,
                        current_episodes[i].episode_id,
                    )
                ] = episode_stats

                if len(config.VIDEO_OPTION) > 0:
                    metric_value = episode_stats["spl"]
                    video_name = (
                        f"episode_{current_episodes[i].episode_id}"
                        f"_SPL_{metric_value:.2f}"
                    )
                    images_to_video(
                        rgb_frames[i], config.VIDEO_DIR, video_name
                    )

                    rgb_frames[i] = []

                print("Episodes finished: {}, Ep id: {}".format(len(stats_episodes), current_episodes[i].episode_id))

            # episode continues
            elif len(config.VIDEO_OPTION) > 0:
                frame = observations_to_image(observations[i], infos[i])
                rgb_frames[i].append(frame)

        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                :, state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if len(config.VIDEO_OPTION) > 0:
                rgb_frames = [rgb_frames[i] for i in state_index]

    envs.close()


if __name__ == "__main__":
    main()
