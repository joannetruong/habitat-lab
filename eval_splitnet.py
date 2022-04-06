import glob
import os
import time

import cv2
import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat_baselines.rl.ddppo.policy.splitnet_policy import \
    PointNavSplitNetPolicy
from habitat_baselines.utils.common import batch_obs


class SplitNetPolicy:
    def __init__(self, checkpoint_path, device, decoder_output):
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint["config"]
        """ Disable observation transforms for real world experiments """
        config.defrost()
        config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
        config.RL.SPLITNET.visual_encoder = "ShallowVisualEncoder"
        config.RL.SPLITNET.decoder_output = decoder_output
        config.freeze()
        observation_space = SpaceDict(
            {
                "spot_left_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "spot_right_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        # Linear, angular, and horizontal velocity (in that order)
        action_space = spaces.Box(-1.0, 1.0, (3,))
        action_space.n = 3
        self.policy = PointNavSplitNetPolicy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        print("Actor-critic architecture:", self.policy)
        # Move it to the device
        self.policy.to(self.device)
        # Load trained weights into the policy
        self.policy.load_state_dict(
            {k[len("actor_critic.") :]: v for k, v in checkpoint["state_dict"].items()},
            strict=False,
        )
        self.prev_actions = None
        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.config = config
        self.num_actions = action_space.shape[0]
        self.reset_ran = False
        self.count = 0

    def reset(self):
        self.reset_ran = True
        self.test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments. Just one for real world.
            self.policy.net.num_recurrent_layers,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )

        # We start an episode with 'done' being True (0 for 'not_done')
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, self.num_actions, device=self.device)

    def act(self, obs_np):
        assert self.reset_ran, "You need to call .reset() on the policy first."
        with torch.no_grad():
            all_pth = os.path.join(f"splitnet_imgs/real_depth_in_{self.count}.png")
            # cv2.imwrite(all_pth, obs_np)

            print(obs_np.shape)
            obs = torch.from_numpy(np.expand_dims(obs_np, axis=0)[:, :, :, :1] / 255.0)
            # obs_left = np.expand_dims(obs_left, axis=0)[:, :, :, :1] / 255.0
            # obs_right = np.expand_dims(obs_right, axis=0)[:, :, :, :1] / 255.0
            #
            # obs = torch.cat(
            #     [
            #         # Spot is cross-eyed; right is on the left on the FOV
            #         torch.from_numpy(obs_right),
            #         torch.from_numpy(obs_left),
            #     ],
            #     dim=2,
            # )

            obs = obs.permute(0, 3, 1, 2)  # NHWC => NCHW

            (
                visual_features,
                decoder_outputs,
                class_pred,
            ) = self.policy.net.visual_encoder(
                obs,
                True,
            )
        print(decoder_outputs.shape)
        self.not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        decoder_outputs = decoder_outputs.detach().cpu().numpy().squeeze() * 255
        all_pth = os.path.join(f"splitnet_imgs/real_depth_pred_{self.count:06}.png")
        print(all_pth)
        cv2.imwrite(all_pth, decoder_outputs)
        self.count += 1
        # GPU/CPU torch tensor -> numpy
        # actions = actions.squeeze().cpu().numpy()

        return decoder_outputs


if __name__ == "__main__":
    # nav_policy = SplitNetPolicy(
    #     "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_splitnet/checkpoints/ckpt.11.pth",
    #     device="cpu",
    #         decoder_output = ["depth"]
    # )
    nav_policy = SplitNetPolicy(
        "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_splitnet_motion_loss/checkpoints/ckpt.17.pth",
        device="cpu",
        decoder_output=["depth"],
    )
    #
    # nav_policy = SplitNetPolicy(
    #     "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_splitnet_surface_normal_v3/checkpoints/ckpt.19.pth",
    #     device="cpu",
    #     decoder_output=["depth", "surface_normals"],
    # )
    nav_policy.reset()

    # obs_left = cv2.imread(
    #     "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_splitnet/eval/kinematic_tmp/imgs/left_img_1.png"
    # )
    # obs_right = cv2.imread(
    #     "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_splitnet/eval/kinematic_tmp/imgs/right_img_1.png"
    # )
    # img = np.concatenate([obs_right, obs_left], axis=1)

    img_pth = "/coc/testnvme/jtruong33/google_nav/habitat-lab/out/"
    for file in sorted(glob.glob(img_pth + "*.png")):
        print(file)
        if "filtered" in file:
            img = cv2.imread(file)
            actions = nav_policy.act(img)
