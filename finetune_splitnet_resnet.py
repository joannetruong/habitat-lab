import glob
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat_baselines.common.auxiliary_tasks import *
from habitat_baselines.rl.ddppo.policy.splitnet_policy import \
    PointNavSplitNetPolicy

LR = 5e-4

EPOCHS = 10000
BATCH_SIZE = 128
NUM_BATCHES_PER_EPOCH = 1000
NUM_TEST_BATCHES_PER_EPOCH = 50
SAVE_INTERVAL = 10000

# EPOCHS = 1
# BATCH_SIZE = 1
# NUM_BATCHES_PER_EPOCH = 1
# NUM_TEST_BATCHES_PER_EPOCH = 1
# SAVE_INTERVAL = 1

CKPT_DIR = "splitnet_resnet_ft_ckpts"
CKPT = "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_resnet_splitnet_white_tmp/checkpoints/ckpt.1.pth"


class SplitNetPolicy:
    def __init__(self, checkpoint_path, device, decoder_output):
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        """ Disable observation transforms for real world experiments """
        config.defrost()
        config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
        config.RL.SPLITNET.visual_encoder = "BaseResNetEncoder"
        config.RL.SPLITNET.decoder_output = decoder_output
        config.RL.SPLITNET.update_visual_decoder_features = False
        config.RL.SPLITNET.freeze_visual_decoder_features = True
        config.RL.SPLITNET.freeze_motion_decoder_features = True
        config.freeze()
        self.config = config
        observation_space = SpaceDict(
            {
                "depth": spaces.Box(
                    low=0.0, high=1.0, shape=(256, 256, 1), dtype=np.float32
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
        aux_tasks = []
        for task in config.RL.AUX_TASKS.tasks:
            aux_module = eval(task)(
                config.RL.PPO,
                config.RL.AUX_TASKS[task],
                config.TASK_CONFIG.TASK,
                self.device,
            ).to(self.device)
            aux_tasks.append(aux_module)
        self.aux_tasks = nn.Sequential(*aux_tasks)

        self.aux_tasks.load_state_dict(
            {
                k[len("aux_tasks.") :]: v
                for k, v in checkpoint["state_dict"].items()
                if "aux_tasks" in k
            },
        )
        print("Aux tasks: ", aux_tasks)

        print("Actor-critic architecture:", self.policy)
        # Move it to the device
        self.policy.to(self.device)
        print("self.policy: ", self.policy)
        # Load trained weights into the policy
        self.policy.load_state_dict(
            {
                k[len("actor_critic.") :]: v
                for k, v in checkpoint["state_dict"].items()
                if "actor_critic" in k
            },
        )
        self.optimizer = optim.Adam(self.policy.net.visual_encoder.parameters(), lr=LR)
        self.criterion = nn.L1Loss(reduction="none")
        self.total_num_steps = 0
        self.train_img_ctr = 0
        self.eval_img_ctr = 0
        self.train_data = []
        self.eval_data = []
        self.ckpt_dir = os.path.abspath(CKPT_DIR)
        print("CHECKPOINT DIR: ", self.ckpt_dir)

    def eval_model(self):
        self.policy.eval()
        val_loss = 0
        n_its = 0
        batch_obs = {}
        observations = {}
        with torch.no_grad():
            for _ in tqdm.tqdm(range(NUM_TEST_BATCHES_PER_EPOCH)):
                obs = random.sample(self.train_data, BATCH_SIZE)
                obs = torch.from_numpy(np.array(obs)) / 255.0
                obs = obs.to(self.device)
                observations["depth"] = obs.permute(0, 2, 3, 1)

                visual_features = self.policy.net.visual_encoder.forward(observations)
                batch_obs["observations"] = observations
                loss = self.aux_tasks[0].get_loss(batch_obs, visual_features)
                n_its += 1
                val_loss += loss.item()
        val_loss /= n_its
        return val_loss

    def train_model(self):
        self.policy.net.train()
        batch_obs = {}
        observations = {}
        for _ in range(NUM_BATCHES_PER_EPOCH):
            obs = random.sample(self.train_data, BATCH_SIZE)
            obs = torch.from_numpy(np.array(obs)) / 255.0
            obs = obs.to(self.device)
            observations["depth"] = obs.permute(0, 2, 3, 1)

            visual_features = self.policy.net.visual_encoder.forward(observations)
            batch_obs["observations"] = observations
            loss = self.aux_tasks[0].get_loss(batch_obs, visual_features)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_num_steps += 1

            if self.total_num_steps % SAVE_INTERVAL == 0:
                print("saving checkpoint")
                checkpoint = {
                    "state_dict": {
                        "actor_critic." + k: v
                        for k, v in self.policy.state_dict().items()
                    },
                    "config": self.config,
                }
                torch.save(
                    checkpoint,
                    os.path.join(self.ckpt_dir, f"ckpt.{self.total_num_steps}.pth"),
                )
        return loss

    def finetune(self, train_data, eval_data):
        self.train_data = train_data
        self.eval_data = eval_data
        for epoch in tqdm.tqdm(range(0, EPOCHS + 1)):
            train_loss = self.train_model()
            val_loss = self.eval_model()
            print(f"Train loss: {train_loss}, Val loss: {val_loss}")
            print(f"Epochs: {epoch}, Total num steps: {self.total_num_steps}")


if __name__ == "__main__":
    nav_policy = SplitNetPolicy(
        CKPT,
        device="cuda",
        decoder_output=["depth"],
    )
    train_pth = os.path.abspath("data/outdoor_imgs/bay_trail_filtered/")
    eval_pth = os.path.abspath("data/outdoor_imgs/outdoor_inspection_route_filtered/")

    train_data = []
    for file in sorted(glob.glob(os.path.join(train_pth, "*.png"))):
        img = cv2.imread(file)
        depth_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        train_data.append(np.expand_dims(depth_img, axis=0))
        # break

    eval_data = []
    for file in sorted(glob.glob(os.path.join(eval_pth, "*.png"))):
        img = cv2.imread(file)
        depth_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eval_data.append(np.expand_dims(depth_img, axis=0))
        # break

    nav_policy.finetune(train_data, eval_data)
