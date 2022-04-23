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
from habitat_baselines.rl.ddppo.policy.splitnet_policy import \
    PointNavSplitNetPolicy

LR = 5e-4

EPOCHS = 10000
BATCH_SIZE = 32
NUM_BATCHES_PER_EPOCH = 1000
NUM_TEST_BATCHES_PER_EPOCH = 50
SAVE_INTERVAL = 1000

# EPOCHS = 1
# BATCH_SIZE = 1
# NUM_BATCHES_PER_EPOCH = 1
# NUM_TEST_BATCHES_PER_EPOCH = 1
# SAVE_INTERVAL = 1

CKPT_DIR = "splitnet_ft_ckpts_v2"
CKPT = "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_splitnet_motion_loss/checkpoints/ckpt.17.pth"


class SplitNetPolicy:
    def __init__(self, checkpoint_path, device, decoder_output):
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        """ Disable observation transforms for real world experiments """
        config.defrost()
        config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
        config.RL.SPLITNET.visual_encoder = "ShallowVisualEncoder"
        config.RL.SPLITNET.decoder_output = decoder_output
        config.RL.SPLITNET.update_visual_decoder_features = False
        config.RL.SPLITNET.freeze_visual_decoder_features = True
        config.RL.SPLITNET.freeze_motion_decoder_features = True
        config.freeze()
        self.config = config
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
            # strict=False,
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
        with torch.no_grad():
            for _ in tqdm.tqdm(range(NUM_TEST_BATCHES_PER_EPOCH)):
                obs = random.sample(self.train_data, BATCH_SIZE)
                obs = torch.from_numpy(np.array(obs)) / 255.0
                obs = obs.to(self.device)
                # obs = self.eval_data[self.eval_img_ctr]
                # obs = torch.from_numpy(np.expand_dims(obs, axis=0) / 255.0)
                visual_features, pred, _ = self.policy.net.visual_encoder.forward(
                    obs.float(), True
                )
                label = obs
                loss = self.criterion(pred, label)
                loss = torch.mean(loss)
                n_its += 1
                val_loss += loss.item()
        val_loss /= n_its
        return val_loss

    def train_model(self):
        self.policy.net.train()
        for _ in tqdm.tqdm(range(NUM_BATCHES_PER_EPOCH)):
            obs = random.sample(self.train_data, BATCH_SIZE)
            # obs = self.train_data[self.train_img_ctr]
            # obs = torch.from_numpy(np.expand_dims(obs, axis=0) / 255.0)
            obs = torch.from_numpy(np.array(obs)) / 255.0
            obs = obs.to(self.device)

            visual_features, pred, _ = self.policy.net.visual_encoder.forward(
                obs.float(), True
            )
            label = obs
            loss = self.criterion(pred, label)
            loss = torch.mean(loss)
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

    def finetune(self, train_data, eval_data):
        self.train_data = train_data
        self.eval_data = eval_data
        for epoch in range(0, EPOCHS + 1):
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
