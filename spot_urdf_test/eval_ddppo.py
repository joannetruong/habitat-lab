import os
import sys
import time
import torch
import utils
import hydra
import numpy as np

from omegaconf import OmegaConf
from logger import TBLogger, CSVLogger
from pytorch_sac_private.load_env import load_env, load_eval_env
from workspaces.daisy_workspace import DaisyWorkspace, Daisy4Workspace
from pytorch_sac_private.workspaces.a1_workspace import (
    A1Workspace,
    AlienGoWorkspace,
    LaikagoWorkspace,
    SpotWorkspace,
    SpotHabWorkspace,
)
from habitat_cont import evaluate_ddppo
from workspaces.sphere_workspace import SphereWorkspace
from gibson2.envs.parallel_env import ParallelNavEnvironment
from termcolor import colored
import logging

log = logging.getLogger(__name__)


sys.path.insert(0, "pytorch_sac_private")

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


@hydra.main(config_path="config/base_config.yaml", strict=True)
def main(cfg):
    log.info(colored(cfg.pretty(), "green"))
    base_dir = hydra.utils.get_original_cwd()
    model_dir = os.path.join(os.getcwd(), "model")

    utils.set_seed_everywhere(cfg.seed)
    sac_cfg = OmegaConf.load(os.path.join(base_dir, "config/agent/sac_up.yaml"))
    cfg = OmegaConf.merge(cfg, sac_cfg.agent_cfg)
    cfg.agent.params.obs_shape = [
        3,
        cfg.gibson_cfg.image_height,
        cfg.gibson_cfg.image_width,
    ]
    cfg.agent.params.linear_x_action_range = [
        -cfg.gibson_cfg.linear_velocity_x,
        cfg.gibson_cfg.linear_velocity_x,
    ]
    cfg.agent.params.linear_y_action_range = [
        -cfg.gibson_cfg.linear_velocity_y,
        cfg.gibson_cfg.linear_velocity_y,
    ]
    cfg.agent.params.angular_action_range = [
        -cfg.gibson_cfg.angular_velocity,
        cfg.gibson_cfg.angular_velocity,
    ]
    cfg.base_dir = base_dir
    
    step = 0
    csv_logger = CSVLogger(
        os.getcwd(),
        save_tb=cfg.log_save_tb,
        log_frequency=cfg.log_frequency,
        agent=cfg.agent.name,
    )
    tb_logger = TBLogger(
        os.getcwd(),
        save_tb=cfg.log_save_tb,
        log_frequency=cfg.log_frequency,
        agent=cfg.agent.name,
    )
    cfg.use_ddppo=True
    cfg.gibson_cfg.image_width=640
    cfg.gibson_cfg.image_height=480
    os.mkdir('imgs_ddppo_' + cfg.ddppo_weights.split('/')[-1])
    # cfg.gibson_cfg.scene = "stadium"
    robots, idxs = select_robots(cfg)
    num_robots = len(robots)
    print('robots: ', robots[0])
    eval_env = load_eval_env(cfg, robots[0], idxs[0])
    weights_dir = os.path.join(base_dir, cfg.ddppo_weights)
    model = evaluate_ddppo.load_model(weights_dir, cfg.dim_actions)
    model = model.eval()

    eval_env.set_agent(model)
    # policies_dir = os.path.join(base_dir, cfg.policies_dir)
    # eval_env.load_ckpt([policies_dir] * num_robots)

    eval_info = eval_env.evaluate(
        cfg.gibson_cfg.target_dist_min,
        cfg.gibson_cfg.target_dist_max,
        step,
    )
    for i in range(len(eval_info[0])):
        for robot_info in eval_info:
            for key in robot_info[i]:
                csv_logger.log("eval/" + key, robot_info[i][key], step)
        csv_logger.dump(step, ty="eval")

def select_robots(cfg):
    if cfg.up:
        a1 = A1Workspace(cfg)
        aliengo = AlienGoWorkspace(cfg)
        daisy = DaisyWorkspace(cfg)
        laikago = LaikagoWorkspace(cfg)
        spot = SpotWorkspace(cfg)
        spot_bd = SpotHabWorkspace(cfg)
        daisy4 = Daisy4Workspace(cfg)
        robots = [a1, aliengo, daisy, laikago, daisy4]
        idxs = None
    else:
        if cfg.robot_cfg.robot == "A1":
            robot = A1Workspace(cfg)
            idxs = [1]
        elif cfg.robot_cfg.robot == "AlienGo":
            robot = AlienGoWorkspace(cfg)
            idxs = [2]
        elif cfg.robot_cfg.robot == "Daisy":
            robot = DaisyWorkspace(cfg)
            idxs = [3]
        elif cfg.robot_cfg.robot == "Laikago":
            robot = LaikagoWorkspace(cfg)
            idxs = [4]
        elif cfg.robot_cfg.robot == "Spot":
            robot = SpotWorkspace(cfg)
            idxs = [4]
        elif cfg.robot_cfg.robot == "Daisy_4legged":
            robot = Daisy4Workspace(cfg)
            idxs = [5]
        elif cfg.robot_cfg.robot == "SpotHab":
            robot = SpotHabWorkspace(cfg)
            idxs = [6]
        log.info(f"ROBOT: {cfg.robot_cfg.robot}")
        robots = [robot]
    return robots, idxs


if __name__ == "__main__":
    main()
