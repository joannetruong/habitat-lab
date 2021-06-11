# [setup]
import os
import sys
import magnum as mn
import numpy as np
from datetime import datetime
import habitat_sim
import squaternion

# import habitat_sim.utils.common as ut
import habitat_sim.utils.viz_utils as vut
from utilities.quadruped_env import A1, AlienGo, Laikago, Spot
from utilities.daisy_env import Daisy, Daisy_4legged
from utilities.raibert_controller import Raibert_controller
from utilities.raibert_controller import Raibert_controller_turn
import cv2
import json
import time
import yaml
import torch
from gym import Space, spaces
from collections import defaultdict, deque
from habitat_cont_v2 import evaluate_ddppo
from habitat_cont_v2.evaluate_ddppo import to_tensor
from utilities import utils
from spot_walking_test import Workspace

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../habitat-sim/data")
output_path = os.path.join(dir_path, "spot_videos/")

class SpotWorkspace(Workspace):
    def __init__(self, robot):
        Workspace.__init__(self, robot)
        self.save_video_dir = 'ddppo_vids/'
        # weights_dir = 'ddppo_policies/spot_collision_0.1_nosliding_xyt_21.pth'
        # weights_dir = 'ddppo_policies/spot_collision_0.1_nosliding_xyt_backwards_23.pth'
        weights_dir = 'ddppo_policies/spot_collision_0.1_nosliding_visual_encoder_nccl_11.pth'
        self.episode_pth = 'spot_waypoints_coda_hard.yaml'
        self.dim_actions = 2
        self.allow_backwards = False
        if self.allow_backwards:
            self.linear_velocity_x_min, self.linear_velocity_x_max = -0.35, 0.35
        else:
            self.linear_velocity_x_min, self.linear_velocity_x_max = 0.0, 0.35
        self.linear_velocity_y_min, self.linear_velocity_y_max = -0.35, 0.35
        self.angular_velocity_min, self.angular_velocity_max = -0.15, 0.15

        self.success_dist = 0.36
        self.dist_to_goal = 100
        self.ctr = 0
        self.max_num_actions = 5
        self.min_depth = 0.3
        self.max_depth = 10.0
        self.device = 'cpu'
        model = evaluate_ddppo.load_model(weights_dir, self.dim_actions)
        self.model = model.eval()

    def place_agent(self):
        # place our agent in the scene
        agent_state = habitat_sim.AgentState()
        agent_state.position = [0.0, 0.0, 0.0]
        agent_state.rotation = np.quaternion(1, 0, 0, 0)

        self.agent = self.sim.initialize_agent(0, agent_state)
        return self.agent.scene_node.transformation_matrix()

    def get_episodes(self):
        start_poses = []
        goal_poses = []
        with open(self.episode_pth) as f:
            episode_dict = yaml.safe_load(f)
        for key in episode_dict:
            start_poses.append(np.array(episode_dict[key]['initial_pos']))
            goal_poses.append(np.array(episode_dict[key]['target_pos']))
        return np.array(start_poses), np.array(goal_poses)

    def get_robot_pos_hab(self):
        robot_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)
        robot_position = np.array([*robot_state.translation])

        robot_ori = utils.quaternion_from_coeff(robot_state.rotation)
        return robot_position, robot_ori

    def _compute_pointgoal(self, source_position, source_rotation, goal_position):
        source_position[1] = 0.0
        goal_position[1] = 0.0
        direction_vector = goal_position - source_position
        direction_vector_agent = utils.quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )
        rho, phi = utils.cartesian_to_polar(
            -direction_vector_agent[2], direction_vector_agent[0]
        )
        print('goal sensor: ', rho, np.rad2deg(-phi), goal_position)
        return np.array([rho, -phi], dtype=np.float32)

    def transform_angle(self, rotation):
        obs_quat = squaternion.Quaternion(rotation.scalar, *rotation.vector)
        inverse_base_transform = utils.scalar_vector_to_quat(np.pi/2,(1,0,0))
        obs_quat = obs_quat*inverse_base_transform
        # quat = squaternion.Quaternion.from_euler(0, 90, 0, degrees=True)
        # inverse_base_transform_yaw = utils.scalar_vector_to_quat(np.pi/2,(1,0,0))
        # obs_quat = obs_quat * inverse_base_transform_yaw * quat
        return obs_quat

    def get_goal_sensor(self, goal_position):
        """
        :return: non-perception observation, such as goal location
        """
        agent_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)
        agent_position = agent_state.translation
        roll, pitch, yaw = utils.get_rpy(agent_state.rotation)
        # yaw = yaw-np.deg2rad(90)
        agent_rotation = squaternion.Quaternion.from_euler(roll, pitch, yaw, degrees=False)
        # source_rotation = utils.quat_from_magnum(agent_rotation)
        # inverse_base_transform = utils.scalar_vector_to_quat(np.pi/2,(1,0,0))
        # agent_rotation = agent_rotation*inverse_base_transform

        rotation_world_agent = utils.quat_from_magnum(agent_rotation)
        print('robot state: ', agent_state.translation, np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw))
        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )

    def evaluate_ddppo(self):
        start_poses, goal_poses = self.get_episodes()
        num_episodes = len(start_poses)
        # for episode in range(num_episodes):
        for episode in range(1):
            self.evaluate_episode(start_poses[episode], goal_poses[episode])

    def check_done(self, action, state):
        linear_velocity, strafe_velocity, angular_velocity = action
        if abs(linear_velocity) < 0.1 and abs(strafe_velocity) < 0.1 and abs(angular_velocity) < 0.1:
            print('ROBOT CALLED STOP!')
            self.done = True
            if self.dist_to_goal < self.success_dist:
                print('EPISODE SUCCESS')
                self.success = True
            else:
                print('EPISODE FAIL')
                self.success = False
        # roll, pitch, _ = state['base_ori_euler']
        # if (np.abs(roll) > 0.75 or np.abs(pitch) > 0.75):
        #     print('FAIL, ROBOT FELL OVER')
        #     self.done = True
    
    def evaluate_episode(self, start_pos, goal_pos):
        self.reset_robot(start_pos)

        self.success = False
        episode_reward = 0
        
        num_processes=1
        test_recurrent_hidden_states = torch.zeros(
                num_processes, 
                self.model.net.num_recurrent_layers,
                512,
                device=self.device,
            )
        prev_actions = torch.zeros(num_processes, self.dim_actions, device=self.device)
        not_done_masks = torch.zeros(num_processes, 1, dtype=torch.bool, device=self.device)
        self.depth_obs = np.zeros((240, 320))

        while not self.done and self.num_actions < self.max_num_actions:
            observations = {}

            # normalize depth images
            self.depth_obs = (self.depth_obs - self.min_depth) / (self.max_depth - self.min_depth)
            observations["depth"] = np.expand_dims(self.depth_obs, axis=2)
            observations["pointgoal_with_gps_compass"] = self.get_goal_sensor(goal_pos)
            observations = [observations]

            batch = defaultdict(list)

            for obs in observations:
                for sensor in obs:
                    batch[sensor].append(to_tensor(obs[sensor]))

            for sensor in batch:
                batch[sensor] = torch.stack(batch[sensor], dim=0).to(
                    device=self.device, dtype=torch.float
                )
            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states = self.model.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )
                actions = actions.reshape([1, self.dim_actions]).to(device="cpu")
                print('[actions]: {}'.format(actions))
                prev_actions.copy_(actions)
            not_done_masks = torch.ones(num_processes, 1, dtype=torch.bool, device=self.device)
            if self.dim_actions == 2:
                linear_velocity, angular_velocity = torch.clip(actions[0], min=-1, max=1)
                strafe_velocity = 0.0
            else:
                linear_velocity, strafe_velocity, angular_velocity = torch.clip(actions[0], min=-1, max=1)

            linear_velocity = (linear_velocity + 1.0) / 2.0
            strafe_velocity = (strafe_velocity + 1.0) / 2.0
            angular_velocity = (angular_velocity + 1.0) / 2.0

            linear_velocity = self.linear_velocity_x_min + linear_velocity * (
            self.linear_velocity_x_max - self.linear_velocity_x_min
            )
            strafe_velocity = self.linear_velocity_y_min + strafe_velocity * (
                self.linear_velocity_y_max - self.linear_velocity_y_min
            )
            angular_velocity = self.angular_velocity_min + angular_velocity * (
                self.angular_velocity_max - self.angular_velocity_min
            )
            print('[commands]: {} {} {}'.format(linear_velocity, strafe_velocity, angular_velocity))

            action = np.array([linear_velocity, strafe_velocity, angular_velocity])
            print('NUM ACTIONS: ', self.num_actions)
            self.depth_obs = self.step_robot(action) 
            self.num_actions +=1
        time_str = datetime.now().strftime("_%d%m%y_%H_%M_")
        save_name = time_str + 'pos_gain=' + str(self.pos_gain) + '_vel_gain=' + \
                    str(self.vel_gain) + '_finite_diff=' + str(self.finite_diff) + '.mp4'
        rate = self.ctrl_freq // 30
        self.save_video(rate, save_name)
        # self.make_video()

if __name__ == "__main__":
    robot = sys.argv[1]
    scene = "data/scene_datasets/coda/coda_hard.glb"
    SW = SpotWorkspace(robot)
    SW.setup(scene)
    SW.place_agent()
    SW.place_camera_agent()
    SW.load_robot()
    SW.evaluate_ddppo()

