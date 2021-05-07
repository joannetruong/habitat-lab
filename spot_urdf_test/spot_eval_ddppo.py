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

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../habitat-sim/data")
output_path = os.path.join(dir_path, "spot_videos/")

class Workspace(object):
    def __init__(self, robot):
        self.raibert_infos = {}
        self.ep_id = 1
        self.depth_ortho_imgs = []
        self.text = []
        self.pos_gain = np.ones((3,)) * 0.2 # 0.2 
        self.vel_gain = np.ones((3,)) * 1.5 # 1.5
        self.pos_gain[2] = 0.7 # 0.7
        self.vel_gain[2] = 1.5 # 1.5
        self.num_steps = 80
        self.ctrl_freq = 240
        self.time_per_step = 80
        self.prev_state=None
        self.finite_diff=False
        self.min_depth = 0.3
        self.max_depth = 10.0
        self.robot_name = robot
        self.setup()
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
        self.device = 'cpu'
        model = evaluate_ddppo.load_model(weights_dir, self.dim_actions)
        self.model = model.eval()

    def make_configuration(self):
        # simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        # backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/empty_room.glb"
        backend_cfg.scene_id = "data/scene_datasets/coda/coda_hard.glb"
        backend_cfg.enable_physics = True

        # sensor configurations
        # Note: all sensors must have the same resolution
        # setup 2 rgb sensors for 1st and 3rd person views
        camera_resolution_large = [540, 720]
        camera_resolution_small = [240, 320]
        sensors = {
            "rgba_camera_1stperson": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": camera_resolution_small,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "sensor_subtype": habitat_sim.SensorSubType.PINHOLE,
            },
            "depth_camera_1stperson": {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": camera_resolution_small,
                "position": [0.0,0.3,-0.1778],#0.0762+0.11
                # "position": [0.0,0.1862,-0.1778],#0.0762+0.11
                # "position": [0.0,0.1862,0.],#0.0762+0.11
                "orientation": [0.0, 0.0, 0.0],
                "sensor_subtype": habitat_sim.SensorSubType.PINHOLE,
            },
            "rgba_camera_3rdperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution_large,
            # "position": [-2.0,2.0,0.0],#[0.0, 1.0, 0.3],
            # "orientation": [0.0,0.0,0.0],#[-45, 0.0, 0.0],
            # "position": [0.0,3.0,1.0],#[0.0, 1.0, 0.3],
            # "orientation": [0.0,np.deg2rad(90),np.deg2rad(20)],#[-45, 0.0, 0.0],
            "position": [-2.0,3.50,10.0],#[0.0, 1.0, 0.3],
            "orientation": [np.deg2rad(-10),np.deg2rad(20.0),0.0],#[-45, 0.0, 0.0],
            "sensor_subtype": habitat_sim.SensorSubType.ORTHOGRAPHIC,
            },
        }

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            sensor_spec = habitat_sim.CameraSensorSpec() #habitat_sim.SensorSpec()
            if sensor_uuid == 'depth_camera_1stperson':
                sensor_spec.hfov = mn.Deg(70)
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.orientation = sensor_params["orientation"]
            sensor_specs.append(sensor_spec)

        # agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        agent_cfg2 = habitat_sim.agent.AgentConfiguration()
        ortho_spec = habitat_sim.CameraSensorSpec()
        ortho_spec.uuid = "ortho"
        ortho_spec.sensor_type    =  habitat_sim.SensorType.COLOR
        ortho_spec.resolution     =  camera_resolution_large
        ortho_spec.position       =  [-2.0,3.50,10.0]
        ortho_spec.orientation    =  [np.deg2rad(-10),np.deg2rad(20.0),0.0]
        ortho_spec.sensor_subtype =  habitat_sim.SensorSubType.ORTHOGRAPHIC
        agent_cfg2.sensor_specifications = [ortho_spec]

        return habitat_sim.Configuration(backend_cfg, [agent_cfg, agent_cfg2])

    def place_agent(self):
        # place our agent in the scene
        agent_state = habitat_sim.AgentState()
        # agent_state.position = [-0.15, 0.7, 1.0]
        agent_state.position = [0.0, 0.0, 0.0]
        agent_state.rotation = np.quaternion(1, 0, 0, 0)
        self.agent = self.sim.initialize_agent(0, agent_state)

        agent_state2 = habitat_sim.AgentState()
        # agent_state2.position = [-6.0,2.0,4.0]
        # agent_rotation = squaternion.Quaternion.from_euler(np.deg2rad(-10.0),np.deg2rad(-80.0),np.deg2rad(0.0), degrees=False)
        agent_state2.position = [-4.0,2.0,5.0]
        agent_rotation = squaternion.Quaternion.from_euler(np.deg2rad(-20.0),np.deg2rad(-80.0),np.deg2rad(0.0), degrees=False)
        agent_state2.rotation = utils.quat_from_magnum(agent_rotation)
        # agent_state.rotation = np.quaternion(1, 0, 0, 0)
        agent2 = self. sim.initialize_agent(1, agent_state2)

        return self.agent.scene_node.transformation_matrix()

    def setup(self):
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # [initialize]
        # create the simulator
        cfg = self.make_configuration()

        self.sim = habitat_sim.Simulator(cfg)
        agent_transform = self.place_agent()

        # [/initialize]
        urdf_files = {
            "Spot": os.path.join(
                data_path, "URDF_demo_assets/spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid.urdf"
            ),
            "A1": os.path.join(
                data_path, "URDF_demo_assets/a1/a1.urdf"
            ),
            "AlienGo": os.path.join(
                data_path, "URDF_demo_assets/aliengo/urdf/aliengo.urdf"
            ),
            "Laikago": os.path.join(
                data_path, "URDF_demo_assets/laikago/laikago.urdf"
            ),
            "Daisy": os.path.join(
                data_path, "URDF_demo_assets/daisy/daisy_advanced_side.urdf"
            ),
            "Daisy4": os.path.join(
                data_path, "URDF_demo_assets/daisy/daisy_advanced_4legged.urdf"
            ),
        }

        # [basics]
        # load a URDF file
        robot_file_name = self.robot_name
        robot_file = urdf_files[robot_file_name]
        self.robot_id = self.sim.add_articulated_object_from_urdf(robot_file, fixed_base=False)

        # local_base_pos = np.array([-5,0.0,0.0]) 
        # agent_transform = self.sim.agents[0].scene_node.transformation_matrix()
        # base_transform = mn.Matrix4.rotation(mn.Rad(-1.57), mn.Vector3(1.0, 0, 0))
        # base_transform.translation = agent_transform.transform_point(local_base_pos)
        # self.sim.set_articulated_object_root_state(self.robot_id, base_transform)

        jms = []

        jms.append(habitat_sim.physics.JointMotorSettings(
                        0,  # position_target
                        self.pos_gain[0],  # position_gain
                        0,  # velocity_target
                        self.vel_gain[0],  # velocity_gain
                        10.0,  # max_impulse
                    ))
        jms.append(habitat_sim.physics.JointMotorSettings(
                        0,  # position_target
                        self.pos_gain[1],  # position_gain
                        0,  # velocity_target
                        self.vel_gain[1],  # velocity_gain
                        10.0,  # max_impulse
                    ))
        jms.append(habitat_sim.physics.JointMotorSettings(
                        0,  # position_target
                        self.pos_gain[2],  # position_gain
                        0,  # velocity_target
                        self.vel_gain[2],  # velocity_gain
                        10.0,  # max_impulse
                    ))      
        for i in range(12):
            self.sim.update_joint_motor(self.robot_id, i, jms[np.mod(i,3)])

        # base_transform = mn.Matrix4.rotation(mn.Rad(-1.57), mn.Vector3(1, 0, 0).normalized())
        # inverse_transform = base_transform.inverted()
        # transform2 = mn.Matrix4.rotation(mn.Rad(1.2), mn.Vector3(0, 0, 1).normalized())
        # base_transform = base_transform.__matmul__(transform2)
        # inverse_transform = base_transform.inverted()
        # base_transform.translation = agent_transform.transform_point(local_base_pos)    
        # sim.set_articulated_object_root_state(robot_id, base_transform)


        # existing_joint_motors = sim.get_existing_joint_motors(robot_id)    
        agent_config = habitat_sim.AgentConfiguration()
        scene_graph = habitat_sim.SceneGraph()
        agent = habitat_sim.Agent(scene_graph.get_root_node().create_child(), agent_config)

        if self.robot_name == 'A1':
            self.robot = A1(sim=self.sim, agent=self.agent, robot_id=self.obot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'AlienGo':
            self.robot = AlienGo(sim=self.sim, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'Daisy':
            self.robot = Daisy(sim=self.sim, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'Laikago':
            self.robot = Laikago(sim=self.sim, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'Daisy_4legged':
            self.robot = Daisy4(sim=self.sim, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'Spot':
            self.robot = Spot(sim=self.sim, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)
            
        self.robot.robot_specific_reset()
        
        # Set up Raibert controller and link it to spot
        action_limit = np.zeros((12, 2))
        action_limit[:, 0] = np.zeros(12) + np.pi / 2
        action_limit[:, 1] = np.zeros(12) - np.pi / 2
        self.raibert_controller = Raibert_controller_turn(control_frequency=self.ctrl_freq, num_timestep_per_HL_action=self.time_per_step, robot=self.robot_name)
        position, orientation = self.get_robot_pos()
        print('position: ', position, 'orientation: ', np.rad2deg(orientation[-1]))
    # [/setup]
    def reset_robot(self, start_pose):
        # local_base_pos = start_pose
        # local_base_pos = start_pose - np.array([-0.15, 0.0, 1.0])
        # local_base_pos = start_pose - np.array([-0.15, 0.7, 2.06]) # [-5.62700033 -0.49999601 -0.70499998]
        local_base_pos = start_pose
        # print('start pose: ', start_pose)
        # local_base_pos = utils.rotate_pos_from_hab(start_pose)
        agent_transform = self.sim.agents[0].scene_node.transformation_matrix()
        base_transform = mn.Matrix4.rotation(mn.Rad(-1.57), mn.Vector3(1.0, 0, 0))
        base_transform.translation = agent_transform.transform_point(local_base_pos)
        self.sim.set_articulated_object_root_state(self.robot_id, base_transform)

        # base_transform = mn.Matrix4.rotation(mn.Rad(-1.57), mn.Vector3(1.0, 0, 0))
        # base_transform.translation = mn.Vector3(*start_pose)
        # self.sim.set_articulated_object_root_state(self.robot_id, base_transform)

        # self.sim.step_physics(1/240.0)
        # link_rigid_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)
        # agent_transform = self.sim.agents[0].scene_node.transformation_matrix()
        # self.start_rotation = utils.quat_from_magnum(link_rigid_state.rotation)
        # quat = squaternion.Quaternion.from_euler(0, 90, 0, degrees=True)
        # base_quat = utils.get_quat(-np.pi/2,(1,0,0))
        # new_quat = quat * base_quat
        # scalar, vector = utils.get_scalar_vector(new_quat)
        # base_transform = mn.Matrix4.rotation(mn.Rad(scalar), mn.Vector3(*vector))
        # # base_transform.translation = agent_transform.transform_point(local_base_pos)
        # base_transform.translation = np.array([-5.627, 0.705, -0.499996])
        # self.sim.set_articulated_object_root_state(self.robot_id, base_transform)
        # link_rigid_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)

        # base_transform_yaw = mn.Matrix4.rotation(mn.Rad(-1.57), mn.Vector3(1.0, 0.0, 0.0))
        # base_transform_yaw.translation = base_transform.translation
        # print('base transform translation yaw: ', base_transform_yaw.translation)
        # self.sim.set_articulated_object_root_state(self.robot_id, base_transform_yaw)

        self.init_state = self.robot.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
        self.raibert_controller.set_init_state(self.init_state)
        time.sleep(1)

        pos, ori = self.get_robot_pos()
        print('robot start pos: ', pos, np.rad2deg(ori[-1]))

    def log_raibert_controller(self):
        self.raibert_infos[str(self.ep_id)] = {}
        self.raibert_infos[str(self.ep_id)]["target_speed"] = [float(ii) for ii in self.target_speed]
        self.raibert_infos[str(self.ep_id)]["target_speed_ang"] = float(self.target_ang_vel)
        self.raibert_infos[str(self.ep_id)]["input_current_speed"] = [float(ii) for ii in self.input_current_speed]
        self.raibert_infos[str(self.ep_id)]["input_joint_pos"] = [float(ii) for ii in self.input_joint_pos]
        self.raibert_infos[str(self.ep_id)]["input_current_yaw_rate"] = float(self.input_current_yaw_rate)
        self.raibert_infos[str(self.ep_id)]["latent_action"] = [float(ii) for ii in self.latent_action]
        self.raibert_infos[str(self.ep_id)]["input_base_ori_euler"] = [float(ii) for ii in self.input_base_ori_euler]
        self.raibert_infos[str(self.ep_id)]["raibert_action_commanded"] = [[float(iii) for iii in ii] for ii in self.raibert_action_commanded]
        self.raibert_infos[str(self.ep_id)]["raibert_action_measured"] = [[float(iii) for iii in ii] for ii in self.raibert_action_measured]
        self.raibert_infos[str(self.ep_id)]["raibert_base_velocity"] = [[float(iii) for iii in ii] for ii in self.raibert_base_velocity]
        
    def cmd_vel_xyt(self, lin_x, lin_y, ang):
        for n in range(self.num_steps):
            action = np.array([lin_x, lin_y, ang])
            self.step_robot(action) 
            # self.log_raibert_controller()
            self.ep_id +=1

    def step_robot(self, action):
        state = self.robot.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
        target_speed = np.array([action[0], action[1]])
        target_ang_vel = action[2]
        latent_action = self.raibert_controller.plan_latent_action(state, target_speed, target_ang_vel=target_ang_vel)
        self.raibert_controller.update_latent_action(state, latent_action)
        for i in range(self.time_per_step):
            # Get actual joint actions 
            raibert_action = self.raibert_controller.get_action(state, i+1)
            # Simulate spot for 1/ctrl_freq seconds and return camera observation
            agent_obs, ortho_obs = self.robot.step(raibert_action, self.pos_gain, self.vel_gain, dt=1/self.ctrl_freq, follow_robot=True)
            # Recalculate spot state for next action
            state = self.robot.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
            self.check_done(action, state)
            if self.done:
                break
        self.stitch_show_img(agent_obs, ortho_obs)        
        return agent_obs[0]['depth_camera_1stperson'] 

    def stitch_show_img(self, agent_obs, ortho_obs, show=True):
        depth_img = cv2.cvtColor(np.uint8(agent_obs[0]['depth_camera_1stperson']/ 10 * 255),cv2.COLOR_RGB2BGR)
        ortho_img = cv2.cvtColor(np.uint8(ortho_obs[0]['ortho']),cv2.COLOR_RGB2BGR)

        height,width,layers=ortho_img.shape
        resize_depth_img = cv2.resize(depth_img, (width, height))
        frames = np.concatenate((resize_depth_img, ortho_img), axis=1)
        self.depth_ortho_imgs.append(frames)

        key = cv2.waitKey(100)
        if key == ord('q'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            self.check_make_dir(self.save_video_dir)
            video=cv2.VideoWriter(os.path.join(self.save_video_dir, 'spot_coda_ddppo.mp4'),fourcc,10,(2*width,height))
            for frame in self.depth_ortho_imgs:
                video.write(frame)
            exit()
        cv2.imshow('depth_ortho_imgs',frames)

    def check_make_dir(self, pth):
        if not os.path.exists(pth):
            os.makedirs(pth)
        return
        
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

    def get_robot_pos(self):
        robot_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)
        base_pos_hab = utils.rotate_pos_from_hab(robot_state.translation)

        robot_position = np.array([base_pos_hab[0], base_pos_hab[1], base_pos_hab[2]])
        robot_ori = utils.get_rpy(robot_state.rotation) 
        # robot_ori[-1] -= np.deg2rad(90)
        return robot_position, robot_ori

    def _compute_pointgoal(self, source_position, source_rotation, goal_position):
        print('source_position: ', source_position)
        print('source_rotation: ', source_rotation)
        source_position[1] = 0.0
        goal_position[1] = 0.0
        direction_vector = goal_position - source_position
        direction_vector_agent = utils.quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )
        rho, phi = utils.cartesian_to_polar(
            -direction_vector_agent[2], direction_vector_agent[0]
        )
        # phi -= np.pi/2
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
        yaw = yaw-np.deg2rad(90)
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

        self.done = False
        self.success = False
        episode_reward = 0
        num_actions = 0
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

        while not self.done:
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
            print('NUM_ACTIONS: ', num_actions)
            self.depth_obs = self.step_robot(action) 
            num_actions +=1
        self.save_video()
        # self.make_video()

if __name__ == "__main__":
    robot = sys.argv[1]
    W = Workspace(robot)
    W.evaluate_ddppo()
