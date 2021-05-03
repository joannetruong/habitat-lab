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
from utilities.utils import get_rpy, rotate_hab_pos

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../habitat-sim/data")
output_path = os.path.join(dir_path, "spot_videos/")

class Workspace(object):
    def __init__(self, robot):
        self.raibert_infos = {}
        self.ep_id = 1
        self.observations = []
        self.text = []
        self.pos_gain = np.ones((3,)) * 0.2 # 0.2 
        self.vel_gain = np.ones((3,)) * 1.5 # 1.5
        self.pos_gain[2] = 0.7 # 0.7
        self.vel_gain[2] = 1.5 # 1.5
        self.num_steps = 30
        self.ctrl_freq = 240
        self.time_per_step = 80
        self.prev_state=None
        self.finite_diff=False
        self.robot_name = robot
        self.setup()
        self.save_img_dir = 'ddppo_imgs/'
        weights_dir = 'ddppo_policies/ckpt.15.json'
        self.episode_pth = 'spot_waypoints_coda_hard.yaml'
        self.dim_actions = 2
        self.linear_velocity_x = 0.35
        self.linear_velocity_y = 0.15    
        self.angular_velocity = 0.15    
        self.allow_backwards = False
        self.ctr = 0
        self.device = 'cuda:0'
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
                "position": [0.0,0,0.0],#[0.0, 0.6, 0.0],
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
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.orientation = sensor_params["orientation"]
            sensor_specs.append(sensor_spec)

        # agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        return habitat_sim.Configuration(backend_cfg, [agent_cfg])

    def place_agent(self):
        # place our agent in the scene
        agent_state = habitat_sim.AgentState()
        agent_state.position = [-0.15, 0.7, 1.0]
        agent_state.rotation = np.quaternion(-0.83147, 0, 0.55557, 0)
        self.agent = self.sim.initialize_agent(0, agent_state)
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
        
        # Set root state for the URDF in the sim relative to the agent and find the inverse transform for finding velocities later
        local_base_pos = np.array([-3,0.0,4.5]) 
        agent_transform = self.sim.agents[0].scene_node.transformation_matrix()
        base_transform = mn.Matrix4.rotation(mn.Rad(-1.57), mn.Vector3(1.0, 0, 0))
        base_transform.translation = agent_transform.transform_point(local_base_pos)
        self.sim.set_articulated_object_root_state(self.robot_id, base_transform)

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
        print('position: ', position, 'orientation: ', orientation)
    # [/setup]
    def reset_robot(self):
        self.init_state = self.robot.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
        self.raibert_controller.set_init_state(self.init_state)
        time.sleep(1)

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
            self.log_raibert_controller()
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
            observation = self.robot.step(raibert_action, self.pos_gain, self.vel_gain, dt=1/self.ctrl_freq, follow_robot=True)
            self.observations += observation
            # Recalculate spot state for next action
            state = self.robot.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
        self.save_img(observation)
        return observation[0]['depth_camera_1stperson'] 

    def save_img(self, observations):
        ds=1
        pov_ext="rgba_camera_3rdperson"
        pov_rgb="rgba_camera_1stperson"
        pov_depth="depth_camera_1stperson"
        frame_ext =  cv2.cvtColor(np.uint8(observations[0][pov_ext]),cv2.COLOR_RGB2BGR)
        frame_rgb =  cv2.cvtColor(np.uint8(observations[0][pov_rgb]),cv2.COLOR_RGB2BGR)
        frame_depth =  cv2.cvtColor(np.uint8(observations[0][pov_depth]/ 10 * 255),cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(self.save_img_dir, 'img_' + str(self.ctr) + '.jpg'), frame_ext)
        cv2.imwrite(os.path.join(self.save_img_dir, 'rgb_img_' + str(self.ctr) + '.jpg'), frame_rgb)
        print('saved img')
        # cv2.imwrite(os.path.join(self.save_img_dir, 'depth_img_' + str(self.ctr) + '.jpg'), frame_depth)
        self.ctr +=1

    def make_video(self):
        time_str = datetime.now().strftime("_%d%m%y_%H_%M_")
        self.make_video_cv2(self.observations, ds=1, output_path=output_path +\
             time_str +  'pos_gain=' + str(self.pos_gain) + '_vel_gain=' + str(self.vel_gain) + '_finite_diff=' + str(self.finite_diff), pov='rgba_camera_3rdperson',fps=self.ctrl_freq)

    # This is wrapped such that it can be added to a unit test
    def test_robot(self):
        self.reset_robot()
        # Set desired linear and angular velocities
        print("MOVING FORWARD")
        self.cmd_vel_xyt(0.35, 0.0, 0.0)
        # print("MOVING BACKWARDS")
        # self.cmd_vel_xyt(-0.35, 0.0, 0.0)
        # print("MOVING RIGHT")
        # self.cmd_vel_xyt(0.0, -0.35, 0.0)
        # print("MOVING LEFT")
        # self.cmd_vel_xyt(0.0, 0.35, 0.0)
        # print("MOVING FORWARD ARC RIGHT")
        # self.cmd_vel_xyt(0.35, 0.0, -0.15)
        # print("MOVING FORWARD ARC LEFT")
        # self.cmd_vel_xyt(0.35, 0.0, 0.15)

        self.make_video()
        with open(os.path.join(output_path, 'controller_log.json'), 'w') as f:
            print('Dumping data!!!')
            json.dump(self.raibert_infos, f)

    def get_episodes(self):
        start_poses = []
        goal_poses = []
        with open(self.episode_pth) as f:
            episode_dict = yaml.safe_load(f)
        for key in episode_dict:
            start_poses.append(np.array(episode_dict[key]['initial_pos']))
            goal_poses.append(np.array(episode_dict[key]['target_pos']))
        return np.array(start_poses), np.array(goal_poses)

    def get_robot_pos(self):
        robot_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)
        base_pos_hab = rotate_hab_pos(robot_state.translation)

        robot_position = np.array([base_pos_hab[0], base_pos_hab[1], base_pos_hab[2]])
        robot_ori = get_rpy(robot_state.rotation)
        return robot_position, robot_ori

    def get_goal_sensor(self, goal_pos):
        """
        :return: non-perception observation, such as goal location
        """
        robot_position, robot_ori = self.get_robot_pos()
        print('robot position: ', robot_position)
        agent_x, agent_y = robot_position[0], robot_position[1]
        agent_rotation = robot_ori[-1]
        goal_x, goal_y = goal_pos[0], goal_pos[1]
        goal = np.array([goal_x, goal_y])
        agent_coordinates = np.array([agent_x, agent_y])
        rho = np.linalg.norm(agent_coordinates - goal)
        theta = (
            np.arctan2(
                goal[1] - agent_coordinates[1], goal[0] - agent_coordinates[0]
            )
            - agent_rotation
        )
        theta = theta % (2 * np.pi)
        if theta >= np.pi:
            theta = -((2 * np.pi) - theta)
        print('goal: ', rho, theta, goal_pos)
        return rho, theta

    def evaluate_ddppo(self):
        start_poses, goal_poses = self.get_episodes()
        num_episodes = len(start_poses)
        # for episode in range(num_episodes):
        for episode in range(1):
            self.evaluate_episode(start_poses[episode], goal_poses[episode])

    def evaluate_episode(self, start_pos, goal_pos):
        self.reset_robot()
        self.done = False
        episode_reward = 0
        num_actions = 0
        num_processes=1
        test_recurrent_hidden_states = torch.zeros(
                1, 
                self.model.net.num_recurrent_layers,
                512,
                device=self.device,
            )
        prev_actions = torch.zeros(num_processes, self.dim_actions, device=self.device)
        not_done_masks = torch.zeros(num_processes, 1, dtype=torch.bool, device=self.device)
        self.depth_obs = np.zeros((240, 320))

        while not self.done:
            observations = {}

            # observations["depth"] = self.depth_obs
            observations["depth"] = np.expand_dims(self.depth_obs, axis=2)
            print(self.depth_obs.shape)
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
            prev_actions.copy_(actions)
            not_done_masks = torch.ones(num_processes, 1, dtype=torch.bool, device=self.device)

            actions = actions.squeeze()
            print('gaussian actions: ', actions)
            move_amount = -torch.tanh(actions[0]).item()
            turn_amount = torch.tanh(actions[1]).item()
            strafe_amount = torch.tanh(actions[2]).item() if self.dim_actions == 3 else 0
            if self.allow_backwards:
                move_amount *= self.linear_velocity_x
            else:
                move_amount = (move_amount+1.)/2.*self.linear_velocity_x
            turn_amount *= self.angular_velocity
            strafe_amount *= self.linear_velocity_y
            print('[commands]: {} {} {}'.format(move_amount, strafe_amount, turn_amount))

            if move_amount/self.linear_velocity_x < 0.1 and strafe_amount/self.linear_velocity_x < 0.1 and abs(turn_amount/self.angular_velocity) < 0.05:
                print('[STOP HAS BEEN CALLED]')
                return

            # observations = env.step(actions)
            action = np.array([move_amount, strafe_amount, turn_amount])
            print('NUM_ACTIONS: ', num_actions)
            self.depth_obs = self.step_robot(action) 
            num_actions +=1
            # episode_reward += reward
        # self.make_video()


if __name__ == "__main__":
    robot = sys.argv[1]
    W = Workspace(robot)
    W.evaluate_ddppo()
