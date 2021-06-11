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
from utilities import utils

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../habitat-sim/data")
output_path = os.path.join(dir_path, "spot_videos/")

class Workspace(object):
    def __init__(self, robot):
        self.raibert_infos = {}
        self.ep_id = 1
        self.observations = []
        self.depth_ortho_imgs = []
        self.text = []
        self.pos_gain = np.ones((3,)) * 0.15 # 0.2 
        self.vel_gain = np.ones((3,)) * 1.5 # 1.5
        self.pos_gain[2] = 0.1 # 0.7
        self.vel_gain[2] = 1.5 # 1.5
        self.num_steps = 5
        self.ctrl_freq = 240
        self.time_per_step = 100
        self.prev_state = None
        self.finite_diff = True
        self.robot_name = robot
        self.done = False
        self.num_actions = 0

    def make_configuration(self, scene):
        # simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        # backend_cfg.scene_dataset_config_file = "../data/default.scene_dataset_config.json"
        # backend_cfg.physics_config_file = "../data/default.physics_config.json"
        backend_cfg.scene_id = scene
        # backend_cfg.scene_id = "data/scene_datasets/coda/empty_room.stage_config.json"
        # backend_cfg.scene_id = 
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
        agent_state.position = [-0.15, -0.7, 1.0]
        agent_state.rotation = np.quaternion(-0.83147, 0, 0.55557, 0)
        # agent_state.position = [0.0, 0.0, 0.0]
        # agent_state.rotation = np.quaternion(1, 0, 0, 0)

        # agent_state.rotation = np.quaternion(1, 0, 0, 0)
        self.agent = self.sim.initialize_agent(0, agent_state)

    def place_camera_agent(self):
        agent_state2 = habitat_sim.AgentState()
        # agent_state2.position = [-6.0,2.0,4.0]
        # agent_rotation = squaternion.Quaternion.from_euler(np.deg2rad(-10.0),np.deg2rad(-80.0),np.deg2rad(0.0), degrees=False)
        agent_state2.position = [-4.0,2.0,5.0]
        agent_rotation = squaternion.Quaternion.from_euler(np.deg2rad(-20.0),np.deg2rad(-80.0),np.deg2rad(0.0), degrees=False)
        agent_state2.rotation = utils.quat_from_magnum(agent_rotation)
        # agent_state.rotation = np.quaternion(1, 0, 0, 0)
        agent2 = self.sim.initialize_agent(1, agent_state2)

    def setup(self, scene):
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # [initialize]
        # create the simulator
        cfg = self.make_configuration(scene)
        self.sim = habitat_sim.Simulator(cfg)
        
    def load_robot(self):
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
        print(robot_file)
        self.robot_id = self.sim.add_articulated_object_from_urdf(robot_file, fixed_base=False)

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

        # existing_joint_motors = sim.get_existing_joint_motors(robot_id)    
        agent_config = habitat_sim.AgentConfiguration()
        scene_graph = habitat_sim.SceneGraph()
        agent = habitat_sim.Agent(scene_graph.get_root_node().create_child(), agent_config)

        if self.robot_name == 'A1':
            self.robot = A1(sim=self.sim, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)
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

    def reset_robot(self, start_pose):
        local_base_pos = start_pose

        agent_transform = self.sim.agents[0].scene_node.transformation_matrix()
        base_transform = mn.Matrix4.rotation(mn.Rad(-1.57), mn.Vector3(1.0, 0, 0))
        base_transform.translation = agent_transform.transform_point(local_base_pos)
        self.sim.set_articulated_object_root_state(self.robot_id, base_transform)

        self.init_state = self.robot.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
        self.raibert_controller.set_init_state(self.init_state)
        time.sleep(1)

        pos, ori = self.get_robot_pos()
        print('robot start pos: ', pos, np.rad2deg(ori[-1]))

    def get_robot_pos(self):
        robot_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)
        base_pos_hab = utils.rotate_pos_from_hab(robot_state.translation)

        robot_position = np.array([base_pos_hab[0], base_pos_hab[1], base_pos_hab[2]])
        robot_ori = utils.get_rpy(robot_state.rotation) 
        return robot_position, robot_ori

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
        self.raibert_infos[str(self.ep_id)]["raibert_base_ang_velocity"] = [[float(iii) for iii in ii] for ii in self.raibert_base_ang_velocity]
        self.raibert_infos[str(self.ep_id)]["raibert_base_ori_euler"] = [[float(iii) for iii in ii] for ii in self.raibert_base_ori_euler]
        self.raibert_infos[str(self.ep_id)]["raibert_base_ori_quat"] = [[float(iii) for iii in ii] for ii in self.raibert_base_ori_quat]

    def cmd_vel_xyt(self, lin_x, lin_y, ang, log=True):
        for n in range(self.num_steps):
            action = np.array([lin_x, lin_y, ang])
            self.step_robot(action) 
            if log:
                self.log_raibert_controller()
            self.ep_id +=1
            self.num_actions +=1

    def step_robot(self, action):
        state = self.robot.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
        target_speed = np.array([action[0], action[1]])
        target_ang_vel = action[2]
        self.target_speed = list(target_speed)
        self.target_ang_vel = target_ang_vel
        self.input_current_speed = state['base_velocity'][0:2]
        self.input_joint_pos = state['j_pos']
        self.input_current_yaw_rate = state['base_ang_vel'][2]
        # Get initial latent action (not necessary)
        latent_action = self.raibert_controller.plan_latent_action(state, target_speed, target_ang_vel=target_ang_vel)
        self.latent_action = latent_action
        self.input_base_ori_euler = state['base_ori_euler']

        # Update latent action in controller 
        self.raibert_controller.update_latent_action(state, latent_action)
        raibert_actions_commanded = []
        raibert_actions_measured = []
        raibert_base_velocity = []
        raibert_base_ang_velocity = []
        raibert_base_ori_euler = []
        raibert_base_ori_quat = []
        raibert_base_ori_quat_hab = []
        for i in range(self.time_per_step):
            # Get actual joint actions 
            self.prev_state = state
            raibert_action = self.raibert_controller.get_action(state, i+1)
            # Simulate spot for 1/ctrl_freq seconds and return camera observation
            raibert_actions_commanded.append(raibert_action)
            agent_obs, ortho_obs = self.robot.step(raibert_action, self.pos_gain, self.vel_gain, dt=1/self.ctrl_freq, follow_robot=True)
            # self.observations += depth_obs
            # print(cur_obs[0]['depth_camera_1stperson'], cur_obs[0]['depth_camera_1stperson'].shape)

            state = self.robot.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
            self.check_done(action, state)
            if self.done:
                break
            # Get text to add to video
            text_to_add = []
            text_to_add.append("Pos: [" + str(np.round(state['base_pos'][0], 3)) + ", " + str(np.round(state['base_pos'][1], 3)) +\
            ", " + str(np.round(state['base_pos'][2], 3)) +  "]")
            text_to_add.append("Ori: [" + str(np.round(state['base_ori_euler'][0], 3)) + ", " + str(np.round(state['base_ori_euler'][1], 3)) +\
            ", " + str(np.round(state['base_ori_euler'][2], 3)) +  "]")
            text_to_add.append("Vel_lin: [" + str(np.round(state['base_velocity'][0], 3)) + ", " + str(np.round(state['base_velocity'][1], 3)) +\
            ", " + str(np.round(state['base_velocity'][2], 3)) +  "]")
            text_to_add.append("Vel_ang: [" + str(np.round(state['base_ang_vel'][0], 3)) + ", " + str(np.round(state['base_ang_vel'][1], 3)) +\
            ", " + str(np.round(state['base_ang_vel'][2], 3)) +  "]")
            text_to_add.append("Commanded Vel (x,y,ang): (" + str(target_speed) + " " +str(target_ang_vel) + ")")
            text_to_add.append("Pos Gain: " + str(self.pos_gain) + " Vel Gain: " +str(self.vel_gain))
            text_to_add.append("Action #: " + str(self.num_actions))
            self.text.append(text_to_add)

            self.stitch_show_img(agent_obs, ortho_obs)  
            # Recalculate spot state for next action
            base_ori_quat = np.array([state['base_ori_quat'].x, state['base_ori_quat'].y, state['base_ori_quat'].z, state['base_ori_quat'].w])
            raibert_base_velocity.append(state['base_velocity'])
            raibert_base_ang_velocity.append(state['base_ang_vel'])
            raibert_base_ori_euler.append(state['base_ori_euler'])
            raibert_base_ori_quat.append(base_ori_quat)
            raibert_actions_measured.append(state['j_pos'])
        self.raibert_action_commanded = raibert_actions_commanded
        self.raibert_action_measured = raibert_actions_measured
        self.raibert_base_velocity = raibert_base_velocity
        self.raibert_base_ang_velocity = raibert_base_ang_velocity
        self.raibert_base_ori_euler = raibert_base_ori_euler
        self.raibert_base_ori_quat = raibert_base_ori_quat
        return agent_obs[0]['depth_camera_1stperson']

    def stitch_show_img(self, agent_obs, ortho_obs):
        depth_img = cv2.cvtColor(np.uint8(agent_obs[0]['depth_camera_1stperson']/ 10 * 255),cv2.COLOR_RGB2BGR)
        ortho_img = cv2.cvtColor(np.uint8(ortho_obs[0]['ortho']),cv2.COLOR_RGB2BGR)

        height,width,layers = ortho_img.shape
        resize_depth_img = cv2.resize(depth_img, (width, height))
        frames = np.concatenate((resize_depth_img, ortho_img), axis=1)
        self.depth_ortho_imgs.append(frames)

    def check_make_dir(self, pth):
        if not os.path.exists(pth):
            os.makedirs(pth)
        return

    def save_video(self, rate, save_name):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        self.check_make_dir(output_path)

        self.depth_ortho_imgs = self.depth_ortho_imgs[1::rate]
        if self.text is not None:
            self.text= self.text[1::rate]

        print(os.path.join(output_path, save_name))
        video=cv2.VideoWriter(os.path.join(output_path, save_name),fourcc,10,(1440,540))
        for idx, frame in enumerate(self.depth_ortho_imgs):
            if self.text is not None:
                for i, line in enumerate(self.text[idx]):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, line, (20, 100 + i*30), font, 0.5, (0, 0, 0), 2)
            video.write(frame)
        print('SAVED VIDEO')

    def check_done(self, action, state):
        pass

    # This is wrapped such that it can be added to a unit test
    def test_robot(self):
        self.reset_robot(np.array([-2,1.3,-4]))
        # Set desired linear and angular velocities
        # print("MOVING FORWARD")
        # self.cmd_vel_xyt(0.35, 0.0, 0.0)
        # print("MOVING BACKWARDS")
        # self.cmd_vel_xyt(-0.35, 0.0, 0.0)
        # print("MOVING RIGHT")
        # self.cmd_vel_xyt(0.0, -0.35, 0.0)
        # print("MOVING LEFT")
        # self.cmd_vel_xyt(0.0, 0.35, 0.0)
        # print("MOVING FORWARD ARC RIGHT")
        # self.cmd_vel_xyt(0.35, 0.0, -0.15)
        print("MOVING FORWARD ARC LEFT")
        self.cmd_vel_xyt(0.35, 0.0, 0.15)
        # print('TURNING IN PLACE LEFT')
        # self.cmd_vel_xyt(0.0, 0.0, 0.15)

        time_str = datetime.now().strftime("_%d%m%y_%H_%M_")
        save_name = time_str + 'pos_gain=' + str(self.pos_gain) + '_vel_gain=' + \
                    str(self.vel_gain) + '_finite_diff=' + str(self.finite_diff) + '.mp4'
        rate = self.ctrl_freq // 30
        self.save_video(rate, save_name)
        with open(os.path.join(output_path, 'controller_log.json'), 'w') as f:
            print('Dumping data!!!')
            json.dump(self.raibert_infos, f)

if __name__ == "__main__":
    robot = sys.argv[1]
    scene = "data/scene_datasets/coda/empty_room.glb"
    W = Workspace(robot)
    W.setup(scene)
    W.place_agent()
    W.place_camera_agent()
    W.load_robot()
    W.test_robot()
