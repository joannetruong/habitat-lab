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

    def make_configuration(self):
        # simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/empty_room.glb"
        # backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/coda_hard.glb"
        backend_cfg.enable_physics = True

        # sensor configurations
        # Note: all sensors must have the same resolution
        # setup 2 rgb sensors for 1st and 3rd person views
        camera_resolution = [540, 720]
        sensors = {
            "rgba_camera_1stperson": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": camera_resolution,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "sensor_subtype": habitat_sim.SensorSubType.ORTHOGRAPHIC,
            },
            "depth_camera_1stperson": {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": camera_resolution,
                "position": [0,0,0.0],#[0.0, 0.6, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "sensor_subtype": habitat_sim.SensorSubType.ORTHOGRAPHIC,
            },
            "rgba_camera_3rdperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [-2.0,2.0,0.0],#[0.0, 1.0, 0.3],
            "orientation": [0.0,0.0,0.0],#[-45, 0.0, 0.0],
            # "position": [0.0,3.0,1.0],#[0.0, 1.0, 0.3],
            # "orientation": [0.0,np.deg2rad(90),np.deg2rad(20)],#[-45, 0.0, 0.0],
            # "position": [-2.0,3.50,10.0],#[0.0, 1.0, 0.3],
            # "orientation": [np.deg2rad(-10),np.deg2rad(20.0),0.0],#[-45, 0.0, 0.0],
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

    def place_agent(self, sim):
        # place our agent in the scene
        agent_state = habitat_sim.AgentState()
        agent_state.position = [-0.15, -0.7, 1.0]
        #agent_state.position = [-0.15, -0.7, 1.0]
        # agent_state.position = [-0.15, -1.6, 1.0]
        agent_state.rotation = np.quaternion(-0.83147, 0, 0.55557, 0)
        self.agent = sim.initialize_agent(0, agent_state)
        return self.agent.scene_node.transformation_matrix()

    def setup(self):
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # [initialize]
        # create the simulator
        cfg = self.make_configuration()
        
        sim = habitat_sim.Simulator(cfg)
        agent_transform = self.place_agent(sim)

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
        robot_id = sim.add_articulated_object_from_urdf(robot_file, fixed_base=False)
        
        # Set root state for the URDF in the sim relative to the agent and find the inverse transform for finding velocities later
        # local_base_pos = np.array([0,1.3,-3]) # forward pillar collision
        # local_base_pos = np.array([-1,1.3,-4]) # right pillar collision
        # local_base_pos = np.array([-3.0,1.3,4.5])
        local_base_pos = np.array([-2,1.3,-4]) # original 
        agent_transform = sim.agents[0].scene_node.transformation_matrix()
        base_transform = mn.Matrix4.rotation(mn.Rad(-1.57), mn.Vector3(1.0, 0, 0))
        base_transform.translation = agent_transform.transform_point(local_base_pos)
        sim.set_articulated_object_root_state(robot_id, base_transform)

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
            sim.update_joint_motor(robot_id, i, jms[np.mod(i,3)])

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
            self.robot = A1(sim=sim, agent=self.agent, robot_id=robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'AlienGo':
            self.robot = AlienGo(sim=sim, agent=self.agent, robot_id=robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'Daisy':
            self.robot = Daisy(sim=sim, agent=self.agent, robot_id=robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'Laikago':
            self.robot = Laikago(sim=sim, agent=self.agent, robot_id=robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'Daisy_4legged':
            self.robot = Daisy4(sim=sim, agent=self.agent, robot_id=robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'Spot':
            self.robot = Spot(sim=sim, agent=self.agent, robot_id=robot_id, dt=1/self.ctrl_freq)
            
        self.robot.robot_specific_reset()
        
        # Set up Raibert controller and link it to spot
        action_limit = np.zeros((12, 2))
        action_limit[:, 0] = np.zeros(12) + np.pi / 2
        action_limit[:, 1] = np.zeros(12) - np.pi / 2
        self.raibert_controller = Raibert_controller_turn(control_frequency=self.ctrl_freq, num_timestep_per_HL_action=self.time_per_step, robot=self.robot_name)

    def make_video_cv2(self, observations, ds=1, output_path = None, fps=60, pov="rgba_camera_3rdperson"):
        if output_path is None:
            return False

        shp = self.observations[0][pov].shape
        
        videodims = (shp[1]//ds, shp[0]//ds)
        
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        vid_name = output_path + ".mp4"
        rate = fps // 30
        self.observations = self.observations[1::rate]
        if self.text is not None:
            self.text= self.text[1::rate]
        video = cv2.VideoWriter(vid_name, fourcc, 30, videodims)
        print('Formatting Video')
        for count, ob in enumerate(self.observations):
            if 'depth' in pov:
                
                ob[pov] = ob[pov][:,:,np.newaxis] / 10 * 255
                bgr_im_3rd_person = ob[pov] * np.ones((shp[0],shp[1], 3))

            else:
                bgr_im_3rd_person = ob[pov][...,0:3]
            
            frame =  cv2.cvtColor(np.uint8(bgr_im_3rd_person),cv2.COLOR_RGB2BGR)
            if self.text is not None:
                for i, line in enumerate(self.text[count]):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, line, (20, 100 + i*30), font, 0.5, (0, 0, 0), 2)
              
            video.write(frame)
        video.release()

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
        self.target_speed = list(target_speed)
        self.target_ang_vel = target_ang_vel
        self.input_current_speed = state['base_velocity'][0:2]
        self.input_joint_pos = state['j_pos']
        self.input_current_yaw_rate = state['base_ang_vel'][2]
        # print('input current speed: ', self.input_current_speed, 'input_current_yaw_rate: ', self.input_current_yaw_rate)

        # Get initial latent action (not necessary)
        latent_action = self.raibert_controller.plan_latent_action(state, target_speed, target_ang_vel=target_ang_vel)
        # latent_action = raibert_controller.plan_latent_action(state, target_speed, target_ang_vel=target_ang_vel)
        self.latent_action = latent_action
        self.input_base_ori_euler = state['base_ori_euler']

        # Update latent action in controller 
        self.raibert_controller.update_latent_action(state, latent_action)
        raibert_actions_commanded = []
        raibert_actions_measured = []
        raibert_base_velocity = []
        for i in range(self.time_per_step):
            # Get actual joint actions 
            self.prev_state = state
            raibert_action = self.raibert_controller.get_action(state, i+1)
            # Simulate spot for 1/ctrl_freq seconds and return camera observation
            raibert_actions_commanded.append(raibert_action)
            cur_obs = self.robot.step(raibert_action, self.pos_gain, self.vel_gain, dt=1/self.ctrl_freq, follow_robot=False)
            self.observations += cur_obs

            # Get text to add to video
            text_to_add = []
            text_to_add.append("Pos: [" + str(np.round(state['base_pos'][0], 3)) + ", " + str(np.round(state['base_pos'][1], 3)) +\
            ", " + str(np.round(state['base_pos'][2], 3)) +  "]")
            text_to_add.append("Vel_lin: [" + str(np.round(state['base_velocity'][0], 3)) + ", " + str(np.round(state['base_velocity'][1], 3)) +\
            ", " + str(np.round(state['base_velocity'][2], 3)) +  "]")
            # text_to_add.append("Vel_lin_b: [" + str(np.round(state['base_velocity_b'][0], 3)) + ", " + str(np.round(state['base_velocity_b'][1], 3)) +\
            # ", " + str(np.round(state['base_velocity_b'][2], 3)) +  "]")
            text_to_add.append("Vel_ang: [" + str(np.round(state['base_ang_vel'][0], 3)) + ", " + str(np.round(state['base_ang_vel'][1], 3)) +\
            ", " + str(np.round(state['base_ang_vel'][2], 3)) +  "]")
            # text_to_add.append("Vel_ang_b: [" + str(np.round(state['base_ang_vel_b]'[0], 3)) + ", " + str(np.round(state['base_ang_vel_b'][1], 3)) +\
            # ", " + str(np.round(state['base_ang_vel_b'][2], 3)) +  "]")
            text_to_add.append("Ori: [" + str(np.round(state['base_ori_euler'][0], 3)) + ", " + str(np.round(state['base_ori_euler'][1], 3)) +\
            ", " + str(np.round(state['base_ori_euler'][2], 3)) +  "]")
            text_to_add.append("Commanded Vel (x,y,ang): (" + str(target_speed) + " " +str(target_ang_vel) + ")")
            text_to_add.append("Pos Gain: " + str(self.pos_gain) + " Vel Gain: " +str(self.vel_gain))
            self.text.append(text_to_add)

            # Recalculate spot state for next action
            state = self.robot.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
            raibert_base_velocity.append(state['base_velocity'])
            raibert_actions_measured.append(state['j_pos'])
        self.raibert_action_commanded = raibert_actions_commanded
        self.raibert_action_measured = raibert_actions_measured
        self.raibert_base_velocity = raibert_base_velocity

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

if __name__ == "__main__":
    robot = sys.argv[1]
    W = Workspace(robot)
    W.test_robot()
