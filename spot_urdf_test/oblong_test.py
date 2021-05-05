# [setup]
import os
import sys
import magnum as mn
import numpy as np
import math
from datetime import datetime
import habitat_sim
import squaternion

# import habitat_sim.utils.common as ut
import habitat_sim.utils.viz_utils as vut
from utilities.quadruped_env import A1, AlienGo, Laikago, Spot
from utilities.daisy_env import Daisy, Daisy_4legged
from utilities.raibert_controller import Raibert_controller
from utilities.raibert_controller import Raibert_controller_turn
from utilities.utils import get_rpy, rotate_pos_from_hab, rotate_pos_to_hab
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
        self.num_steps = 80
        self.ctrl_freq = 240
        self.time_per_step = 80
        self.prev_state=None
        self.finite_diff=False
        self.robot_name = robot
        self.ctr = 0
        self.save_img_dir = 'oblong_imgs/'
        self.prev_angle = 0.
        self.setup()

    def make_configuration(self):
        # self.simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = "data/scene_datasets/coda/empty_room.glb"
        # backend_cfg.scene_id = "data/scene_datasets/coda/coda_hard.glb"
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
            # "position": [-2.0,2.0,0.0],#[0.0, 1.0, 0.3],
            # "orientation": [0.0,0.0,0.0],#[-45, 0.0, 0.0],
            # "position": [0.0,3.0,1.0],#[0.0, 1.0, 0.3],
            # "orientation": [0.0,np.deg2rad(90),np.deg2rad(20)],#[-45, 0.0, 0.0],
            "position": [-9.0,4.0,-8.0],#[0.0, 1.0, 0.3],
            "orientation": [np.deg2rad(- 10), np.deg2rad(-120),0.0],#[-45, 0.0, 0.0],
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
        agent_state.position = [-0.15, -0.7, 1.0]
        # agent_state.position = [-0.15, -0.7, 1.0]
        # agent_state.position = [-0.15, -1.0, 1.0]
        agent_state.rotation = np.quaternion(-0.83147, 0, 0.55557, 0)
        self.agent = self.sim.initialize_agent(0, agent_state)
        return self.agent.scene_node.transformation_matrix()


    def setup(self):
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # [initialize]
        # create the self.simulator
        cfg = self.make_configuration()
        
        self.sim = habitat_sim.Simulator(cfg)
        agent_transform = self.place_agent()

        # [/initialize]
        urdf_files = {
            "Spot": os.path.join(
                data_path, "URDF_demo_assets/spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid.urdf"
            ),
            "Can": os.path.join(
                data_path, "URDF_demo_assets/objects/chefcan.glb"
            )
        }

        # [basics]
        # load a URDF file
        robot_file_name = self.robot_name
        robot_file = urdf_files[robot_file_name]
        self.robot_id = self.sim.add_articulated_object_from_urdf(robot_file, fixed_base=False)
        obj_mgr = self.sim.get_object_template_manager()
        self.cube_id = self.sim.add_object_by_handle(obj_mgr.get_template_handles("cube")[0])
        self.cube_id_2 = self.sim.add_object_by_handle(obj_mgr.get_template_handles("cube")[0])

        self.sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, self.cube_id)
        self.sim.set_object_is_collidable(False, self.cube_id)

        self.sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, self.cube_id_2)
        self.sim.set_object_is_collidable(False, self.cube_id_2)
        # can_id1 = self.sim.add_articulated_object_from_urdf(urdf_files['Can'], fixed_base=False)
        # can_id2 = self.sim.add_articulated_object_from_urdf(urdf_files['Can'], fixed_base=False)
        
        local_base_pos = np.array([-2,1.3,-5]) # original 
        agent_transform = self.sim.agents[0].scene_node.transformation_matrix()
        base_transform = mn.Matrix4.rotation(mn.Rad(-1.57), mn.Vector3(1.0, 0, 0))
        base_transform.translation = agent_transform.transform_point(local_base_pos)
        self.sim.set_articulated_object_root_state(self.robot_id, base_transform)

        link_rigid_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)
        print('robot start: ', link_rigid_state.translation)
        tmp = link_rigid_state.translation
        copy1 = mn.Vector3(tmp.x, tmp.y, tmp.z)
        copy2 = mn.Vector3(tmp.x, tmp.y, tmp.z)

        cube_1_translation = copy1
        cube_1_translation.x = copy1.x + 0.3
        self.sim.set_translation(cube_1_translation, self.cube_id)
        self.sim.set_rotation(link_rigid_state.rotation, self.cube_id)

        cube_2_translation = copy2
        cube_2_translation.x = copy2.x - 0.3
        # cube_1_translation.x = copy1.x + 0.3*np.cos(theta[-1])
        # cube_1_translation.z = copy1.z - 0.3*np.sin(np.deg2rad(90)-theta[-1])
        self.sim.set_translation(cube_2_translation, self.cube_id_2)
        self.sim.set_rotation(link_rigid_state.rotation, self.cube_id_2)

        print('cube1 start: ', self.sim.get_translation(self.cube_id))
        print('cube2 start: ', self.sim.get_translation(self.cube_id_2))
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
        # self.sim.set_articulated_object_root_state(robot_id, base_transform)


        # existing_joint_motors = self.sim.get_existing_joint_motors(robot_id)    
        agent_config = habitat_sim.AgentConfiguration()
        scene_graph = habitat_sim.SceneGraph()
        agent = habitat_sim.Agent(scene_graph.get_root_node().create_child(), agent_config)


        self.robot = Spot(sim=self.sim, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)
            
        self.robot.robot_specific_reset()
        
        # Set up Raibert controller and link it to spot
        action_limit = np.zeros((12, 2))
        action_limit[:, 0] = np.zeros(12) + np.pi / 2
        action_limit[:, 1] = np.zeros(12) - np.pi / 2
        self.raibert_controller = Raibert_controller_turn(control_frequency=self.ctrl_freq, num_timestep_per_HL_action=self.time_per_step, robot=self.robot_name)

    # [/setup]
    def reset_robot(self):
        self.init_state = self.robot.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
        self.raibert_controller.set_init_state(self.init_state)
        time.sleep(1)

    def cmd_vel_xyt(self, lin_x, lin_y, ang):
        for n in range(self.num_steps):
            action = np.array([lin_x, lin_y, ang])
            self.step_robot(action) 
            self.ep_id +=1

    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy   

    def set_cube_pos(self):
        link_rigid_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)
        theta = get_rpy(link_rigid_state.rotation)

        tmp = link_rigid_state.translation
        copy1 = mn.Vector3(tmp.x, tmp.y, tmp.z)
        copy2 = mn.Vector3(tmp.x, tmp.y, tmp.z)
        print('robot: ', link_rigid_state.translation, np.rad2deg(theta[-1]), np.cos(theta[-1]), np.sin(theta[-1]))

        cube_1_translation = copy1
        cube_1_translation.x = copy1.x + 0.3*np.cos(theta[-1])
        cube_1_translation.z = copy1.z - 0.3*np.sin(theta[-1])
        self.sim.set_translation(cube_1_translation, self.cube_id)
        self.sim.set_rotation(link_rigid_state.rotation, self.cube_id)
        print('cube1: ', cube_1_translation)

        cube_2_translation = copy2
        cube_2_translation.x = copy2.x - 0.3*np.cos(theta[-1])
        cube_2_translation.z = copy2.z + 0.3*np.sin(theta[-1])
        self.sim.set_translation(cube_2_translation, self.cube_id_2)
        self.sim.set_rotation(link_rigid_state.rotation, self.cube_id_2)
        print('cube2: ', cube_2_translation)
        self.prev_angle = theta[-1]

    def set_cube_pos_v2(self):
        link_rigid_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)
        theta = get_rpy(link_rigid_state.rotation)

        base_position = link_rigid_state.translation
        print('base_position: ', base_position)
        base_pos_tmp = rotate_pos_from_hab(link_rigid_state.translation)
        print('base_pos_tmp: ', base_pos_tmp)
        base_pos_inv = rotate_pos_to_hab(base_position)
        print('base_pos_inv: ', base_pos_inv)

        base_position.x = base_pos_tmp[0]
        base_position.y = base_pos_tmp[1]
        base_position.z = base_pos_tmp[2]

        tmp = link_rigid_state.translation
        copy1 = np.array([base_position.x, base_position.y, base_position.z])
        copy2 = np.array([base_position.x, base_position.y, base_position.z])

        cube_1_translation = base_pos_tmp
        print('cube_1_translation: ', cube_1_translation)
        cube_1_translation[0] = copy1[0] + 0.3*np.cos(self.prev_angle - theta[-1])
        cube_1_translation[1] = copy1[1] + 0.3*np.sin(self.prev_angle - theta[-1])

        cube_1_translation_inv = rotate_pos_to_hab(cube_1_translation)
        print('cube_1_translation_inv: ', cube_1_translation_inv)

        self.sim.set_translation(cube_1_translation_inv, self.cube_id)
        self.sim.set_rotation(link_rigid_state.rotation, self.cube_id)


        cube_2_translation = base_pos_tmp
        cube_2_translation[0] = copy2[0] - 0.3*np.cos(self.prev_angle - theta[-1])
        cube_2_translation[1] = copy2[1] - 0.3*np.sin(self.prev_angle - theta[-1])

        print('cube_2_translation: ', cube_2_translation)
        cube_2_translation_inv = rotate_pos_to_hab(cube_2_translation)
        print('cube_2_translation_inv: ', cube_2_translation_inv)
        self.sim.set_translation(cube_2_translation_inv, self.cube_id_2)
        self.sim.set_rotation(link_rigid_state.rotation, self.cube_id_2)
        print('cube_2_translation_inv: ', cube_2_translation_inv)
        self.prev_angle = theta[-1]

    def step_robot(self, action):
        state = self.robot.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
        target_speed = np.array([action[0], action[1]])
        target_ang_vel = action[2]
        # print('input current speed: ', self.input_current_speed, 'input_current_yaw_rate: ', self.input_current_yaw_rate)

        # Get initial latent action (not necessary)
        latent_action = self.raibert_controller.plan_latent_action(state, target_speed, target_ang_vel=target_ang_vel)
        # latent_action = raibert_controller.plan_latent_action(state, target_speed, target_ang_vel=target_ang_vel)

        # Update latent action in controller 
        self.raibert_controller.update_latent_action(state, latent_action)
        raibert_actions_commanded = []
        raibert_actions_measured = []
        raibert_base_velocity = []
        for i in range(self.time_per_step):
            # Get actual joint actions 
            self.prev_state = state
            raibert_action = self.raibert_controller.get_action(state, i+1)
            # self.simulate spot for 1/ctrl_freq seconds and return camera observation
            raibert_actions_commanded.append(raibert_action)
            cur_obs = self.robot.step(raibert_action, self.pos_gain, self.vel_gain, dt=1/self.ctrl_freq, follow_robot=False)
            # print(cur_obs[0]['depth_camera_1stperson'], cur_obs[0]['depth_camera_1stperson'].shape)

            # Recalculate spot state for next action
            state = self.robot.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
            raibert_base_velocity.append(state['base_velocity'])
            raibert_actions_measured.append(state['j_pos'])
        self.set_cube_pos()
        self.save_img(cur_obs)

    def save_img(self, observations):
        ds=1
        pov_ext="rgba_camera_3rdperson"
        pov_rgb="rgba_camera_1stperson"
        pov_depth="depth_camera_1stperson"
        frame_ext =  cv2.cvtColor(np.uint8(observations[0][pov_ext]),cv2.COLOR_RGB2BGR)
        frame_rgb =  cv2.cvtColor(np.uint8(observations[0][pov_rgb]),cv2.COLOR_RGB2BGR)
        frame_depth =  cv2.cvtColor(np.uint8(observations[0][pov_depth]/ 10 * 255),cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.save_img_dir, 'img_' + str(self.ctr) + '.jpg'), frame_ext)
        # cv2.imwrite(os.path.join(self.save_img_dir, 'rgb_img_' + str(self.ctr) + '.jpg'), frame_rgb)
        print('saved img')
        # cv2.imwrite(os.path.join(self.save_img_dir, 'depth_img_' + str(self.ctr) + '.jpg'), frame_depth)
        self.ctr +=1

    # This is wrapped such that it can be added to a unit test
    def test_robot(self):
        self.reset_robot()
        # Set desired linear and angular velocities
        # print("MOVING FORWARD")
        # self.cmd_vel_xyt(0.35, 0.0, 0.0)
        # print("MOVING BACKWARDS")
        # self.cmd_vel_xyt(-0.35, 0.0, 0.0)
        print("MOVING RIGHT")
        self.cmd_vel_xyt(0.0, -0.35, 0.0)
        # print("MOVING LEFT")
        # self.cmd_vel_xyt(0.0, 0.35, 0.0)
        # print("MOVING FORWARD ARC RIGHT")
        # self.cmd_vel_xyt(0.35, 0.0, -0.15)
        # print("MOVING FORWARD ARC LEFT")
        # self.cmd_vel_xyt(0.35, 0.0, 0.15)

    def get_scalar_vector(self, quat):
        scalar = np.arccos(quat.normalize.scalar)*2
        vector = quat.normalize.vector/np.sin(scalar/2)
        return scalar, vector

    def get_quat(self, scalar, vector):
        new_scalar = np.cos(scalar/2)
        new_vector = np.array(vector)*np.sin(scalar/2)
        quat = squaternion.Quaternion(new_scalar, *new_vector)
        return quat

    

    def simulate(self, dt=1.0, get_frames=True, show=True, text=None):
        global last_render
        # simulate dt seconds at 60Hz to the nearest fixed timestep
        # print("Simulating " + str(dt) + " world seconds.")
        observations = []
        start_time = self.sim.get_world_time()
        while self.sim.get_world_time() < start_time + dt:
            self.sim.step_physics(1.0 / 60.0)
            if get_frames:
                observation = self.sim.get_sensor_observations()
                if show:
                    img = cv2.cvtColor(observation['rgba_camera_3rdperson'], cv2.COLOR_RGB2BGR)
                    if text is not None:
                        height = img.shape[0]
                        cv2.putText(img, text, (0,height-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
                    wait_ms = max(1,int(1000*round(self.last_render+0.017 - time.time())))
                    key = cv2.waitKey(wait_ms)
                    if key == ord('q'):
                        exit()
                    cv2.imshow('aliengo',img)
                    self.last_render = time.time()
                observation['rgba_camera_3rdperson'] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                observations.append(observation)
        return observations

    def test_cube(self):
        self.reset_robot()
        import time
        self.last_render = time.time()
        quats = []
        for i in range(1,9):
            quats.append(squaternion.Quaternion.from_euler(0, i*10, 0, degrees=True))
        base_quat     = self.get_quat(-np.pi/2,(1,0,0))
        print('base_quat: ', base_quat)
        # inv_base_quat = get_quat(-np.pi/2,(-1,0,0))
        inv_base_quat = self.get_quat(np.pi/2,(1,0,0))
        observations = []
        for quat in quats:
            new_quat = quat * base_quat

            scalar, vector = self.get_scalar_vector(new_quat)
            base_transform = mn.Matrix4.rotation(mn.Rad(scalar), mn.Vector3(*vector))
            link_rigid_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)
            base_transform.translation = link_rigid_state.translation
            self.sim.set_articulated_object_root_state(self.robot_id, base_transform)
            self.set_cube_pos()

            self.sim.step_physics(60.0)
            # link_rigid_state = sim.get_articulated_link_rigid_state(robot_id, 17)
            link_rigid_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)


            # sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, cube_id)
            # sim.set_translation(link_rigid_state.translation, cube_id)
            # sim.set_rotation(link_rigid_state.rotation, cube_id)
            self.sim.step_physics(1.0 / 60.0)


            print(link_rigid_state.translation, link_rigid_state.rotation)
            obs_quat = squaternion.Quaternion(link_rigid_state.rotation.scalar, *link_rigid_state.rotation.vector)
            # obs_quat = squaternion.Quaternion(mn_quat.scalar, *mn_quat.vector)
            obs_quat = obs_quat * inv_base_quat
            rpy = 'Roll: {:.2f} Yaw: {:.2f} Pitch: {:.2f}'.format(*obs_quat.to_euler(degrees=False))
            observations += self.simulate(dt=1, get_frames=True, text=rpy)

if __name__ == "__main__":
    W = Workspace('Spot')
    W.test_robot()
    # W.test_cube()
