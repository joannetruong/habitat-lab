import math
import numpy as np

from .quad_kinematics_solver import quadruped_kinematics_solver

EPSILON = 1e-4

a1_config = {
    'thigh_length': 0.2,
    'shank_length': 0.2,
    'hip_joint_pos': [0.183, 0.047, 0],
    'thigh_joint_loc': 0.08505,
}

aliengo_config = {
    'thigh_length': 0.25,
    'shank_length': 0.25,
    'hip_joint_pos': [0.2399, 0.051, 0],
    'thigh_joint_loc': 0.083,
}

laikago_config = {
    'thigh_length': 0.25,
    'shank_length': 0.25,
    'hip_joint_pos': [0.21935, 0.0875, 0.01675],
    'thigh_joint_loc': 0.037,
}

spot_config = {
    'thigh_length': 0.42,
    'shank_length': 0.42,
    'hip_joint_pos': [0.29785, 0.05500, 0.0],
    'thigh_joint_loc': -0.11,
}

class Raibert_controller():
    def __init__(self, robot='A1', num_timestep_per_HL_action=50, target=None, speed_gain=0.15, des_body_ori=None,
                 control_frequency=100, leg_set_1=[1,2], leg_set_2=[0,3], leg_clearance=0.15, action_limit=None):
        self.swing_set = leg_set_1
        self.stance_set = leg_set_2
        self.robot = robot
        if 'A1' in robot:
            self.kinematics_solver = quadruped_kinematics_solver(**a1_config)
        elif 'AlienGo' in robot:
            self.kinematics_solver = quadruped_kinematics_solver(**aliengo_config)
        elif '4legged' in robot:
            self.kinematics_solver = quadruped_daisy_kinematics_solver()
            self.swing_set = [1, 2]
            self.stance_set = [0, 3]
        elif 'Daisy' in robot: #this condition triggers for daisy_4legged too
            self.kinematics_solver = hexapod_kinematics_solver()
            self.swing_set = [1, 2, 5]
            self.stance_set = [0, 3, 4]
        elif 'Laikago' in robot:
            self.kinematics_solver = quadruped_kinematics_solver(**laikago_config)
        elif 'Spot' in robot:
            self.kinematics_solver = quadruped_kinematics_solver(**spot_config)


        self.control_frequency = control_frequency
        self.speed_gain = speed_gain 
        self.leg_clearance = leg_clearance
        self.num_timestep_per_HL_action = num_timestep_per_HL_action
        self.latent_action = None
        self.action_limit  = action_limit
        self.num_legs = self.kinematics_solver.num_legs
        self.n_dof = self.num_legs*self.kinematics_solver.num_motors_per_leg
        
        if target is None:
            self.target = np.array([0,0,0])
        else:
            self.target = target

        
        if action_limit is None:
            self.action_limit = np.zeros((self.n_dof, 2))
            self.action_limit[:, 0] = np.zeros(self.n_dof) + np.pi / 2.0
            self.action_limit[:, 1] = np.zeros(self.n_dof) - np.pi / 2.0
            
            # Add clipping limit on hip joints...controller is wack
            self.action_limit[[0,3,6,9], 0] = np.zeros(4) + np.pi / 20.0
            self.action_limit[[0,3,6,9], 1] = np.zeros(4) - np.pi / 20.0

        if des_body_ori is None:
            self.des_body_ori = np.array([0,0,0]) # this is for des orientation at each timestep
        else:
            self.des_body_ori = des_body_ori
        self.final_des_body_ori = self.des_body_ori

    def set_init_state(self, init_state, action_limits=None):
        self.init_state = init_state
        if action_limits is not None:
            j_pos = np.array(self.init_state['j_pos'])
            self.action_limit[:, 0] = j_pos + np.array([action_limits] * self.num_legs).reshape(self.n_dof)
            self.action_limit[:, 1] = j_pos - np.array([action_limits] * self.num_legs).reshape(self.n_dof)
        self.set_control_params(init_state)
        print('Set initial state')

    def set_control_params(self, state):
        self.init_foot_pos = self.kinematics_solver.forward_kinematics_robot(state['j_pos']).reshape(self.num_legs, 3)
        self.standing_height = -(self.init_foot_pos[0][2] + self.init_foot_pos[1][2])/2.0
        # here we set the des orientation to be only yaw/ but in theory can also set roll and pitch
        self.des_body_ori = np.array([0, 0, state['base_ori_euler'][2]])
        self.final_des_body_ori = np.array([0, 0, state['base_ori_euler'][2]])
        self.init_r_yaw = self.get_init_r_yaw(self.init_foot_pos)
        self.swing_start_foot_pos_robot = self.kinematics_solver.forward_kinematics_robot(state['j_pos'])
        self.swing_start_foot_pos_world = self.kinematics_solver.robot_frame_to_world_robot(state['base_ori_euler'],
                                                                                            self.swing_start_foot_pos_robot)
        # self.swing_start_foot_pos_world = self.init_foot_pos

    def get_init_r_yaw(self,init_foot_pos):
        r_yaw = np.zeros((self.num_legs,2))
        for i in range(self.num_legs):
            r_yaw[i][0] = np.linalg.norm(init_foot_pos[i])
            r_yaw[i][1] = math.atan2(init_foot_pos[i][1], init_foot_pos[i][0])
        return r_yaw

    def plan_latent_action(self, state, target_speed, target_ori=0.0):
        self.latent_action = np.zeros(3)
        self.target_speed = target_speed[:2]
        current_speed = state['base_velocity'][0:2]

        speed_term = self.num_timestep_per_HL_action / (self.control_frequency) * current_speed
        acceleration_term = -self.speed_gain * (current_speed - self.target_speed)
        orientation_speed_term = self.num_timestep_per_HL_action / self.control_frequency * target_ori

        des_footstep = (speed_term + acceleration_term)
        self.latent_action[0:2] = des_footstep
        self.latent_action[2] = orientation_speed_term
        print(self.latent_action)
        return self.latent_action

    def switch_swing_stance(self):
        self.swing_set, self.stance_set = np.copy(self.stance_set), np.copy(self.swing_set)

    def update_latent_action(self, state, latent_action):
        self.switch_swing_stance()
        self.latent_action = latent_action

        self.swing_start_foot_pos_robot = self.kinematics_solver.forward_kinematics_robot(state['j_pos'])

        self.last_com_ori = state['base_ori_euler']
        self.final_des_body_ori[2] = self.last_com_ori[2] + self.latent_action[-1]

        target_delta_xy = np.zeros(self.num_legs*3)
        for i in range(self.num_legs):
            if i in self.swing_set:
                target_delta_xy[3*i] = self.init_foot_pos[i][0] + self.latent_action[0] - \
                                        self.swing_start_foot_pos_robot[3*i]
                target_delta_xy[3*i+1] = self.init_foot_pos[i][1] + self.latent_action[1] - \
                                        self.swing_start_foot_pos_robot[3*i+1]
            else:
                target_delta_xy[3*i] = self.init_foot_pos[i][0] - self.latent_action[0] - \
                                        self.swing_start_foot_pos_robot[3*i]
                target_delta_xy[3*i+1] = self.init_foot_pos[i][1] - self.latent_action[1] - \
                                        self.swing_start_foot_pos_robot[3*i+1]
            target_delta_xy[3*i+2] = -self.standing_height
        # transform target x y to the 0 yaw frame
        self.target_delta_xyz = target_delta_xy

    def get_action(self, state, t):
        phase = float(t) / self.num_timestep_per_HL_action
        action = self._get_action(state, phase)
        action = np.clip(action, a_min=self.action_limit[:,1], a_max=self.action_limit[:,0])
        return action

    def _get_action(self, state, phase):
        self.des_foot_position_com = np.array([])
        self.des_body_ori[2] = (self.final_des_body_ori[2] - self.last_com_ori[2]) * phase + self.last_com_ori[2]
        # this seems to be designed only when walking on a flat ground
        des_foot_height_delta = (self.leg_clearance * math.sin(math.pi * phase + EPSILON))

        for i in range(self.num_legs):
            des_single_foot_pos = np.zeros(3)
            if i in self.swing_set:
                des_single_foot_pos[2] = des_foot_height_delta - self.standing_height
            else:
                des_single_foot_pos[2] = des_foot_height_delta*0.15 -self.standing_height

            des_single_foot_pos[0] = self.target_delta_xyz[3*i] * phase + self.swing_start_foot_pos_robot[3*i]
            des_single_foot_pos[1] = self.target_delta_xyz[3*i+1] * phase + self.swing_start_foot_pos_robot[3*i+1]
            self.des_foot_position_com = np.append(self.des_foot_position_com, des_single_foot_pos)

        des_leg_pose = self.kinematics_solver.inverse_kinematics_robot(self.des_foot_position_com)

        return des_leg_pose

    def reset(self, state):
        self.final_des_body_ori = np.array([0, 0, state['base_ori_euler'][2]])

class Raibert_controller_turn(Raibert_controller):
    def __init__(self, robot='A1', num_timestep_per_HL_action=50, target=None, speed_gain=0.15, des_body_ori=None,
                 control_frequency=100, leg_set_1=[1, 2], leg_set_2=[0, 3], leg_clearance=0.1, action_limit=None):
        Raibert_controller.__init__(
            self, robot=robot, num_timestep_per_HL_action=num_timestep_per_HL_action, target=target,
            speed_gain=speed_gain, des_body_ori=des_body_ori,
            control_frequency=control_frequency, leg_set_1=leg_set_1, leg_set_2=leg_set_2, leg_clearance=leg_clearance,
            action_limit=action_limit
        )

    def plan_latent_action(self, state, target_speed, target_ang_vel=0.0):
        current_speed = np.array([state['base_velocity'][0], state['base_velocity'][1]])
        current_yaw_rate = state['base_ang_vel'][2]
        self.latent_action = np.zeros(3)
        self.target_speed = target_speed[:2]
        # print("target ", self.target_speed, "current ", current_speed)

        acceleration_term = self.speed_gain*(self.target_speed-current_speed) + 0.5*current_speed*self.num_timestep_per_HL_action / self.control_frequency
        orientation_speed_term = self.speed_gain*(target_ang_vel-current_yaw_rate) + 0.5*self.num_timestep_per_HL_action / self.control_frequency * current_yaw_rate

        des_footstep = acceleration_term
        self.latent_action[0:2] = des_footstep
        self.latent_action[2] = orientation_speed_term
        print(self.latent_action)
        return self.latent_action

    def update_latent_action(self, state, latent_action):
        self.switch_swing_stance()
        self.latent_action = latent_action
        self.last_com_ori = np.array(state['base_ori_euler'])
        self.last_com_ori[-1] = 0.0
        self.final_des_body_ori[2] = self.latent_action[-1]

        target_delta_xy = np.zeros(self.num_legs * 3)
        for i in range(self.num_legs):
            angle = self.latent_action[-1]
            init_angle = self.init_r_yaw[i][1]
            if i in self.swing_set:
                # angle = self.latent_action[-1] + self.init_r_yaw[i][1]
                target_delta_xy[3 * i] = self.latent_action[0] + (
                            self.init_r_yaw[i][0] * math.cos(angle + init_angle) - self.init_r_yaw[i][0] * math.cos(
                        init_angle))
                target_delta_xy[3 * i + 1] = self.latent_action[1] + (
                            self.init_r_yaw[i][0] * math.sin(angle + init_angle) - self.init_r_yaw[i][0] * math.sin(
                        init_angle))
            else:
                target_delta_xy[3 * i] = -self.latent_action[0] - (
                            self.init_r_yaw[i][0] * math.cos(angle + init_angle) - self.init_r_yaw[i][0] * math.cos(
                        init_angle))
                target_delta_xy[3 * i + 1] = -self.latent_action[1] - (
                            self.init_r_yaw[i][0] * math.sin(angle + init_angle) - self.init_r_yaw[i][0] * math.sin(
                        init_angle))
            target_delta_xy[3 * i + 2] = -self.standing_height
        self.target_delta_xyz_world = self.kinematics_solver.robot_frame_to_world_robot(self.last_com_ori,
                                                                                        target_delta_xy)

    def _get_action(self, state, phase):
        self.des_foot_position_com = np.array([])
        self.des_body_ori = (self.final_des_body_ori - self.last_com_ori) * phase + self.last_com_ori
        # this seems to be designed only when walking on a flat ground
        des_foot_height_delta = (self.leg_clearance * math.sin(math.pi * phase + EPSILON))

        for i in range(self.num_legs):
            des_single_foot_pos = np.zeros(3)
            if i in self.swing_set:
                des_single_foot_pos[2] = des_foot_height_delta - self.standing_height
            else:
                des_single_foot_pos[2] = 0.1*des_foot_height_delta - self.standing_height

            des_single_foot_pos[0] = self.target_delta_xyz_world[3*i] * phase + self.swing_start_foot_pos_world[3*i]
            des_single_foot_pos[1] = self.target_delta_xyz_world[3*i+1] * phase + self.swing_start_foot_pos_world[3*i+1]
            # if i in self.swing_set:
            #     self.des_foot_position_com = np.append(self.des_foot_position_com,self.kinematics_solver.world_frame_to_robot_leg(state['base_ori_euler'], des_single_foot_pos))
            # else:
            self.des_foot_position_com = np.append(self.des_foot_position_com,self.kinematics_solver.world_frame_to_robot_leg(self.des_body_ori, des_single_foot_pos))

        des_leg_pose = self.kinematics_solver.inverse_kinematics_robot(self.des_foot_position_com)

        return des_leg_pose
