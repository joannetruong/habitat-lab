import math
import numpy as np

EPSILON = 1e-4

class quadruped_kinematics_solver():
    def __init__(self, thigh_length, shank_length, hip_joint_pos, thigh_joint_loc):
        # initial each leg start point position
        self.thigh_length = thigh_length
        self.shank_length = shank_length
        self.hip_joint_location = np.array([
            [ hip_joint_pos[0], -hip_joint_pos[1], hip_joint_pos[2]],
            [ hip_joint_pos[0],  hip_joint_pos[1], hip_joint_pos[2]],
            [-hip_joint_pos[0], -hip_joint_pos[1], hip_joint_pos[2]],
            [-hip_joint_pos[0],  hip_joint_pos[1], hip_joint_pos[2]],
            ])
        self.thigh_joint_loc = thigh_joint_loc
        self.rpy = np.zeros(3)
        self.num_legs = 4
        self.num_motors_per_leg = 3


    def gen_translation_matrix(self, value):
        return  np.array([
                [1, 0, 0, value[0]],
                [0, 1, 0, value[1]],
                [0, 0, 1, value[2]],
                [0, 0, 0, 1],
                ])


    def gen_rotation_matrix(self, value, axis):
        cos_value, sin_value = math.cos(value), math.sin(value)
        if axis==0:
            rot_matrix = [
                        [1, 0, 0, 0],
                        [0, cos_value , -sin_value , 0],
                        [0, sin_value , cos_value , 0],
                        [0, 0, 0, 1],
                        ]   
        if axis==1:
            rot_matrix = [
                        [cos_value ,0, sin_value , 0],
                        [0, 1, 0, 0],
                        [-sin_value ,0, cos_value, 0],
                        [0, 0, 0, 1],
                        ]   
        if axis==2:
            rot_matrix = [
                        [cos_value , -sin_value , 0, 0],
                        [sin_value , cos_value, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        ]   
        return rot_matrix


    def world2com_ori(self, com_ori):
        roll = self.gen_rotation_matrix(com_ori[0],0)
        pitch = self.gen_rotation_matrix(com_ori[1],1)
        yaw = self.gen_rotation_matrix(com_ori[2],2)
        return np.dot(np.dot(roll,pitch),yaw)
    

    def com2world_ori(self, com_ori):
        roll = self.gen_rotation_matrix(-com_ori[0],0)
        pitch = self.gen_rotation_matrix(-com_ori[1],1)
        yaw = self.gen_rotation_matrix(-com_ori[2],2)
        return np.dot(np.dot(yaw,pitch),roll)


    def world_frame_to_robot_robot(self, com_ori, foot_pos_world):
        foot_pos_robot = np.array([])
        for i in range(self.num_legs):
            foot_pos_robot_leg = self.world_frame_to_robot_leg(com_ori, foot_pos_world[3*i:3*i+3])
            foot_pos_robot = np.append(foot_pos_robot, foot_pos_robot_leg)
        return foot_pos_robot


    def world_frame_to_robot_leg(self, com_ori, foot_pos_world_leg):
        com2world_ori = self.com2world_ori(com_ori)
        foot_pos_world_vec = np.reshape(np.append(foot_pos_world_leg, [1]),(4,1))
        foot_pos_robot = np.reshape(np.dot(com2world_ori, foot_pos_world_vec),(4))
        return foot_pos_robot[0:3]


    def robot_frame_to_world_robot(self, com_ori, foot_pos_robot):
        foot_pos_world = np.array([])
        for i in range(self.num_legs):
            foot_pos_world_leg = self.robot_frame_to_world_leg(com_ori, foot_pos_robot[3*i: 3*i+3])
            foot_pos_world = np.append(foot_pos_world, foot_pos_world_leg)
        return foot_pos_world


    def robot_frame_to_world_leg(self, com_ori, foot_pos_robot_leg): 
        world2com_ori = self.world2com_ori(com_ori)
        foot_pos_robot_vec = np.reshape(np.append(foot_pos_robot_leg, [1]),(4,1))
        foot_pos_world = np.reshape(np.dot(world2com_ori, foot_pos_robot_vec),(4))
        return foot_pos_world[0:3]


    def forward_kinematics_robot(self, joint_pos):
        '''
        joint_pos: 12-dim 
        '''
        foot_in_com_all = np.array([])
        for i in range(self.num_legs):
            foot_in_com = self.forward_kinematics_leg(joint_pos[3*i: 3*i+3], i)
            foot_in_com_all = np.append(foot_in_com_all, foot_in_com)
        return foot_in_com_all
    

    def forward_kinematics_leg(self, joint_pos_leg, leg_index):
        hip_rot = self.gen_rotation_matrix(joint_pos_leg[0], axis = 0)
        hip2thigh_translation = self.gen_translation_matrix([0, (-1)**(leg_index+1)*self.thigh_joint_loc, 0])
        thigh_joint_in_hip = np.dot(hip_rot,hip2thigh_translation)
        
        thigh_rot = self.gen_rotation_matrix(joint_pos_leg[1], axis = 1)
        thigh2shank_translation = self.gen_translation_matrix([0, 0, -self.thigh_length])
        shank_joint_in_hip = np.dot(thigh_joint_in_hip, np.dot(thigh_rot, thigh2shank_translation))
        
        shank_rot = self.gen_rotation_matrix(joint_pos_leg[2], axis = 1)
        foot_in_shank = np.array([[0], [0], [-self.shank_length], [1]])
        foot_in_hip = np.dot(shank_joint_in_hip, np.dot(shank_rot, foot_in_shank))
        
        com2hip_translation = self.gen_translation_matrix(self.hip_joint_location[leg_index])
        foot_in_com = np.reshape(np.dot(com2hip_translation, foot_in_hip),(4))[0:3]
        return foot_in_com

    
    def inverse_kinematics_robot(self, target_pos_robot):
        target_joint_pos_all = np.array([])
        for i in range(self.num_legs):
            target_pos_leg_robot = target_pos_robot[3*i: 3*i+3]
            target_joint_pos = self.inverse_kinematics_leg(target_pos_leg_robot, i)
            target_joint_pos_all = np.append(target_joint_pos_all, target_joint_pos)
        return target_joint_pos_all

    
    def inverse_kinematics_leg(self, target_pos_leg_robot, leg_index):
        target_pos_in_hip = target_pos_leg_robot - self.hip_joint_location[leg_index]
        x, y ,z = target_pos_in_hip[0], target_pos_in_hip[1], target_pos_in_hip[2]
        l_projected = math.sqrt(np.max([y**2 + z**2 - self.thigh_joint_loc**2, EPSILON]))
        
        if leg_index%2==1:
            hip_pos = (math.asin(np.clip((l_projected/(math.sqrt(y**2 + z**2)+EPSILON)), a_max=1.0, a_min=-1.0)) 
                + math.asin(np.clip((y/(math.sqrt(y**2 + z**2)+EPSILON)), a_max=1.0, a_min=-1.0))) -  math.pi/2.0
        else:
            hip_pos = math.pi/2.0 - (math.asin(np.clip((l_projected/(math.sqrt(y**2 + z**2)+EPSILON)), a_max=1.0, a_min=-1.0)) - math.asin(np.clip((y/(math.sqrt(y**2 + z**2)+EPSILON)), a_max=1.0, a_min=-1.0)))

        thigh2foot = math.sqrt(l_projected**2 + x**2)
        l = (self.thigh_length**2 + self.shank_length**2 - thigh2foot**2)/ (2*self.thigh_length*self.shank_length+ EPSILON)
        shank_pos = - (math.pi - math.acos(np.clip(l, a_max=1.0, a_min=-1.0)))

        l = (self.thigh_length**2 - self.shank_length**2 + thigh2foot**2)/ (2*self.thigh_length*thigh2foot+ EPSILON)
        thigh_pos = math.acos(np.clip(l, a_max=1.0, a_min=-1.0)) - math.asin(np.clip(x/thigh2foot+ EPSILON, a_max=1, a_min=-1))

        return np.array([hip_pos, thigh_pos, shank_pos])
