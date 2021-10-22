from .utils import rotate_vector_3d, euler_from_quaternion, get_rpy, quat_to_rad
import gym, gym.spaces
import numpy as np
import magnum as mn
import torch

class Daisy():
    def __init__(self, sim=None, robot=None, dt=1/60):
        # config["urdf_path"] = "daisy/daisy_advanced_side.urdf"
        self.linear_velocity = 0.35
        self.angular_velocity = 0.15
        self.torque = 1.0
        self.torque_coef = 1.0
        self.high_level_action_dim = 2
        self.control = "position"
        self.id = 0
        self._action_mapping = {
            'L_F_motor_1/X8_9': 0,
            'L_F_motor_2/X8_16': 1,
            'L_F_motor_3/X8_9': 2,
            'L_M_motor_1/X8_9': 6,
            'L_M_motor_2/X8_16': 7,
            'L_M_motor_3/X8_9': 8,
            'L_B_motor_1/X8_9': 12,
            'L_B_motor_2/X8_16': 13,
            'L_B_motor_3/X8_9': 14,
            'R_F_motor_1/X8_9': 3,
            'R_F_motor_2/X8_16': 4,
            'R_F_motor_3/X8_9': 5,
            'R_M_motor_1/X8_9': 9,
            'R_M_motor_2/X8_16': 10,
            'R_M_motor_3/X8_9': 11,
            'R_B_motor_1/X8_9': 15,
            'R_B_motor_2/X8_16': 16,
            'R_B_motor_3/X8_9': 17,
        }
        self._initial_joint_positions = [0.0, 1.2, -0.5,
                                         0.0, -1.2, 0.5,
                                         0.0, 1.2, -0.5,
                                         0.0, -1.2, 0.5,
                                         0.0, 1.2, -0.5,
                                         0.0, -1.2, 0.5]
        self.rotation_offset = mn.Matrix4.rotation_y(
            mn.Rad(-np.pi / 2),  # Rotate -90 deg yaw (agent offset)
        ).__matmul__(
            mn.Matrix4.rotation_y(
                mn.Rad(np.pi),  # Rotate 180 deg yaw
            )
        ).__matmul__(
            mn.Matrix4.rotation_x(
                mn.Rad(-np.pi / 2.0),  # Rotate 90 deg roll
            )
        )

        # Spawn the URDF 0.425 meters above the navmesh upon reset
        self.spawn_offset = np.array([0.0, 0.30, -0.09])
        self.z_in = torch.rand(1).requires_grad_(True)

    def joint_mapping(self, joint):
        index = [0, 1, 2, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17]
        joint_hardware = []
        for i in range(len(index)):
            joint_hardware.append(joint[index[i]])
        return joint_hardware

    def foot_mapping(self, foot_pos):
        index = [0, 3, 1, 4, 2, 5]
        foot_hardware = []
        for i in range(6):
            foot_hardware.append(foot_pos[index[i]])
        return foot_hardware

    def set_up_continuous_action_space(self):
        self.high_level_lin_action_space = gym.spaces.Box(shape=(self.high_level_action_dim,),
                                           low=-self.linear_velocity,
                                           high=self.linear_velocity,
                                           dtype=np.float32)
        self.high_level_ang_action_space = gym.spaces.Box(shape=(1,),
                                           low=-self.angular_velocity,
                                           high=self.angular_velocity,
                                           dtype=np.float32)
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high
        self.high_level_lin_action_high = self.torque * np.ones([self.high_level_action_dim])
        self.high_level_lin_action_low = -self.high_level_lin_action_high
        self.high_level_ang_action_high = self.torque * np.ones([1])
        self.high_level_ang_action_low = -self.high_level_ang_action_high

    def calc_state(self):
        """Computes the state.

        Unlike the original gym environment, which returns only a single
        array, we return here a dict because this is much more intuitive later on.

        Returns:
        DaisyRobot    dict: The dict contains four different states. 'j_pos' are the
                joint positions. 'j_vel' are the current velocities of the
                joint angles. 'base_pos' is the 3D position of the base of Daisy
                in the world. 'base_ori_euler' is the orientation of the robot
                in euler angles.
        """
        joint_positions = self.sim.get_articulated_object_positions(self.robot_id)
        joint_velocities = self.sim.get_articulated_object_velocities(self.robot_id)

        joint_effort = [j.get_state()[2] for j in self.ordered_joints]
        foot_pose = [f.get_pose() for f in self.ordered_foot]

        robot_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)
        vels = self.sim.get_articulated_link_velocity(self.robot_id, 0)
        lin_vel, ang_vel = vels[0], vels[1]
        base_pos = robot_state.translation
        # base_position[2] = -base_position[2]
        base_orientation_quat = robot_state.rotation
        base_position = base_pos
        base_pos_tmp = self.rotate_hab_pos(base_pos)
        base_position.x = base_pos_tmp[0]
        base_position.y = base_pos_tmp[1]
        base_position.z = base_pos_tmp[2]

        base_orientation_euler = get_rpy(base_orientation_quat)
        base_orientation_euler_origin = get_rpy(base_orientation_quat, transform=False)

        if finite_diff:
            if prev_state is None:
                base_velocity = mn.Vector3() #self.sim.get_articulated_link_angular_velocity(self.robot_id, 0)
                frame_pos = np.zeros((3))
                base_angular_velocity_euler = mn.Vector3() # self.sim.get_articulated_link_angular_velocity(self.robot_id, 0)
            else:
                # print(prev_state['base_pos'], base_position)
                base_velocity = (base_position - prev_state['base_pos']) / self.dt
                if base_velocity == mn.Vector3():
                    base_velocity = mn.Vector3(prev_state['base_velocity'])
                base_angular_velocity_euler = (base_orientation_euler - prev_state['base_ori_euler']) / self.dt
        else:
            base_velocity = lin_vel
            base_angular_velocity_euler = ang_vel

        return {
            'base_pos_x': base_position.x,
            'base_pos_y': base_position.y,
            'base_pos_z': base_position.z,
            'base_pos': np.array([base_position.x, base_position.y, base_position.z]) ,
            'base_ori_euler': base_orientation_euler,
            'base_ori_quat': base_orientation_quat,
            'base_velocity': rotate_vector_3d(base_velocity, *base_orientation_euler),
            # 'base_velocity': list(lin_vel),
            'base_ang_vel': rotate_vector_3d(base_angular_velocity_euler, *base_orientation_euler),
            # 'base_ang_vel': list(ang_vel),
            'j_pos': joint_positions,
            'j_vel': joint_velocities,
            'j_eff': self.joint_mapping(joint_effort),
            'foot_pose': self.foot_mapping(foot_pose)
        }

    def apply_robot_action(self, action):
        """Applies actions to the robot.

        Args:
            a (list): List of floats. Length must be equal to len(self.ordered_joints).
        """
        assert (np.isfinite(action).all())
        assert len(action) == len(self.ordered_joints)
        for n, j in enumerate(self.ordered_joints):
            a = action[self._action_mapping[j.joint_name]]
            if self.control == 'velocity':
                j.set_motor_velocity(self.velocity_coef * j.max_velocity * float(np.clip(a, -1, +1)))

            elif self.control == 'position':
                j.set_motor_position(float(np.clip(a, -np.pi, np.pi)))

            elif self.control == 'effort':
                j.set_motor_torque(self.torque_coef * j.max_torque * float(np.clip(a, -1, +1)))
            else:
                print('not implemented yet')

    def robot_specific_reset(self, joint_pos=None):
        self.reset(joint_pos)

    def reset(self, joint_pos=None):
        if joint_pos is None:
            joint_pos = self._initial_joint_positions
        for j in self.ordered_joints:
            a = joint_pos[self._action_mapping[j.joint_name]]
            j.reset_joint_state(position=a, velocity=0.0)

    def step(self, action):
        self.apply_robot_action(action=action)
        p.stepSimulation()

    def apply_action(self, action):
        self.apply_robot_action(action)

class Daisy_4legged(Daisy):
    def __init__(self, sim=None, robot=None, dt=1/60):
        self._action_mapping = {
            'L_F_motor_1/X8_9': 0,
            'L_F_motor_2/X8_16': 1,
            'L_F_motor_3/X8_9': 2,
            'L_B_motor_1/X8_9': 6,
            'L_B_motor_2/X8_16': 7,
            'L_B_motor_3/X8_9': 8,
            'R_F_motor_1/X8_9': 3,
            'R_F_motor_2/X8_16': 4,
            'R_F_motor_3/X8_9': 5,
            'R_B_motor_1/X8_9': 9,
            'R_B_motor_2/X8_16': 10,
            'R_B_motor_3/X8_9': 11,
        }
        self._initial_joint_positions = [0.0, 1.2, -0.5,
                                         0.0, -1.2, 0.5,
                                         0.0, 1.2, -0.5,
                                         0.0, -1.2, 0.5]
        super().__init__(sim=sim, robot=robot, dt=dt)

    def joint_mapping(self, joint):
        index = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]
        joint_hardware = []
        for i in range(len(index)):
            joint_hardware.append(joint[index[i]])
        return joint_hardware

    def foot_mapping(self, foot_pos):
        index = [0, 1, 2, 3]
        foot_hardware = []
        for i in range(len(index)):
            foot_hardware.append(foot_pos[index[i]])
        return foot_hardware
