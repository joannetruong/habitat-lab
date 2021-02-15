from gibson2.core.physics.robot_locomotors import LocomotorRobot
from gibson2.utils.utils import rotate_vector_3d
import gym, gym.spaces
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc

class A1(LocomotorRobot):
    def __init__(self, config, urdf_file="a1/a1.urdf"):
        self.config = config
        self.torque = config.get("torque", 1.0)
        self.high_level_action_dim = 2
        LocomotorRobot.__init__(
            self,
            urdf_file,
            action_dim=12,
            torque_coef=1.0,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="position",
        )
        self._initial_joint_positions = [-0.05, 0.60, -1.5,
                                         0.05, 0.60, -1.5,
                                         -0.05, 0.65, -1.5,
                                         0.05, 0.65, -1.5]
        self.p = bc.BulletClient(connection_mode=p.DIRECT)

    def set_up_continuous_action_space(self):
        self.high_level_action_space = gym.spaces.Box(shape=(self.high_level_action_dim,),
                                           low=-0.35,
                                           high=0.35,
                                           dtype=np.float32)
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high
        self.high_level_action_high = self.torque * np.ones([self.high_level_action_dim])
        self.high_level_action_low = -self.high_level_action_high

    def set_up_discrete_action_space(self):
        assert False, "A1 does not support discrete actions"

    def calc_state(self):
        """Computes the state.
        Unlike the original gym environment, which returns only a single
        array, we return here a dict because this is much more intuitive later on.
        Returns:
        dict: The dict contains four different states. 'j_pos' are the
                joint positions. 'j_vel' are the current velocities of the
                joint angles. 'base_pos' is the 3D position of the base of Daisy
                in the world. 'base_ori_euler' is the orientation of the robot
                in euler angles.
        """
        # print('joints: ', [j.joint_index for j in self.ordered_joints], [j.body_index for j in self.ordered_joints])
        # print('feet: ', [f.body_part_index for f in self.ordered_foot])
        joint_positions = [j.get_state()[0] for j in self.ordered_joints]
        joint_velocities = [j.get_state()[1] for j in self.ordered_joints]
        joint_effort = [j.get_state()[2] for j in self.ordered_joints]
        foot_pos = [f.get_pose() for f in self.ordered_foot]
        base_position = self.get_position()
        base_velocity = self.get_linear_velocity()
        base_angular_velocity = self.get_angular_velocity()
        # print(base_position)
        self.body_xyz = base_position
        base_orientation_quat = self.get_orientation()
        base_orientation_euler = self.get_rpy()

        return {
            'base_pos_x': base_position[0:1],
            'base_pos_y': base_position[1:2],
            'base_pos_z': base_position[2:],
            'base_pos': base_position,
            'base_ori_euler': base_orientation_euler,
            'base_ori_quat': base_orientation_quat,
            'base_velocity': rotate_vector_3d(base_velocity, *base_orientation_euler),
            'base_ang_vel': rotate_vector_3d(base_angular_velocity, *base_orientation_euler),
            'j_pos': joint_positions,
            'j_vel': joint_velocities,
            'j_eff': joint_effort,
            'foot_pos': foot_pos,
        }

    def apply_robot_action(self, action):
        """Applies actions to the robot.

        Args:
            a (list): List of floats. Length must be equal to len(self.ordered_joints).
        """
        assert (np.isfinite(action).all())
        assert len(action) == len(self.ordered_joints)
        for n, j in enumerate(self.ordered_joints):
            a = action[n]
            if self.control == 'velocity':
                j.set_motor_velocity(self.velocity_coef * j.max_velocity * float(np.clip(a, -1, +1)))

            elif self.control == 'position':
                j.set_motor_position(float(np.clip(a, -np.pi, np.pi)))

            elif self.control == 'effort':
                j.set_motor_torque(self.torque_coef * j.max_torque * float(np.clip(a, -1, +1)))
            else:
                print('not implemented yet')

    def robot_specific_reset(self, joint_pos=None):
        if joint_pos is None:
            joint_pos = self._initial_joint_positions
        for n, j in enumerate(self.ordered_joints):
            a = joint_pos[n]
            j.reset_joint_state(position=a, velocity=0.0)

    def step(self, action):
        self.apply_robot_action(action=action)
        p.stepSimulation()

    def apply_action(self, action):
        self.apply_robot_action(action)


class AlienGo(A1):
    def __init__(self, config, urdf_file="aliengo/aliengo.urdf"):
        super().__init__(config=config, urdf_file=urdf_file)
        self._initial_joint_positions = [-0.1, 0.60, -1.5,
                                         0.1, 0.60, -1.5,
                                         -0.1, 0.6, -1.5,
                                         0.1, 0.6, -1.5]

class Laikago(A1):
    # def __init__(self, config, urdf_file="laikago_pb/laikago.urdf"):
    def __init__(self, config, urdf_file="laikago/laikago.urdf"):
        super().__init__(config=config, urdf_file=urdf_file)
        self._initial_joint_positions = [-0.1, 0.65, -1.2,
                                         0.1, 0.65, -1.2,
                                         -0.1, 0.65, -1.2,
                                         0.1, 0.65, -1.2]

class Spot(A1):
    def __init__(self, config, urdf_file="spot/spot.urdf"):
        super().__init__(config=config, urdf_file=urdf_file)
        self._initial_joint_positions = [-.124, 0.876, -1.5,
                                         0.124, 0.876, -1.5,
                                         -0.124, 0.876, -1.5,
                                         0.124, 0.876, -1.5]

class SpotBD(A1):
    def __init__(self, config, urdf_file="spot_bd/spot_bd.urdf"):
        super().__init__(config=config, urdf_file=urdf_file)
        self._initial_joint_positions = [-.124, 0.876, -1.5,
                                         0.124, 0.876, -1.5,
                                         -0.124, 0.876, -1.5,
                                         0.124, 0.876, -1.5]

class A1Turn(A1):
    def __init__(self, config, urdf_file="a1/a1.urdf"):
        self.linear_velocity = config.get('linear_velocity', 0.35)
        self.angular_velocity = config.get('angular_velocity', 0.15)
        A1.__init__(self,
                    config, 
                    urdf_file=urdf_file)

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

class AlienGoTurn(AlienGo):
    def __init__(self, config, urdf_file="aliengo/aliengo.urdf"):
        self.linear_velocity = config.get('linear_velocity', 0.35)
        self.angular_velocity = config.get('angular_velocity', 0.15)
        AlienGo.__init__(self,
                     config, 
                     urdf_file=urdf_file)
        
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

class LaikagoTurn(Laikago):
    # def __init__(self, config, urdf_file="laikago_pb/laikago.urdf"):
    def __init__(self, config, urdf_file="laikago/laikago.urdf"):
        self.linear_velocity = config.get('linear_velocity', 0.35)
        self.angular_velocity = config.get('angular_velocity', 0.15)
        Laikago.__init__(self,
                     config, 
                     urdf_file=urdf_file)
        
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

class SpotTurn(Spot):
    def __init__(self, config, urdf_file="spot/spot.urdf"):
        self.linear_velocity = config.get('linear_velocity', 0.35)
        self.angular_velocity = config.get('angular_velocity', 0.15)
        Spot.__init__(self,
                     config, 
                     urdf_file=urdf_file)
        
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

class SpotBDTurn(Spot):
    def __init__(self, config, urdf_file="spot_bd/spot_bd.urdf"):
        self.linear_velocity = config.get('linear_velocity', 0.35)
        self.angular_velocity = config.get('angular_velocity', 0.15)
        Spot.__init__(self,
                     config, 
                     urdf_file=urdf_file)
        
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

