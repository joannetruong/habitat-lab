import gym, gym.spaces
import numpy as np
import magnum as mn
import habitat_sim 
from scipy.spatial.transform import Rotation as R
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff
from habitat.tasks.utils import cartesian_to_polar
from .utils import euler_from_quaternion, get_rpy, rotate_pos_from_hab, scalar_vector_to_quat, rotate_vector_2d
import squaternion

class A1():
    def __init__(self, sim=None, robot=None, robot_id=0, dt=1/60):
        #self.torque = config.get("torque", 1.0)
        self.robot = robot
        self.high_level_action_dim = 2
        self.sim = sim
        self.control = "position"
        self.ordered_joints = np.arange(12) # hip out, hip forward, knee
        self.linear_velocity = 0.35
        self.angular_velocity = 0.15
        self._initial_joint_positions = [-0.05, 0.60, -1.5,
                                         0.05, 0.60, -1.5,
                                         -0.05, 0.65, -1.5,
                                         0.05, 0.65, -1.5]
        self.robot_specific_reset()
        self.dt = dt
        # self.inverse_transform_quat = mn.Quaternion.from_matrix(inverse_transform.rotation())



    def set_up_continuous_action_space(self):
        self.high_level_action_space = gym.spaces.Box(shape=(self.high_level_action_dim,),
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

    def set_up_discrete_action_space(self):
        assert False, "A1 does not support discrete actions"

    
    def remap_joints(self, joints):
        joint_remapped = [0]*len(joints)
        joint_remapped[0] = joints[3]
        joint_remapped[1] = joints[4]
        joint_remapped[2] = joints[5]
        joint_remapped[3] = joints[0]
        joint_remapped[4] = joints[1]
        joint_remapped[5] = joints[2]
        joint_remapped[6] = joints[9]
        joint_remapped[7] = joints[10]
        joint_remapped[8] = joints[11]
        joint_remapped[9] = joints[6]
        joint_remapped[10] = joints[7]
        joint_remapped[11] = joints[8]
        return joint_remapped

    def quat_to_rad(self, rotation):
        heading_vector = quaternion_rotate_vector(
            rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return phi

    def rotate_vector_3d(self, v, r, p, y):
        """Rotates 3d vector by roll, pitch and yaw counterclockwise"""
        local_to_global = R.from_euler('xyz', [r, p, y]).as_dcm()
        global_to_local = local_to_global.T
        return np.dot(global_to_local, v)

    def convert_pose_from_robot(self, rigid_state):
        # pos as a mn.Vector3
        # ROT is list, W X Y Z
        # np.quaternion takes in as input W X Y Z
        rot_mn = mn.Matrix4.from_(rigid_state.rotation.to_matrix(), mn.Vector3(0,0,0))
        rs_m = rot_mn.__matmul__(
                    mn.Matrix4.rotation(
                    mn.Rad(np.pi / 2.0), # rotate 90 deg in roll
                    mn.Vector3((1.0, 0.0, 0.0)),
                    )
                    ).__matmul__(
                    mn.Matrix4.rotation(
                        mn.Rad(-np.pi), # rotate 180 deg in yaw
                        mn.Vector3((0.0, 1.0, 0.0)),
                    )
                    )
        trans_rs = mn.Quaternion.from_matrix(rs_m.rotation())
        trans_rs_wxyz = [trans_rs.scalar, *trans_rs.vector]

        heading = np.quaternion(*trans_rs_wxyz) # np.quaternion takes in as input W X Y Z
        heading = -self.quat_to_rad(heading)- np.pi / 2 # add 90 to yaw

        agent_rot_m = mn.Matrix4.rotation_y(
                mn.Rad(-heading),
            )
        agent_rot = mn.Quaternion.from_matrix(agent_rot_m.rotation())

        pos = np.array([*rigid_state.translation]) - np.array([0.0, 0.425, 0.0])
        curr_agent_pos = mn.Vector3(pos[0], pos[1], pos[2])

        return curr_agent_pos, agent_rot

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
        joint_positions = self.robot.joint_positions
        joint_velocities = self.robot.joint_velocities
        joint_positions_remapped = self.remap_joints(joint_positions)
        joint_velocities_remapped = self.remap_joints(joint_velocities)

        robot_state = self.robot.rigid_state 
        base_pos = robot_state.translation
        base_position = base_pos
        base_pos_tmp = rotate_pos_from_hab(base_pos)
        base_position.x = base_pos_tmp[0]
        base_position.y = base_pos_tmp[1]
        base_position.z = base_pos_tmp[2]

        _, robot_rot = self.convert_pose_from_robot(robot_state)

        tmp_quat = squaternion.Quaternion(robot_rot.scalar, *robot_rot.vector)
        roll, yaw, pitch = tmp_quat.to_euler()
        # base_orientation_euler = np.array([roll, pitch, yaw])
        base_orientation_euler = np.array([0, 0, 0])

        lin_vel = self.robot.root_linear_velocity
        ang_vel = self.robot.root_angular_velocity
        base_velocity = mn.Vector3(lin_vel.x, lin_vel.z, lin_vel.y)
        base_angular_velocity_euler = ang_vel

        return {
            'base_pos_x': base_position.x,
            'base_pos_y': base_position.y,
            'base_pos_z': base_position.z,
            'base_pos': np.array([base_position.x, base_position.y, base_position.z]),
            'base_ori_euler': base_orientation_euler,
            'base_ori_quat': robot_state.rotation,
            'base_velocity': base_velocity,
            'base_ang_vel': base_angular_velocity_euler,
            'j_pos': joint_positions_remapped,
            'j_vel': joint_velocities_remapped    
        }
        # 'base_velocity': rotate_vector_3d(base_velocity, *base_orientation_euler),
        # 'base_ang_vel': rotate_vector_3d(base_angular_velocity_euler, *base_orientation_euler),
            

    def set_mtr_pos(self, joint, ctrl):
        jms = self.robot.get_joint_motor_settings(joint)
        jms.position_target = ctrl
        self.robot.update_joint_motor(joint, jms)

    def set_joint_pos(self, joint_idx, angle):
        set_pos = np.array(self.robot.joint_positions) 
        set_pos[joint_idx] = angle
        self.robot.joint_positions = set_pos

    def apply_robot_action(self, action, pos_gain, vel_gain):
        """Applies actions to the robot.

        Args:
            a (list): List of floats. Length must be equal to len(self.ordered_joints).
        """
        assert (np.isfinite(action).all())
        assert len(action) == len(self.ordered_joints)
        for n, j in enumerate(self.ordered_joints):
            a = float(np.clip(action[n], -np.pi/2, np.pi/2)) 
            self.set_mtr_pos(n, a)

    def robot_specific_reset(self, joint_pos=None):
        if joint_pos is None:
            joint_pos = self._initial_joint_positions
            self.robot.joint_positions = joint_pos
        else:
            self.robot.joint_positions = joint_pos


    def step(self, action, pos_gain, vel_gain, dt=1/240.0, verbose=False, get_frames=True, follow_robot=False):
        
        self.apply_robot_action(action, pos_gain, vel_gain)
            # simulate dt seconds at 60Hz to the nearest fixed timestep
        if verbose:
            print("Simulating " + str(dt) + " world seconds.")
        depth_obs = []
        ortho_obs = []
        start_time = self.sim.get_world_time()
        count = 0
        
        if follow_robot:
            self._follow_robot()

        # while self.sim.get_world_time() < start_time + dt:
        self.sim.step_physics(dt)
        if get_frames:
            depth_obs.append(self.sim.get_sensor_observations(0))
            ortho_obs.append(self.sim.get_sensor_observations(1))
                
        return depth_obs, ortho_obs

    def _follow_robot(self):
        #robot_state = self.sim.get_articulated_object_root_state(self.robot_id)
        robot_state = self.robot.transformation
        node = self.sim._default_agent.scene_node
        self.h_offset = 0.69
        cam_pos = mn.Vector3(0, 0.0, 0)

        look_at = mn.Vector3(1, 0.0, 0)
        look_at = robot_state.transform_point(look_at)

        cam_pos = robot_state.transform_point(cam_pos)

        node.transformation = mn.Matrix4.look_at(
                cam_pos,
                look_at,
                mn.Vector3(0, 1, 0))

        self.cam_trans = node.transformation
        self.cam_look_at = look_at
        self.cam_pos = cam_pos

    def apply_action(self, action):
        self.apply_robot_action(action)

class AlienGo(A1):
    def __init__(self, sim=None, robot_id=0, dt=1/60):
        super().__init__(sim=sim, robot_id=robot_id, dt=dt)
        self._initial_joint_positions = [-0.1, 0.60, -1.5,
                                         0.1, 0.60, -1.5,
                                         -0.1, 0.6, -1.5,
                                         0.1, 0.6, -1.5]

class Laikago(A1):
    def __init__(self, sim=None, robot_id=0, dt=1/60):
        super().__init__(sim=sim, robot_id=robot_id, dt=dt)
        self._initial_joint_positions = [-0.1, 0.65, -1.2,
                                         0.1, 0.65, -1.2,
                                         -0.1, 0.65, -1.2,
                                         0.1, 0.65, -1.2]

class Spot(A1):
    def __init__(self, sim=None, robot=None, robot_id=0, dt=1/60):
        super().__init__(sim=sim,robot=robot, robot_id=robot_id, dt=dt)
        self._initial_joint_positions = [0.05, 0.7, -1.3,
                                         -0.05, 0.7, -1.3,
                                         0.05, 0.7, -1.3,
                                         -0.05, 0.7, -1.3]
