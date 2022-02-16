import gym, gym.spaces
import numpy as np
import magnum as mn
import habitat_sim
from scipy.spatial.transform import Rotation as R
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
    quaternion_from_coeff,
)
from habitat.tasks.utils import cartesian_to_polar
from .utils import (
    euler_from_quaternion,
    get_rpy,
    rotate_pos_from_hab,
    scalar_vector_to_quat,
    rotate_vector_2d,
)
import squaternion
import torch


class A1:
    def __init__(self, sim=None, robot=None, rand_id=None, reset=True):
        # self.torque = config.get("torque", 1.0)
        self.name = "A1"
        self.id = rand_id
        self.robot = robot
        self.high_level_action_dim = 2
        self.sim = sim
        self.control = "position"
        self.ordered_joints = np.arange(12)  # hip out, hip forward, knee
        self.linear_velocity = 0.35
        self.angular_velocity = 0.15
        # Gibson mapping: FR, FL, RR, RL
        self._initial_joint_positions = [
            0.05,
            0.60,
            -1.5,  # FL
            -0.05,
            0.60,
            -1.5,  # FR
            0.05,
            0.65,
            -1.5,  # RL
            -0.05,
            0.65,
            -1.5,
        ]  # RR
        self.feet_link_ids = [5, 9, 13, 17]
        # Spawn the URDF 0.18 meters above the navmesh upon reset
        self.robot_spawn_offset = np.array([0.0, 0.35, 0])
        self.robot_dist_to_goal = 0.24
        self.camera_spawn_offset = np.array([0.0, 0.18, -0.24])
        self.urdf_params = [12.46, 0.40, 0.62, 0.30]
        # The robots need to rolled 90 deg then yaw'd 180 deg relative to agent
        self.rotation_offset = (
            mn.Matrix4.rotation_y(
                mn.Rad(-np.pi / 2),  # Rotate -90 deg yaw (agent offset)
            )
            .__matmul__(
                mn.Matrix4.rotation_y(
                    mn.Rad(np.pi),  # Rotate 180 deg yaw
                )
            )
            .__matmul__(
                mn.Matrix4.rotation_x(
                    mn.Rad(-np.pi / 2.0),  # Rotate 90 deg roll
                )
            )
        )

        if reset:
            self.robot_specific_reset()
            # self.inverse_transform_quat = mn.Quaternion.from_matrix(inverse_transform.rotation())

    def set_up_continuous_action_space(self):
        self.high_level_action_space = gym.spaces.Box(
            shape=(self.high_level_action_dim,),
            low=-self.linear_velocity,
            high=self.linear_velocity,
            dtype=np.float32,
        )
        self.high_level_ang_action_space = gym.spaces.Box(
            shape=(1,),
            low=-self.angular_velocity,
            high=self.angular_velocity,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            shape=(self.action_dim,), low=-1.0, high=1.0, dtype=np.float32
        )
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high
        self.high_level_lin_action_high = self.torque * np.ones(
            [self.high_level_action_dim]
        )
        self.high_level_lin_action_low = -self.high_level_lin_action_high
        self.high_level_ang_action_high = self.torque * np.ones([1])
        self.high_level_ang_action_low = -self.high_level_ang_action_high

    def set_up_discrete_action_space(self):
        assert False, "A1 does not support discrete actions"

    def remap_gib_hab(self, habitat_mapping):
        gibson_mapping = [0] * len(habitat_mapping)
        # remap from Gibson mapping: FR, FL, RR, RL to Habitat mapping: FL, FR, RL, RR
        gibson_mapping[0] = habitat_mapping[3]
        gibson_mapping[1] = habitat_mapping[4]
        gibson_mapping[2] = habitat_mapping[5]
        gibson_mapping[3] = habitat_mapping[0]
        gibson_mapping[4] = habitat_mapping[1]
        gibson_mapping[5] = habitat_mapping[2]
        gibson_mapping[6] = habitat_mapping[9]
        gibson_mapping[7] = habitat_mapping[10]
        gibson_mapping[8] = habitat_mapping[11]
        gibson_mapping[9] = habitat_mapping[6]
        gibson_mapping[10] = habitat_mapping[7]
        gibson_mapping[11] = habitat_mapping[8]
        return gibson_mapping

    def quat_to_rad(self, rotation):
        heading_vector = quaternion_rotate_vector(
            rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return phi

    def rotate_vector_3d(self, v, r, p, y):
        """Rotates 3d vector by roll, pitch and yaw counterclockwise"""
        local_to_global = R.from_euler("xyz", [r, p, y]).as_dcm()
        global_to_local = local_to_global.T
        return np.dot(global_to_local, v)

    def convert_pose_from_robot(self, rigid_state):
        # pos as a mn.Vector3
        # ROT is list, W X Y Z
        # np.quaternion takes in as input W X Y Z
        rot_mn = mn.Matrix4.from_(
            rigid_state.rotation.to_matrix(), mn.Vector3(0, 0, 0)
        )
        rs_m = rot_mn.__matmul__(
            mn.Matrix4.rotation(
                mn.Rad(-np.pi / 2.0),  # rotate 90 deg in roll
                mn.Vector3((1.0, 0.0, 0.0)),
            )
        ).__matmul__(
            mn.Matrix4.rotation(
                mn.Rad(-np.pi * 0.01),  # rotate 180 deg in yaw
                mn.Vector3((0.0, 1.0, 0.0)),
            )
        )
        trans_rs = mn.Quaternion.from_matrix(rs_m.rotation())
        trans_rs_wxyz = [trans_rs.scalar, *trans_rs.vector]

        heading = np.quaternion(
            *trans_rs_wxyz
        )  # np.quaternion takes in as input W X Y Z
        heading = -self.quat_to_rad(heading) - np.pi / 2  # add 90 to yaw

        agent_rot_m = mn.Matrix4.rotation_y(
            mn.Rad(-heading),
        )
        agent_rot = mn.Quaternion.from_matrix(agent_rot_m.rotation())

        pos = np.array([*rigid_state.translation]) - np.array(
            [0.0, 0.425, 0.0]
        )
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
        joint_positions_remapped = self.remap_gib_hab(joint_positions)
        joint_velocities_remapped = self.remap_gib_hab(joint_velocities)

        robot_state = self.robot.rigid_state
        base_pos = robot_state.translation
        base_position = base_pos
        base_pos_tmp = rotate_pos_from_hab(base_pos)
        base_position.x = base_pos_tmp[0]
        base_position.y = base_pos_tmp[1]
        base_position.z = base_pos_tmp[2]

        # _, robot_rot = self.convert_pose_from_robot(robot_state)

        base_trans = (
            mn.Matrix4.rotation_y(
                mn.Rad(-np.pi / 2),
            )
            .__matmul__(  # Rotate 180 deg yaw
                mn.Matrix4.rotation(
                    mn.Rad(np.pi),
                    mn.Vector3((0.0, 1.0, 0.0)),
                )
            )
            .__matmul__(  # Rotate 90 deg roll
                mn.Matrix4.rotation(
                    mn.Rad(-np.pi / 2.0),
                    mn.Vector3((1.0, 0.0, 0.0)),
                )
            )
        )

        final_rotation = mn.Quaternion.from_matrix(
            self.robot.transformation.__matmul__(
                base_trans.inverted()
            ).rotation()
        )
        robot_rot = final_rotation

        # robot_rot = robot_state.rotation.__mul__(
        #     mn.Quaternion.rotation(mn.Rad(np.pi / 2), mn.Vector3(1.0, 0.0, 0.0)))
        tmp_quat = squaternion.Quaternion(robot_rot.scalar, *robot_rot.vector)
        roll, yaw, pitch = tmp_quat.to_euler()
        # roll, pitch, yaw = tmp_quat.to_euler()
        base_orientation_euler = np.array([roll, pitch, yaw])
        # base_orientation_euler = np.array([0, 0, 0])

        lin_vel = self.robot.root_linear_velocity
        ang_vel = self.robot.root_angular_velocity
        base_velocity = mn.Vector3(lin_vel.x, lin_vel.z, lin_vel.y)
        base_angular_velocity_euler = ang_vel

        return {
            "base_pos_x": base_position.x,
            "base_pos_y": base_position.y,
            "base_pos_z": base_position.z,
            "base_pos": np.array(
                [base_position.x, base_position.y, base_position.z]
            ),
            "base_ori_euler": base_orientation_euler,
            "base_ori_quat": robot_state.rotation,
            "base_velocity": base_velocity,
            "base_ang_vel": base_angular_velocity_euler,
            "j_pos": joint_positions_remapped,
            "j_vel": joint_velocities_remapped,
        }

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
        actions_remapped = self.remap_gib_hab(action)
        assert np.isfinite(action).all()
        assert len(action) == len(self.ordered_joints)
        for n, j in enumerate(self.ordered_joints):
            a = float(np.clip(actions_remapped[n], -np.pi / 2, np.pi / 2))
            self.set_mtr_pos(n, a)

    def robot_specific_reset(self, joint_pos=None):
        if joint_pos is None:
            joint_pos = self._initial_joint_positions
            self.robot.joint_positions = joint_pos
        else:
            self.robot.joint_positions = joint_pos

    def step(
        self,
        action,
        pos_gain,
        vel_gain,
        dt=1 / 240.0,
        verbose=False,
        get_frames=True,
        follow_robot=False,
    ):

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
        # robot_state = self.sim.get_articulated_object_root_state(self.robot_id)
        robot_state = self.robot.transformation
        node = self.sim._default_agent.scene_node
        self.h_offset = 0.69
        cam_pos = mn.Vector3(0, 0.0, 0)

        look_at = mn.Vector3(1, 0.0, 0)
        look_at = robot_state.transform_point(look_at)

        cam_pos = robot_state.transform_point(cam_pos)

        node.transformation = mn.Matrix4.look_at(
            cam_pos, look_at, mn.Vector3(0, 1, 0)
        )

        self.cam_trans = node.transformation
        self.cam_look_at = look_at
        self.cam_pos = cam_pos

    def apply_action(self, action):
        self.apply_robot_action(action)


class AlienGo(A1):
    def __init__(self, sim=None, robot=None, rand_id=None):
        super().__init__(sim=sim, robot=robot, rand_id=None)
        self.name = "AlienGo"
        self._initial_joint_positions = [
            0.1,
            0.60,
            -1.5,
            -0.1,
            0.60,
            -1.5,
            0.1,
            0.6,
            -1.5,
            -0.1,
            0.6,
            -1.5,
        ]

        self.feet_link_ids = [4, 8, 12, 16]  ### FL, FR, RL, RR
        # self.feet_link_ids = [12]
        self.robot_spawn_offset = np.array([0.0, 0.475, 0])
        self.robot_dist_to_goal = 0.3235
        self.camera_spawn_offset = np.array([0.0, 0.25, -0.3235])
        self.urdf_params = np.array([20.64, 0.50, 0.89, 0.34])


class Laikago(A1):
    def __init__(self, sim=None, robot=None, rand_id=None):
        super().__init__(sim=sim, robot=robot, rand_id=None)
        self.name = "Laikago"
        self._initial_joint_positions = [
            0.1,
            0.65,
            -1.2,
            -0.1,
            0.65,
            -1.2,
            0.1,
            0.65,
            -1.2,
            -0.1,
            0.65,
            -1.2,
        ]
        self.robot_spawn_offset = np.array([0.0, 0.475, 0])
        self.robot_dist_to_goal = 0.3235
        self.camera_spawn_offset = np.array([0.0, 0.25, -0.3235])


class Spot(A1):
    def __init__(self, sim=None, robot=None, rand_id=None):
        super().__init__(sim=sim, robot=robot, rand_id=None)
        self.name = "Spot"
        self._initial_joint_positions = np.deg2rad([0, 60, -120] * 4)
        # self._initial_joint_positions = [0.05, 0.7, -1.3,
        #                                  -0.05, 0.7, -1.3,
        #                                  0.05, 0.7, -1.3,
        #                                  -0.05, 0.7, -1.3]

        # Spawn the URDF 0.425 meters above the navmesh upon reset
        ## if evaluating coda episodes, manually increase offset by an extra 0.1m
        # self.robot_spawn_offset = np.array([0.0, 0.60, 0])
        self.robot_spawn_offset = np.array([0.0, 0.625, 0])
        self.robot_dist_to_goal = 0.425
        self.camera_spawn_offset = np.array([0.0, 0.325, -0.325])
        self.urdf_params = np.array([32.70, 0.88, 1.10, 0.50])


class Locobot(A1):
    def __init__(self, sim=None, robot=None, rand_id=None):
        super().__init__(sim=sim, robot=robot, rand_id=None, reset=False)
        self.name = "Locobot"
        self._initial_joint_positions = []

        # Spawn the URDF 0.425 meters above the navmesh upon reset
        self.robot_spawn_offset = np.array([0.0, 0.25, 0])
        self.robot_dist_to_goal = 0.2
        self.camera_spawn_offset = np.array([0.0, 0.31, -0.55])
        self.urdf_params = np.array([4.19, 0.00, 0.35, 0.35])
