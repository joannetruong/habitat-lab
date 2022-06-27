import magnum as mn
import numpy as np
from habitat.utils.geometry_utils import euler_from_quaternion, wrap_heading
from habitat_sim.physics import JointMotorSettings


class A1:
    def __init__(self):
        self.name = "A1"
        self.robot_id = None
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
        # Spawn the URDF 0.35 meters above the navmesh upon reset
        self.robot_spawn_offset = np.array([0.0, 0.35, 0])
        self.robot_dist_to_goal = 0.24
        self.camera_spawn_offset = np.array([0.0, 0.18, -0.24])
        self.urdf_params = [12.46, 0.40, 0.62, 0.30]

        self.pos_gain = 0.6
        self.vel_gain = 1.0
        self.max_impulse = 1.0

        self.gibson_mapping = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

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

    def reset(self, pos=None, rot=0):
        """Resets robot's movement, moves it back to center of platform"""
        # Zero out the link and root velocities
        self.robot_id.clear_joint_states()

        self.robot_id.root_angular_velocity = mn.Vector3(0.0, 0.0, 0.0)
        self.robot_id.root_linear_velocity = mn.Vector3(0.0, 0.0, 0.0)

        self.set_pose_jms(self._initial_joint_positions, True)

        squat = np.normalized(np.quaternion(rot[3], *rot[:3]))

        agent_rot = mn.Matrix4.from_(
            mn.Quaternion(squat.imag, squat.real).to_matrix(),
            mn.Vector3(0.0, 0.0, 0.0),
        )  # 4x4 homogenous transform with no translation

        self.robot_id.transformation = mn.Matrix4.from_(
            (agent_rot @ self.rotation_offset).rotation(),  # 3x3 rotation
            mn.Vector3(*pos)
            + mn.Vector3(*self.robot_spawn_offset),  # translation vector
        )  # 4x4 homogenous transform

        self.start_height = self.robot_id.transformation.translation[1]

    def position(self):
        return self.robot_id.transformation.translation

    def position_xyz(self):
        pos = self.position()
        return np.array([pos.x, -pos.z, pos.y])

    def global_linear_velocity(self):
        """linear velocity in global frame"""
        return self.robot_id.root_linear_velocity

    def global_angular_velocity(self):
        """angular velocity in global frame"""
        return self.robot_id.root_angular_velocity

    def local_velocity(self, velocity):
        """returns local velocity and corrects for initial rotation of quadruped robots
        [forward, right, up]
        """
        local_vel = self.robot_id.transformation.inverted().transform_vector(velocity)
        return np.array([local_vel[0], local_vel[2], -local_vel[1]])

    def get_base_ori(self):
        return mn.Quaternion.from_matrix(
            self.robot_id.transformation.__matmul__(
                self.base_transform.inverted()
            ).rotation()
        )

    def get_ori_quat(self):
        """Given a numpy quaternion we'll return the roll pitch yaw
        :return: rpy: tuple of roll, pitch yaw
        """
        quat = self.get_base_ori()
        x, y, z = quat.vector
        w = quat.scalar
        return np.array([x, y, z, w])

    def get_ori_rpy(self):
        ori_quat = self.get_ori_quat()
        roll, pitch, yaw = euler_from_quaternion(*ori_quat)
        rpy = wrap_heading(np.array([roll, pitch, yaw]))
        return rpy

    def _new_jms(self, pos):
        """Returns a new jms with default settings at a given position
        :param pos: the new position to set to
        """
        return JointMotorSettings(
            pos,  # position_target
            self.pos_gain,  # position_gain
            0.0,  # velocity_target
            self.vel_gain,  # velocity_gain
            self.max_impulse,  # max_impulse
        )

    def set_pose_jms(self, pose, kinematic_snap=True):
        """Sets a robot's pose and changes the jms to that pose (rests at
        given position)
        """
        # Snap joints kinematically
        if kinematic_snap:
            self.robot_id.joint_positions = pose

        # Make motor controllers maintain this position
        for idx, p in enumerate(pose):
            self.robot_id.update_joint_motor(idx, self._new_jms(p))

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
        # joint_positions = self.robot.joint_positions
        # joint_velocities = self.robot.joint_velocities
        joint_positions = np.array(self.robot_id.joint_positions)[self.gibson_mapping]
        joint_velocities = np.array(self.robot_id.joint_velocities)[self.gibson_mapping]
        base_pos = self.position_xyz()
        base_ori_euler = self.get_ori_rpy()
        base_ori_quat = self.get_ori_quat()

        lin_vel = self.local_velocity(self.global_linear_velocity())
        ang_vel = self.local_velocity(self.global_angular_velocity())

        return {
            "base_pos": base_pos,
            "base_ori_euler": base_ori_euler,
            "base_ori_quat": base_ori_quat,
            "base_velocity": lin_vel,
            "base_ang_vel": ang_vel,
            "j_pos": joint_positions,
            "j_vel": joint_velocities,
        }


class AlienGo(A1):
    def __init__(self):
        super().__init__()
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
    def __init__(self):
        super().__init__()
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


class Locobot(A1):
    def __init__(self):
        super().__init__()
        self.name = "Locobot"
        self._initial_joint_positions = []

        # Spawn the URDF 0.425 meters above the navmesh upon reset
        self.robot_spawn_offset = np.array([0.0, 0.25, 0])
        self.robot_dist_to_goal = 0.2
        self.camera_spawn_offset = np.array([0.0, 0.31, -0.55])
        self.urdf_params = np.array([4.19, 0.00, 0.35, 0.35])


class Spot(A1):
    def __init__(self):
        super().__init__()
        self.name = "Spot"
        self._initial_joint_positions = [
            0.05,
            0.7,
            -1.3,
            -0.05,
            0.7,
            -1.3,
            0.05,
            0.7,
            -1.3,
            -0.05,
            0.7,
            -1.3,
        ]
        ## compact form
        # self._initial_joint_positions = [
        #     0.0,
        #     1.57,
        #     3.14,
        #     0.0,
        #     1.57,
        #     3.14,
        #     0.0,
        #     -1.57,
        #     3.14,
        #     0.0,
        #     -1.57,
        #     3.14,
        # ]

        self.robot_spawn_offset = np.array([0.0, 0.625, 0])
        self.robot_dist_to_goal = 0.325
        self.camera_spawn_offset = np.array([0.0, 0.325, -0.325])
        self.urdf_params = np.array([32.70, 0.88, 1.10, 0.50])
        self.pos_gain = 0.4
        self.vel_gain = 1.8
        self.max_impulse = 1.0
