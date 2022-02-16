import magnum as mn
import numpy as np
import squaternion
from habitat.utils.geometry_utils import wrap_heading

from .utils import rotate_pos_from_hab


class A1:
    def __init__(self, robot_id):
        self.name = "A1"
        self.robot_id = robot_id
        self.ordered_joints = [
            "FR_hip",
            "FR_thigh",
            "FR_calf",
            "FL_hip",
            "FL_thigh",
            "FL_calf",
            "RR_hip",
            "RR_thigh",
            "RR_calf",
            "RL_hip",
            "RL_thigh",
            "RL_calf",
        ]
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
        # self.robot_spawn_offset = np.array([0.0, 0.35, 0])
        self.robot_dist_to_goal = 0.24
        self.camera_spawn_offset = np.array([0.0, 0.18, -0.24])
        self.urdf_params = [12.46, 0.40, 0.62, 0.30]

        self.base_transform = mn.Matrix4.rotation(
            mn.Rad(np.deg2rad(-90)), mn.Vector3(1.0, 0.0, 0.0)
        ) @ mn.Matrix4.rotation(
            mn.Rad(np.deg2rad(yaw)), mn.Vector3(0.0, 0.0, 1.0)
        )
        self.base_transform.translation = mn.Vector3(0.0, 0.35, 0.0)

    def reset(self, yaw=180):
        """Resets robot's movement, moves it back to center of platform"""
        # Zero out the link and root velocities
        self.robot_id.clear_joint_states()
        self.robot_id.root_angular_velocity = mn.Vector3(0.0, 0.0, 0.0)
        self.robot_id.root_linear_velocity = mn.Vector3(0.0, 0.0, 0.0)

        # Roll robot 90 deg
        self.robot_id.transformation = self.base_transform
        self.robot_id.joint_positions = self._initial_joint_positions

    def position(self):
        self.robot_id.transformation.translation

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
        local_vel = self.robot_id.transformation.inverted().transform_vector(
            velocity
        )
        return np.array([local_vel[0], local_vel[2], -local_vel[1]])

    def get_ori_quat(self):
        """Given a numpy quaternion we'll return the roll pitch yaw
        :return: rpy: tuple of roll, pitch yaw
        """
        quat = self.robot_id.rotation.normalized()
        undo_rot = mn.Quaternion(
            ((np.sin(np.deg2rad(45)), 0.0, 0.0), np.cos(np.deg2rad(45)))
        ).normalized()
        quat = quat * undo_rot

        x, y, z = quat.vector
        w = quat.scalar
        return np.array([x, y, z, w])

    def get_ori_rpy(self):
        ori_quat = self.get_ori_quat()
        roll, pitch, yaw = self._euler_from_quaternion(*ori_quat)
        rpy = wrap_heading(np.array([roll, pitch, yaw]))
        return rpy

    def _euler_from_quaternion(self, x, y, z, w):
        """Convert a quaternion into euler angles (roll, yaw, pitch)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = 1.0 if t2 > 1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, -yaw_z, pitch_y  # in radians

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

        base_pos = rotate_pos_from_hab(self.position())
        base_ori_euler = self.get_ori_rpy()
        base_ori_quat = self.get_ori_quat()

        lin_vel = self.local_velocity(self.global_linear_velocity)
        ang_vel = self.local_velocity(self.angular_linear_velocity)

        return {
            "base_pos": base_pos,
            "base_ori_euler": base_ori_euler,
            "base_ori_quat": base_ori_quat,
            "base_velocity": lin_vel,
            "base_ang_vel": ang_vel,
            "j_pos": joint_positions,
            "j_vel": joint_velocities,
        }

    def _new_jms(self, pos):
        """Returns a new jms with default settings at a given position
        :param pos: the new position to set to
        """
        return JointMotorSettings(
            pos,  # position_target
            0.03,  # position_gain
            0.0,  # velocity_target
            1.8,  # velocity_gain
            1.0,  # max_impulse
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


class AlienGo(A1):
    def __init__(self, robot_id):
        super().__init__(robot_id)
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
    def __init__(self, robot_id):
        super().__init__(robot_id)
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
    def __init__(self, robot_id):
        super().__init__(robot_id)
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
    def __init__(self, robot_id):
        super().__init__(robot_id)
        self.name = "Locobot"
        self._initial_joint_positions = []

        # Spawn the URDF 0.425 meters above the navmesh upon reset
        self.robot_spawn_offset = np.array([0.0, 0.25, 0])
        self.robot_dist_to_goal = 0.2
        self.camera_spawn_offset = np.array([0.0, 0.31, -0.55])
        self.urdf_params = np.array([4.19, 0.00, 0.35, 0.35])
