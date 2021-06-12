import gym, gym.spaces
import numpy as np
import magnum as mn
import habitat_sim 
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff
from habitat.tasks.utils import cartesian_to_polar
from utilities.utils import rotate_vector_3d, euler_from_quaternion, get_rpy, quat_to_rad, rotate_pos_from_hab, scalar_vector_to_quat
import squaternion

class A1():
    def __init__(self, sim=None, agent=None, robot_id=0, dt=1/60):
        #self.torque = config.get("torque", 1.0)
        self.high_level_action_dim = 2
        self.sim = sim
        self.agent = agent
        self.robot_id = robot_id
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

    

    def calc_state(self, prev_state=None, finite_diff=False):
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

        joint_positions = self.sim.get_articulated_object_positions(self.robot_id)
        joint_velocities = self.sim.get_articulated_object_velocities(self.robot_id)

        robot_state = self.sim.get_articulated_link_rigid_state(self.robot_id, 0)
        vels = self.sim.get_articulated_link_velocity(self.robot_id, 0)
        lin_vel, ang_vel = vels[0], vels[1]
        base_pos = robot_state.translation
        # base_position[2] = -base_position[2]
        base_orientation_quat = robot_state.rotation
        base_position = base_pos
        base_pos_tmp = rotate_pos_from_hab(base_pos)
        base_position.x = base_pos_tmp[0]
        base_position.y = base_pos_tmp[1]
        base_position.z = base_pos_tmp[2]

        base_orientation_euler = get_rpy(base_orientation_quat)
        # base_orientation_euler_origin = get_rpy(base_orientation_quat, transform=False)

        obs_quat = squaternion.Quaternion(base_orientation_quat.scalar, *base_orientation_quat.vector)
        inverse_base_transform = scalar_vector_to_quat(np.pi/2,(1, 0, 0))
        base_orientation_quat_trans = obs_quat*inverse_base_transform

        if finite_diff:
            if prev_state is None:
                base_velocity = mn.Vector3() #self.sim.get_articulated_link_angular_velocity(self.robot_id, 0)
                frame_pos = np.zeros((3))
                base_angular_velocity_euler = mn.Vector3() # self.sim.get_articulated_link_angular_velocity(self.robot_id, 0)
            else:
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
            'base_ori_quat_hab': base_orientation_quat,
            'base_ori_quat': base_orientation_quat_trans,
            'base_velocity': rotate_vector_3d(base_velocity, *base_orientation_euler),
            # 'base_velocity': list(base_velocity),
            'base_ang_vel': rotate_vector_3d(base_angular_velocity_euler, *base_orientation_euler),
            # 'base_ang_vel': list(base_angular_velocity_euler),
            'j_pos': joint_positions,
            'j_vel': joint_velocities
        }

    def set_mtr_pos(self, joint, ctrl):
        jms = self.sim.get_joint_motor_settings(self.robot_id, joint)
        jms.position_target = ctrl
        self.sim.update_joint_motor(self.robot_id, joint, jms)

    def set_joint_pos(self, joint_idx, angle):
        set_pos = np.array(self.sim.get_articulated_object_positions(self.robot_id))
        set_pos[joint_idx] = angle
        self.sim.set_articulated_object_positions(self.robot_id, set_pos)

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
            self.sim.set_articulated_object_positions(self.robot_id, joint_pos)
        else:
            self.sim.set_articulated_object_positions(self.robot_id, joint_pos)
        # for n, j in enumerate(self.ordered_joints):
        #     a = joint_pos[n]
        #     j.reset_joint_state(position=a, velocity=0.0)

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
        robot_state = self.sim.get_articulated_object_root_state(self.robot_id)
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
    def __init__(self, sim=None, agent=None, robot_id=0, dt=1/60):
        super().__init__(sim=sim, agent=agent, robot_id=robot_id, dt=dt)
        self._initial_joint_positions = [-0.1, 0.60, -1.5,
                                         0.1, 0.60, -1.5,
                                         -0.1, 0.6, -1.5,
                                         0.1, 0.6, -1.5]

class Laikago(A1):
    def __init__(self, sim=None, agent=None, robot_id=0, dt=1/60):
        super().__init__(sim=sim, agent=agent, robot_id=robot_id, dt=dt)
        self._initial_joint_positions = [-0.1, 0.65, -1.2,
                                         0.1, 0.65, -1.2,
                                         -0.1, 0.65, -1.2,
                                         0.1, 0.65, -1.2]

class Spot(A1):
    def __init__(self, sim=None, agent=None, robot_id=0, dt=1/60):
        super().__init__(sim=sim, agent=agent, robot_id=robot_id, dt=dt)
        self._initial_joint_positions = [0.05, 0.7, -1.3,
                                         -0.05, 0.7, -1.3,
                                         0.05, 0.7, -1.3,
                                         -0.05, 0.7, -1.3]
