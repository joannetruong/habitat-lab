import gym, gym.spaces
import numpy as np
import magnum as mn
import habitat_sim 
from utilities.utils import rotate_vector_3d, euler_from_quaternion, get_rpy

class Spot():
    def __init__(self, config, urdf_file="", sim=None, agent=None,robot_id=0, dt=1/60, inverse_transform=None):
        self.config = config
        #self.torque = config.get("torque", 1.0)
        self.high_level_action_dim = 2
        self.sim = sim
        self.agent = agent
        self.robot_id = robot_id
        self.control = "position"
        self.ordered_joints = np.arange(12) # hip out, hip forward, knee
        self._initial_joint_positions = [-.124, 0.876, -1.5,
                                         0.124, 0.876, -1.5,
                                         -0.124, 0.876, -1.5,
                                         0.124, 0.876, -1.5]
        self.dt = dt
        self.inverse_transform_quat = mn.Quaternion.from_matrix(inverse_transform.rotation())
        # self._initial_joint_positions = [0, 0.45, -1,
        #                                  0, 0.45, -1,
        #                                  0, 0.45, -1,
        #                                  0, 0.45, -1]
        #  hip_ids = [1,4,7,10]
        #  hip_out_ids = [0, 3, 6, 9]
        #  knee_ids = [2, 5, 8, 11]


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

    def calc_state(self, prev_state=None):
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
        
        base_position = robot_state.translation
        base_position[2] = -base_position[2]
        base_orientation_quat = robot_state.rotation
        
        # base_orientation_euler = euler_from_quaternion(base_orientation_quat)
        
        
        base_orientation_euler = get_rpy(base_orientation_quat, inverse_transform=self.inverse_transform_quat)

        if prev_state is None:
            base_velocity = mn.Vector3() #self.sim.get_articulated_link_angular_velocity(self.robot_id, 0)
            frame_pos = np.zeros((3))
            base_angular_velocity_euler = mn.Vector3() # self.sim.get_articulated_link_angular_velocity(self.robot_id, 0)
        else:
            base_velocity = (base_position - prev_state['base_pos']) / self.dt
            base_angular_velocity_euler = (base_orientation_euler - prev_state['base_ori_euler']) / self.dt
            frame_pos = rotate_vector_3d(base_velocity, *base_orientation_euler) * self.dt + prev_state['frame_pos']
        base_velocity[1] = base_velocity[2]
        base_angular_velocity_euler = np.clip(base_angular_velocity_euler, -10, 10)
        
        # base_velocity =  np.array([base_velocity.x,base_velocity.y,base_velocity.z]) 

        return {
            'base_pos_x': base_position.x,
            'base_pos_y': base_position.y,
            'base_pos_z': base_position.z,
            'base_pos': np.array([base_position.x,base_position.y,base_position.z]) ,
            'base_ori_euler': base_orientation_euler,
            'base_ori_quat': base_orientation_quat,
            'base_velocity': rotate_vector_3d(base_velocity, *base_orientation_euler),
            'base_ang_vel': rotate_vector_3d(base_angular_velocity_euler, *base_orientation_euler),
            'j_pos': joint_positions,
            'j_vel': joint_velocities,
            'frame_pos': frame_pos,
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
                joint_settings = habitat_sim.physics.JointMotorSettings(0, 0, float(np.clip(a, -1, +1)),.1, 10)
                self.sim.update_joint_motor(self.robot_id, n, joint_settings)

            elif self.control == 'position':
                joint_settings = habitat_sim.physics.JointMotorSettings(float(np.clip(a, -np.pi/2, np.pi/2)), 1, 0,1, 10) # .01 in pos gain
                self.sim.update_joint_motor(self.robot_id, n, joint_settings)

            else:
                print('not implemented yet')

    def robot_specific_reset(self, joint_pos=None):
        if joint_pos is None:
            joint_pos = self._initial_joint_positions
            self.sim.set_articulated_object_positions(self.robot_id, joint_pos)
        # for n, j in enumerate(self.ordered_joints):
        #     a = joint_pos[n]
        #     j.reset_joint_state(position=a, velocity=0.0)

    def step(self, action, dt=1.0/240, verbose=False, get_frames=True, follow_robot=False):
        
        self.apply_robot_action(action=action)
            # simulate dt seconds at 60Hz to the nearest fixed timestep
        if verbose:
            print("Simulating " + str(dt) + " world seconds.")
        observations = []
        start_time = self.sim.get_world_time()
        count = 0
        
        if follow_robot:
            self._follow_robot()

        # while self.sim.get_world_time() < start_time + dt:
        self.sim.step_physics(dt)
        
        if get_frames:
            observations.append(self.sim.get_sensor_observations())
                
        return observations

    def _follow_robot(self):
        robot_state = self.sim.get_articulated_object_root_state(self.robot_id)

        node = self.sim._default_agent.scene_node
        self.h_offset = .5
        cam_pos = mn.Vector3(0, 0.0, 0+self.h_offset)

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


