from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask, merge_sim_episode_config
import habitat
from habitat.core.dataset import Dataset
import numpy as np
import magnum as mn
import attr
import quaternion
import habitat_sim
import os.path as osp
from habitat.config.default import _C, CN
from habitat_sim.scene import SceneNode
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration
)
from habitat_sim.utils import profiling_utils
import torch
import math
import re
from collections import defaultdict

from habitat_sim.physics import MotionType

from .spot_utils.quadruped_env import A1, AlienGo, Laikago, Spot
from .spot_utils.daisy_env import Daisy, Daisy_4legged
from .spot_utils.raibert_controller import Raibert_controller
from .spot_utils.raibert_controller import Raibert_controller_turn_stable
import cv2
import json
import time
from .spot_utils import utils

from typing import Callable

import random
import copy


def merge_sim_episode_with_object_config(sim_config, episode):
    sim_config.defrost()
    sim_config.ep_info = [episode.__dict__]
    sim_config.freeze()
    return sim_config

@registry.register_task(name="SpotTask-v0")
class SpotTask(NavigationTask):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)



@attr.s(auto_attribs=True, slots=True)
class SimEvent:
    is_ready: Callable[[], bool]
    run: Callable[[], None]


EE_GRIPPER_OFFSET = mn.Vector3(0.08,0,0)

@registry.register_simulator(name="SpotSim-v1")
class SpotSimv2(HabitatSim):
    def __init__(self, config):

        super().__init__(config)
        print('INIT SPOT SIM')

        agent_config = self.habitat_config
        #self.navmesh_settings = get_nav_mesh_settings(self._get_agent_config())
        self.robot_id = None
        self.first_setup = True
        self.is_render_obs = False
        self.fixed_base = False
        
        self.update_i = 0
        self.h_offset = 0.3
        self.ep_info = None
        self.do_grab_using_constraint = True
        self.snap_to_link_on_grab = True
        self.snapped_obj_id = None
        self.snapped_marker_name = None
        self.snapped_obj_constraint_id = []
        self.prev_loaded_navmesh = None
        self.prev_scene_id = None
        self.robot_name = agent_config.ROBOT_URDF.split("/")[-1].split(".")[0]

        self.pos_gain = np.ones((3,)) * 0.1 # 0.2
        self.vel_gain = np.ones((3,)) * 1.0 # 1.5
        self.pos_gain[2] = 0.1 # 0.7
        self.vel_gain[2] = 1.0 # 1.5
        
        if 'CAM_START' in agent_config:
            self.move_cam_pos = np.array(agent_config.CAM_START)

        # Number of physics updates per action
        self.ac_freq_ratio = agent_config.AC_FREQ_RATIO
        # The physics update time step.
        self.ctrl_freq = agent_config.CTRL_FREQ
        # Effective control speed is (ctrl_freq/ac_freq_ratio)


        # Horrible hack to get data from the RL environment class to sensors.
        self.track_markers = []
        self._goal_pos = None
        self._load_robot()

    @property
    def _sim(self):
        return self

    def set_robot_pos(self, set_pos):
        """
        - set_pos: 2D coordinates of where the robot will be placed. The height
          will be same as current position.
        """
        base_transform = self._sim.get_articulated_object_root_state(self.robot_id)
        pos = base_transform.translation
        base_transform.translation = mn.Vector3(set_pos[0], pos[1], set_pos[1])
        self._sim.set_articulated_object_root_state(self.robot_id, base_transform)

    def set_robot_rot(self, rot_rad):
        """
        Set the rotation of the robot along the y-axis. The position will
        remain the same.
        """
        cur_trans = self._sim.get_articulated_object_root_state(self.robot_id)
        pos = cur_trans.translation

        rot_trans = mn.Matrix4.rotation(mn.Rad(-1.56), mn.Vector3(1.0, 0, 0))
        add_rot_mat = mn.Matrix4.rotation(mn.Rad(rot_rad), mn.Vector3(0.0, 0, 1))
        new_trans = rot_trans @ add_rot_mat
        new_trans.translation = pos
        self._sim.set_articulated_object_root_state(self.robot_id, new_trans)


    def _load_robot(self):
        print('loading robot')
        if not self.habitat_config.get('LOAD_ROBOT', True):
            return

        if self.robot_id is None:
            agent_config = self.habitat_config
            robot_file = agent_config.ROBOT_URDF
            art_obj_mgr = self._sim.get_articulated_object_manager()
            print('robot file: ', robot_file)
            backend_cfg = habitat_sim.SimulatorConfiguration()
            self.robot_hab = art_obj_mgr.add_articulated_object_from_urdf(robot_file, fixed_base=self.fixed_base)
            print('self.robot hab: ', self.robot_hab)
            self.robot_id = self.robot_hab.object_id
            if self.robot_id == -1:
                raise ValueError('Could not load ' + robot_file)

        jms = []
        jms.append(habitat_sim.physics.JointMotorSettings(
                0,  # position_target
                self.pos_gain[0],  # position_gain
                0,  # velocity_target
                self.vel_gain[0],  # velocity_gain
                10.0,  # max_impulse
            ))

        jms.append(habitat_sim.physics.JointMotorSettings(
                        0,  # position_target
                        self.pos_gain[1],  # position_gain
                        0,  # velocity_target
                        self.vel_gain[1],  # velocity_gain
                        10.0,  # max_impulse
                    ))
        jms.append(habitat_sim.physics.JointMotorSettings(
                        0,  # position_target
                        self.pos_gain[2],  # position_gain
                        0,  # velocity_target
                        self.vel_gain[2],  # velocity_gain
                        10.0,  # max_impulse
                    ))      
        for i in range(12):
            self.robot_hab.update_joint_motor(i, jms[np.mod(i,3)])
        rot_trans = mn.Matrix4.rotation(mn.Rad(-1.56), mn.Vector3(1.0, 0, 0))

        if self.robot_name == 'A1':
            self.robot_wrapper = A1(sim=self.sim, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'AlienGo':
            self.robot_wrapper = AlienGo(sim=self.sim, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'Daisy':
            self.robot_wrapper = Daisy(sim=self.sim, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'Laikago':
            self.robot_wrapper = Laikago(sim=self.sim, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'Daisy_4legged':
            self.robot_wrapper = Daisy4(sim=self.sim, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)
        elif self.robot_name == 'Spot':
            self.robot_wrapper = Spot(sim=self.sim, robot=self.robot_hab, agent=self.agent, robot_id=self.robot_id, dt=1/self.ctrl_freq)

        self.robot_wrapper.robot_specific_reset()
        
        # Set up Raibert controller and link it to spot
        action_limit = np.zeros((12, 2))
        action_limit[:, 0] = np.zeros(12) + np.pi / 2
        action_limit[:, 1] = np.zeros(12) - np.pi / 2
        self.raibert_controller = Raibert_controller_turn_stable(control_frequency=self.ctrl_freq, num_timestep_per_HL_action=self.time_per_step, robot=self.robot_name)

    def reset_robot(self, start_pose):
        self.robot_hab.translation = start_pose
        self.robot_hab.rotation = mn.Quaternion.rotation(mn.Rad(-1.57), mn.Vector3(1.0, 0.0, 0.0).normalized())
        self.init_state = self.robot_wrapper.calc_state(prev_state=self.prev_state, finite_diff=self.finite_diff)
        self.raibert_controller.set_init_state(self.init_state)
        pos, ori = self.get_robot_pos()
        print('robot start pos: ', pos, np.rad2deg(ori[-1]))

    def get_robot_pos(self):
        robot_state = self.robot_hab.rigid_state
        base_pos_hab = utils.rotate_pos_from_hab(robot_state.translation)

        robot_position = np.array([base_pos_hab[0], base_pos_hab[1], base_pos_hab[2]])
        robot_ori = utils.get_rpy(robot_state.rotation)

        return robot_position, robot_ori

    def move_cam(self, delta_xyz):
        self.move_cam_pos += np.array(delta_xyz)

    def _follow_robot(self):
        robot_state = self._sim.get_articulated_object_root_state(self.robot_id)

        node = self._sim._default_agent.scene_node

        # if self.pov_mode == 'bird':
        #     cam_pos = mn.Vector3(0.0, 0.0, 4.0)
        # elif self.pov_mode == '3rd':
        #     cam_pos = mn.Vector3(0.0, -1.2, 1.5)
        # elif self.pov_mode == '1st':
        #     cam_pos = mn.Vector3(0.17, 0.0, 0.90+self.h_offset)
        # elif self.pov_mode == 'move':
        #     cam_pos = mn.Vector3(*self.move_cam_pos)
        # else:
        #     raise ValueError()

        look_at = mn.Vector3(1, 0.0, 0.75)
        look_at = robot_state.transform_point(look_at)
        # if self.pov_mode == 'move':
        #     agent_config = self.habitat_config
        #     if 'LOOK_AT' in agent_config:
        #         x,y,z = agent_config.LOOK_AT
        #     else:
        #         x,y,z = self.get_end_effector_pos()
        #     look_at = mn.Vector3(x,y,z)
        # else:
        cam_pos = robot_state.transform_point(cam_pos)

        node.transformation = mn.Matrix4.look_at(
                cam_pos,
                look_at,
                mn.Vector3(0, -1, 0))
        #print('node at  :', ['%.2f' % x for x in node.transformation.translation])

        self.cam_trans = node.transformation
        self.cam_look_at = look_at
        self.cam_pos = cam_pos

        # Lock all arm cameras to the end effector.
        for k in self._sensors:
            if 'arm' not in k:
                continue
            sens_obj = self._sensors[k]._sensor_object
            cur_t = sens_obj.node.transformation

            link_rigid_state = self._sim.get_articulated_link_rigid_state(self.robot_id, self.ee_link)
            ee_trans = mn.Matrix4.from_(link_rigid_state.rotation.to_matrix(), link_rigid_state.translation)

            offset_trans = mn.Matrix4.translation(mn.Vector3(0, 0.0, 0.1))
            rot_trans = mn.Matrix4.rotation_y(mn.Deg(-90))
            spin_trans = mn.Matrix4.rotation_z(mn.Deg(90))
            arm_T = ee_trans @ offset_trans @ rot_trans @ spin_trans
            sens_obj.node.transformation = node.transformation.inverted() @ arm_T

        # Viz the camera position
        #self.viz_marker = self.viz_pos(self.cam_pos, self.viz_marker)


    def path_to_point(self, point):
        trans = self.get_robot_transform()
        agent_pos = trans.translation
        closest_point = self._sim.pathfinder.snap_point(point)
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = closest_point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        if len(path.points) == 1:
            return [agent_pos, path.points[0]]
        return path.points

    def inter_target(self, targs, idxs, seconds):
        curs = np.array([self.get_mtr_pos(i) for i in idxs])
        diff = targs - curs
        T = int(seconds * self.ctrl_freq)
        delta = diff / T

        for i in range(T):
            for j, jidx in enumerate(idxs):
                self.set_mtr_pos(jidx, delta[j]*(i+1)+curs[j])
                self.set_joint_pos(jidx, delta[j]*(i+1)+curs[j])
            self._sim.step_world(1/self.ctrl_freq)

    def step(self, action): 
        sim_obs = super().step(action)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        return observations

    def get_agent_state(self, agent_id=0):
        prev_state = super().get_agent_state()
        trans = self.get_robot_transform()
        pos = np.array(trans.translation)
        rot = mn.Quaternion.from_matrix(trans.rotation())
        rot = quaternion.quaternion(*rot.vector, rot.scalar)
        new_state = copy.copy(prev_state)
        new_state.position = pos
        new_state.rotation = rot
        return new_state

