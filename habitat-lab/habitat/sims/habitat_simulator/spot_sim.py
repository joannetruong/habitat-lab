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

@registry.register_simulator(name="SpotSim-v0")
class SpotSim(HabitatSim):
    def __init__(self, config):

        super().__init__(config)

        agent_config = self.habitat_config
        #self.navmesh_settings = get_nav_mesh_settings(self._get_agent_config())
        self.robot_id = None
        self.first_setup = True
        self.is_render_obs = False
        self.pov_mode = agent_config.POV
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


    def reconfigure(self, config):
        ep_info = config['ep_info'][0]

        config['SCENE'] = ep_info['scene_id']
        super().reconfigure(config)

        self.ep_info = ep_info
        self.fixed_base = ep_info['fixed_base']

        self.target_obj_ids = []
        self.event_callbacks = []

        if ep_info['scene_id'] != self.prev_scene_id:
            # Object instances are not valid between scenes.
            self.art_obj_ids = []

            self.robot_id = None

        self.prev_scene_id = ep_info['scene_id']


        set_pos = {}
        # Set articulated object joint states.
        if self.habitat_config.get('LOAD_ART_OBJS', True):
            for i, art_state in self.start_art_states.items():
                set_pos[i] = art_state
            for i,art_state in ep_info['art_states']:
                set_pos[self.art_obj_ids[i]] = art_state
            init_art_objs(set_pos.items(), self._sim,
                    self.habitat_config.get('AUTO_SLEEP_ART_OBJS', True))

        # Get the positions after things have settled down.
        self.settle_sim(self.habitat_config.get("SETTLE_TIME", 0.1))

        # Get the starting positions of the target objects.
        scene_pos = self.get_scene_pos()
        self.target_start_pos = np.array([scene_pos[idx]
            for idx, _ in self.ep_info['targets']])



        if self.first_setup:
            self.first_setup = False
            self._ik.setup_sim()
            # Capture the starting art states
            for i in self.art_obj_ids:
                self.start_art_states[i] = self._sim.get_articulated_object_positions(i)

        self.update_i = 0
        self.allowed_region = ep_info['allowed_region']
        self._load_markers(ep_info)


    def _load_navmesh(self):
        """
        Generates the navmesh if it was not specified. This must be called
        BEFORE adding any object / articulated objects to the scene.
        """
        art_bb_ids = self._add_art_bbs()
        # Add bounding boxes for articulated objects
        self._sim.recompute_navmesh(self._sim.pathfinder,
                self.navmesh_settings, include_static_objects=True)
        for idx in art_bb_ids:
            self._sim.remove_object(idx)
        if self.habitat_config.get('SAVE_NAVMESH', False):
            scene_name = self.ep_info['scene_id']
            inferred_path = scene_name.split('.glb')[0] + '.navmesh'
            self._sim.pathfinder.save_nav_mesh(inferred_path)
            print('Cached navmesh to ', inferred_path)





    def reset(self):
        self.event_callbacks = []
        ret = super().reset()
        if self._light_setup:
            # Lighting reconfigure NEEDS to be in the reset function and NOT
            # the reconfigure function!
            self._sim.set_light_setup(self._light_setup)

        return ret

    def viz_pos(self, pos, viz_id=None, r=0.05):
        if viz_id is None:
            obj_mgr = self._sim.get_object_template_manager()
            template = obj_mgr.get_template_by_handle(obj_mgr.get_template_handles("sphere")[0])
            template.scale = mn.Vector3(r,r,r)
            new_template_handle = obj_mgr.register_template(template, "ball_new_viz")
            viz_id = self._sim.add_object(new_template_handle)
            make_render_only(viz_id, self._sim)
        self._sim.set_translation(mn.Vector3(*pos), viz_id)

        return viz_id

    @property
    def _sim(self):
        return self

    def clear_objs(self, art_names=None):
        # Clear the objects out.
        for scene_obj in self.scene_obj_ids:
            self._sim.remove_object(scene_obj)
        self.scene_obj_ids = []

        if art_names is None or self.cached_art_obj_ids != art_names:
            for art_obj in self.art_obj_ids:
                self._sim.remove_articulated_object(art_obj)
            self.art_obj_ids = []


    def _add_objs(self, ep_info):
        art_names = [x[0] for x in ep_info['art_objs']]
        self.clear_objs(art_names)

        if self.habitat_config.get('LOAD_ART_OBJS', True):
            self.art_obj_ids = load_articulated_objs(
                    convert_legacy_cfg(ep_info['art_objs']),
                    self._sim, self.art_obj_ids,
                    auto_sleep=self.habitat_config.get('AUTO_SLEEP', True))
            self.cached_art_obj_ids = art_names
            self.art_name_to_id = {name.split('/')[-1]: art_id for name, art_id in
                    zip(art_names, self.art_obj_ids)}
            self._create_art_bbs()

        self._load_navmesh()

        if self.habitat_config.get('LOAD_OBJS', True):
            self.scene_obj_ids = load_objs(
                    convert_legacy_cfg(ep_info['static_objs']),
                    self._sim, obj_ids=self.scene_obj_ids,
                    auto_sleep=self.habitat_config.get('AUTO_SLEEP', True))

            for idx, _ in ep_info['targets']:
                self.target_obj_ids.append(self.scene_obj_ids[idx])
        else:
            self.ep_info['targets'] = []

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


    def _load_robot(self, ep_info):
        if not self.habitat_config.get('LOAD_ROBOT', True):
            return

        if self.robot_id is None:
            agent_config = self.habitat_config
            urdf_name = agent_config.ROBOT_URDF
            art_obj_mgr = self.sim.get_articulated_object_manager()
            self.robot_hab = art_obj_mgr.add_articulated_object_from_urdf(robot_file, fixed_base=self.fixed_base)
            self.robot_id = self.robot_hab.object_id
            if self.robot_id == -1:
                raise ValueError('Could not load ' + urdf_name)

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
        
        # Setting the rotation as a single number indicating rotation on z-axis
        # or as a full quaternion consisting of 4 numbers

        # start_rot = ep_info['start_rotation']
        # if isinstance(start_rot, list):
        #     rot_quat = mn.Quaternion(start_rot[:3], start_rot[3])
        #     add_rot_mat = mn.Matrix4.from_(rot_quat.to_matrix(), mn.Vector3(0,0,0))
        # else:
        #     add_rot_mat = mn.Matrix4.rotation(mn.Deg(start_rot), mn.Vector3(0.0, 0, 1))
        # base_transform = rot_trans @ add_rot_mat
        # self._sim.set_articulated_object_root_state(self.robot_id, base_transform)
        # robo_start = self.habitat_config.get('ROBOT_START', None)

        # if robo_start is not None:
        #     self.start_pos = eval(robo_start.replace('/', ','))
        #     self.start_pos = [self.start_pos[0], 0.15, self.start_pos[1]]
        #     self.start_pos = self._sim.pathfinder.snap_point(self.start_pos)
        # else:
        #     start_pos = ep_info['start_position']
        #     if start_pos == [0,0]:
        #         # Hand tuned constants for the ReplicaCAD dataset to spawn the
        #         # robot in reasonable areas.
        #         start_pos = get_largest_island_point(self._sim, 0.15, -0.2)
        #     elif len(start_pos) == 2:
        #         start_pos = [start_pos[0], 0.15, start_pos[1]]
        #         start_pos = self._sim.pathfinder.snap_point(start_pos)
        #     self.start_pos = start_pos

        # base_transform.translation = mn.Vector3(self.start_pos)
        # self._sim.set_articulated_object_root_state(self.robot_id, base_transform)


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

        if self.pov_mode == 'bird':
            cam_pos = mn.Vector3(0.0, 0.0, 4.0)
        elif self.pov_mode == '3rd':
            cam_pos = mn.Vector3(0.0, -1.2, 1.5)
        elif self.pov_mode == '1st':
            cam_pos = mn.Vector3(0.17, 0.0, 0.90+self.h_offset)
        elif self.pov_mode == 'move':
            cam_pos = mn.Vector3(*self.move_cam_pos)
        else:
            raise ValueError()

        look_at = mn.Vector3(1, 0.0, 0.75)
        look_at = robot_state.transform_point(look_at)
        if self.pov_mode == 'move':
            agent_config = self.habitat_config
            if 'LOOK_AT' in agent_config:
                x,y,z = agent_config.LOOK_AT
            else:
                x,y,z = self.get_end_effector_pos()
            look_at = mn.Vector3(x,y,z)
        else:
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
        self.update_i += 1

        if self.is_render_obs:
            self._sim._try_acquire_context()
            for obj_idx, _ in self.ep_info['targets']:
                self._sim.set_object_bb_draw(False, self.scene_obj_ids[obj_idx])
        for viz_obj in self.viz_obj_ids:
            self._sim.remove_object(viz_obj)

        add_back_viz_objs = {}
        for name, viz_id in self.viz_ids.items():
            if viz_id is None:
                continue

            before_pos = self._sim.get_translation(viz_id)
            self._sim.remove_object(viz_id)
            add_back_viz_objs[name] = before_pos
        self.viz_obj_ids = []
        self.viz_ids = defaultdict(lambda: None)
        self._follow_robot()

        remove_idxs = []
        for i, event in enumerate(self.event_callbacks):
            if event.is_ready():
                event.run()
                remove_idxs.append(i)

        for i in reversed(remove_idxs):
            del self.event_callbacks[i]

        if not self.concur_render:
            if self.habitat_config.get('STEP_PHYSICS', True):
                with rutils.TimeProfiler("sim.step.phys_step", self):
                    for i in range(self.ac_freq_ratio):
                        self.internal_step(-1)

            with rutils.TimeProfiler("sim.step.sensors", self):
                self._prev_sim_obs = self._sim.get_sensor_observations()
                obs = self._sensor_suite.get_observations(self._prev_sim_obs)

        else:
            with rutils.TimeProfiler("sim.step.sensors_async_start", self):
                self._prev_sim_obs = self._sim.get_sensor_observations_async_start()

            if self.habitat_config.get('STEP_PHYSICS', True):
                with rutils.TimeProfiler("sim.step.phys_step", self):
                    for i in range(self.ac_freq_ratio):
                        self.internal_step(-1)

            with rutils.TimeProfiler("sim.step.sensors_async_finish", self):
                self._prev_sim_obs = self._sim.get_sensor_observations_async_finish()
                obs = self._sensor_suite.get_observations(self._prev_sim_obs)

        if 'high_rgb' in obs:
            self.is_render_obs = True
            self._sim._try_acquire_context()
            with rutils.TimeProfiler("sim.step.high_render", self):
                for k, pos in add_back_viz_objs.items():
                    self.viz_ids[k] = self.viz_pos(pos)

                # Also render debug information
                if self.habitat_config.get('RENDER_TARGS', True):
                    self._create_obj_viz(self.ep_info)

                # Always draw the target
                for obj_idx, _ in self.ep_info['targets']:
                    self._sim.set_object_bb_draw(True, self.scene_obj_ids[obj_idx])

                debug_obs = self._sim.get_sensor_observations()
                obs['high_rgb'] = debug_obs['high_rgb'][:,:,:3]

        if self.habitat_config.HABITAT_SIM_V0.get("ENABLE_GFX_REPLAY_SAVE", False):
            self._sim.gfx_replay_manager.save_keyframe()

        return obs

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

