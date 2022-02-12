from typing import Union, Optional, List

from habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorSuite,
    Simulator,
    VisualObservation,
)
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.config.default import Config

from habitat.utils.geometry_utils import (
    get_heading_error,
    quat_to_rad,
)
from habitat.tasks.utils import cartesian_to_polar

import habitat_sim

import random
import quaternion
import magnum as mn
import math
import numpy as np

@registry.register_simulator(name="iGibsonInteractiveNav")
class iGibsonInteractiveNav(HabitatSim):
    def __init__(self, config: Config) -> None:
        super().__init__(config=config)
        obj_templates_mgr = self.get_object_template_manager()
        self.obj_template_ids = obj_templates_mgr.load_configs(
            "/private/home/naokiyokoyama/gc/datasets/object_meshes"
        )
        self.social_nav = False
        self.interactive_nav = True

    def reset_objects(self, episode):
        agent_position = self.get_agent_state().position
        obj_templates_mgr = self.get_object_template_manager()

        # Check if humans have been erased (sim was reset)
        if not self.get_existing_object_ids():
            self.obj_ids = []
            for obj_template_id in self.obj_template_ids:
                obj_id = self.add_object(obj_template_id)
                self.obj_ids.append(obj_id)
                self.set_object_motion_type(
                    habitat_sim.physics.MotionType.KINEMATIC,
                    obj_id
                )

        # Get points around which the objects will spawn
        path = habitat_sim.ShortestPath()
        path.requested_start = episode.start_position
        path.requested_end = np.array(
            episode.goals[0].position, dtype=np.float32
        )
        self.pathfinder.find_path(path)

        spawn_areas = []
        point_idx = 0
        for idx, p1 in enumerate(path.points[:-1]):
            p1 = path.points[point_idx]
            p2 = path.points[point_idx+1]
            curr_dist = 0.5
            euclid_dist = np.sqrt( (p1[0]-p2[0])**2 + (p1[2]-p2[2])**2 )
            while curr_dist < euclid_dist:
                new_x = p1[0]+(p2[0]-p1[0]*curr_dist/euclid_dist)
                new_y = p1[2]+(p2[2]-p1[2]*curr_dist/euclid_dist)
                spawn_areas.append([new_x, p1[1], new_y])
                curr_dist += 0.5
            curr_dist = 0
            spawn_areas.append(path.points[point_idx+1])
            point_idx += 1

        spawn_points = []
        for sa in spawn_areas:
            found = False
            for _ in range(20):
                rand_r = 0.5 * np.sqrt(np.random.rand()) # TODO make this adjustable
                rand_theta = np.random.rand() * 2 * np.pi
                x = sa[0] + rand_r * np.cos(rand_theta)
                y = sa[2] + rand_r * np.sin(rand_theta)
                if self.pathfinder.is_navigable([x, sa[1], y]):
                    found = True
                    break
            if not found:
                spawn_points.append(sa)
            else:
                spawn_points.append([x, sa[1], y])

        shuffled_obj_ids = list(self.obj_ids)
        np.random.shuffle(shuffled_obj_ids)
        self.object_positions = []
        for idx, sp in enumerate(spawn_points):
            if idx >= len(shuffled_obj_ids):
                break
            obj_id = shuffled_obj_ids[idx]

            pos = np.array([sp[0], sp[1]+0.05, sp[2]], dtype=np.float32)
            self.set_translation(pos, obj_id)
            self.object_positions.append([sp[0], sp[1]+0.05, sp[2]])

            heading = np.random.rand()*2*np.pi-np.pi
            
            rotation = np.quaternion(np.cos(heading),0,np.sin(heading),0)
            rotation = np.normalized(rotation)
            rotation = mn.Quaternion(
                rotation.imag, rotation.real
            )
            self.set_rotation(rotation, obj_id)

        for idx_left in range(idx, len(shuffled_obj_ids)):
            self.set_translation([0.0, 10.0, 0.0], shuffled_obj_ids[idx_left])

