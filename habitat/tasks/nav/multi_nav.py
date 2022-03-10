from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import (IntegratedPointGoalGPSAndCompassSensor,
                                   NavigationTask, TopDownMap)
from habitat.utils.visualizations import maps

from .spot_utils.daisy_env import *
from .spot_utils.quadruped_env import *


@registry.register_task(name="MultiNav-v0")
class MultiNavigationTask(NavigationTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.robot_id = None
        self.robot = self._config.ROBOT
        self.robot_file = self._config.ROBOT_URDF
        # if task reset happens everytime episode is reset, then create previous state deck here

    def reset(self, episode: Episode):
        # If robot was never spawned or was removed with previous scene
        # if randomly selected robot is not the current robot already spawned
        # if there is a robot/ URDF created
        if self.robot_id is not None and self.robot_id.object_id != -1 and self.robot_wrapper.id != 0:
            self.art_obj_mgr.remove_object_by_id(self.robot_id.object_id)
            self.robot_id = None
        if self.robot_id is None or self.robot_id.object_id == -1:
            self._load_robot()

        observations = super().reset(episode)
        return observations

    def _load_robot(self):
        # Add robot into the simulator

        self.art_obj_mgr = self._sim.get_articulated_object_manager()
        self.robot_id = self.art_obj_mgr.add_articulated_object_from_urdf(
            self.robot_file, fixed_base=False
        )

        if self.robot_id.object_id == -1:
            raise ValueError("Could not load " + self.robot_file)

        # Initialize robot wrapper
        self.robot_wrapper = eval(self.robot)(
            sim=self._sim, robot=self.robot_id, rand_id=0
        )
        self.robot_wrapper.id = 0
        if self.robot_wrapper.name != "Locobot":
            self.robot_id.joint_positions = self.robot_wrapper._initial_joint_positions

        if self._sim._sensors.get("depth", False):
            depth_sensor = self._sim._sensors["depth"]
            depth_pos_offset = (
                np.array([0.0, 0.0, 0.0]) + self.robot_wrapper.camera_spawn_offset
            )
            depth_sensor._spec.position = depth_pos_offset
            depth_sensor._sensor_object.set_transformation_from_spec()

    def step(self, action: Dict[str, Any], episode: Episode):
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        action_name = action["action"]
        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        assert (
            action_name in self.actions
        ), f"Can't find '{action_name}' action in {self.actions.keys()}."

        task_action = self.actions[action_name]
        observations = task_action.step(**action["action_args"], task=self)
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                action=action,
                task=self,
            )
        )
        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )

        return observations
