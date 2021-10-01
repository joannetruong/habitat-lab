from typing import Any, Dict, Optional

import numpy as np

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.simulator import Simulator
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask, TopDownMap
from habitat.utils.visualizations import maps
from .spot_utils.quadruped_env import *
from .spot_utils.daisy_env import *

@registry.register_task(name="MultiNav-v0")
class MultiNavigationTask(NavigationTask):
    def __init__(
            self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.robot_id = None
        self.robots = self._config.ROBOTS
        self.robot_files = self._config.ROBOT_URDFS

    def reset(self, episode: Episode):
         # If robot was never spawned or was removed with previous scene
        if self.robot_id is None or self.robot_id.object_id == -1:
            self._load_robot()

        observations = super().reset(episode)
        return observations

    def _load_robot(self):
        # Add robot into the simulator
        rand_robot = np.random.randint(0,len(self.robot_files))

        art_obj_mgr = self._sim.get_articulated_object_manager()
        self.robot_id = art_obj_mgr.add_articulated_object_from_urdf(
            self.robot_files[rand_robot], fixed_base=False
        )
        if self.robot_id.object_id == -1:
            raise ValueError('Could not load ' + robot_file)

        # Initialize robot wrapper
        self.robot_wrapper = eval(self.robots[rand_robot])(
            sim=self._sim, robot=self.robot_id
        )
        self.robot_wrapper.id = rand_robot
        self.robot_id.joint_positions = self.robot_wrapper._initial_joint_positions

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
        observations['z_in'] = self.robot_wrapper.z_in
        # observations['z_model'] = self.robot_wrapper.z_model

        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )

        return observations








