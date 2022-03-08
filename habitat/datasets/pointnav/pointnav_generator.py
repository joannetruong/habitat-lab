#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""A minimum radius of a plane that a point should be part of to be
considered  as a target or source location. Used to filter isolated points
that aren't part of a floor.
"""

from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

try:
    import magnum as mn
except ModuleNotFoundError:
    pass

import numpy as np
from habitat.core.simulator import ShortestPathPoint
from habitat.datasets.utils import get_action_shortest_path
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from numpy import float64

try:
    from habitat_sim.errors import GreedyFollowerError
except ImportError:
    GreedyFollower = BaseException
try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    habitat_sim = BaseException
ISLAND_RADIUS_LIMIT = 1.5


def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.

    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(
    s: Sequence[float],
    t: Sequence[float],
    sim: "HabitatSim",
    near_dist: float,
    far_dist: float,
    geodesic_to_euclid_ratio: float,
) -> Union[Tuple[bool, float], Tuple[bool, int]]:
    euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)

    if np.abs(s[1] - t[1]) > 0.5:  # check height difference to assure s and
        #  t are from same floor
        return False, 0
    d_separation = sim.geodesic_distance(s, [t])
    if d_separation == np.inf:
        return False, 0
    if not near_dist <= d_separation <= far_dist:
        return False, 0
    distances_ratio = d_separation / euclid_dist
    # if distances_ratio < geodesic_to_euclid_ratio and (
    #         np.random.rand()
    #         > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    # ):
    if distances_ratio < geodesic_to_euclid_ratio:
        return False, 0
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 0

    return True, d_separation


def _create_episode(
    episode_id: Union[int, str],
    scene_id: str,
    start_position: List[float],
    start_rotation: List[Union[int, float64]],
    target_position: List[float],
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None,
    radius: Optional[float] = None,
    info: Optional[Dict[str, float]] = None,
) -> Optional[NavigationEpisode]:
    goals = [NavigationGoal(position=target_position, radius=radius)]
    return NavigationEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
    )


def generate_pointnav_episode(
    sim: "HabitatSim",
    num_episodes: int = -1,
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 10,
    robot: str = "Spot",
) -> Generator[NavigationEpisode, None, None]:
    r"""Generator function that generates PointGoal navigation episodes.

    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.


    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """

    # Load Spot model
    if robot == "A1":
        robot_file = "/coc/testnvme/jtruong33/data/URDF_demo_assets/a1/a1.urdf"
        init_joint_positions = [
            0.05,
            0.60,
            -1.5,
            -0.05,
            0.60,
            -1.5,
            0.05,
            0.65,
            -1.5,
            -0.05,
            0.65,
            -1.5,
        ]
        z_offset = 0.28
    elif robot == "AlienGo":
        robot_file = (
            "/coc/testnvme/jtruong33/data/URDF_demo_assets/aliengo/urdf/aliengo.urdf"
        )
        init_joint_positions = [
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
        z_offset = 0.35
    elif robot == "Daisy":
        robot_file = "/coc/testnvme/jtruong33/data/URDF_demo_assets/daisy/daisy_advanced_akshara.urdf"
        init_joint_positions = [
            0.0,
            1.2,
            -0.5,
            0.0,
            -1.2,
            0.5,
            0.0,
            1.2,
            -0.5,
            0.0,
            -1.2,
            0.5,
            0.0,
            1.2,
            -0.5,
            0.0,
            -1.2,
            0.5,
        ]
        z_offset = 0.14
    elif robot == "Locobot":
        robot_file = "/coc/testnvme/jtruong33/data/URDF_demo_assets/locobot/urdf/locobot_description2.urdf"
        z_offset = -0.02
    elif robot == "Spot":
        robot_file = "/coc/testnvme/jtruong33/data/URDF_demo_assets/spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid.urdf"
        init_joint_positions = [
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
        z_offset = 0.525

    ao_mgr = sim.get_articulated_object_manager()
    robot_id = ao_mgr.add_articulated_object_from_urdf(robot_file, fixed_base=False)

    if robot != "Locobot":
        # Set Spot's joints to default walking position
        robot_id.joint_positions = init_joint_positions

    # Rotation offset matrices
    roll_offset = mn.Matrix4.rotation(
        mn.Rad(-np.pi / 2.0),
        mn.Vector3((1.0, 0.0, 0.0)),
    )
    yaw_offset = mn.Matrix4.rotation(
        mn.Rad(np.pi),
        mn.Vector3((0.0, 1.0, 0.0)),
    )

    episode_count = 0
    num_trials = num_episodes * 100
    ctr = -1
    while episode_count < num_episodes or num_episodes < 0:
        print(episode_count)
        if ctr > num_trials:
            print("EPISODE COUNT: ", episode_count)
            break
        ctr += 1
        try:
            target_position = sim.sample_navigable_point()
        except:
            print("unable to find navigable point")
            ctr = num_trials + 1
            continue

        if sim.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
            continue

        for _retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()

            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )
            if is_compatible:
                break
        if is_compatible:

            def cartesian_to_polar(x, y):
                rho = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                return rho, phi

            def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
                r"""Rotates a vector by a quaternion
                Args:
                    quaternion: The quaternion to rotate by
                    v: The vector to rotate
                Returns:
                    np.array: The rotated vector
                """
                vq = np.quaternion(0, 0, 0, 0)
                vq.imag = v
                return (quat * vq * quat.inverse()).imag

            def quat_to_rad(rotation):
                heading_vector = quaternion_rotate_vector(
                    rotation.inverse(), np.array([0, 0, -1])
                )
                phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
                return phi

            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

            # Move Spot model to source position and rotation
            heading = np.quaternion(source_rotation[-1], *source_rotation[:-1])
            heading = -quat_to_rad(heading) + np.pi / 2

            robot_rigid_state = (
                mn.Matrix4.rotation_y(
                    mn.Rad(-heading),
                )
                .__matmul__(yaw_offset)
                .__matmul__(roll_offset)
            )
            robot_rigid_state.translation = np.array(source_position) + np.array(
                [
                    0.0,
                    z_offset,
                    0.0,
                ]
            )

            robot_id.transformation = robot_rigid_state
            collided = sim.contact_test(robot_id.object_id)

            if collided:
                continue

            shortest_paths = None
            if is_gen_shortest_path:
                try:
                    shortest_paths = [
                        get_action_shortest_path(
                            sim,
                            source_position=source_position,
                            source_rotation=source_rotation,
                            goal_position=target_position,
                            success_distance=shortest_path_success_distance,
                            max_episode_steps=shortest_path_max_steps,
                        )
                    ]
                # Throws an error when it can't find a path
                except GreedyFollowerError:
                    continue

            episode = _create_episode(
                episode_id=episode_count,
                scene_id=sim.habitat_config.SCENE,
                start_position=source_position,
                start_rotation=source_rotation,
                target_position=target_position,
                shortest_paths=shortest_paths,
                radius=shortest_path_success_distance,
                info={"geodesic_distance": dist},
            )

            episode_count += 1
            yield episode
