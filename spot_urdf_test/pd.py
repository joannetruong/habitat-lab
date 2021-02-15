# [setup]

import os
import tqdm
import cv2

import magnum as mn
from datetime import datetime
import numpy as np

import habitat_sim

# import habitat_sim.utils.common as ut
import habitat_sim.utils.viz_utils as vut

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../habitat-sim/data")
output_path = os.path.join(dir_path, "URDF_robotics_tutorial_output/")


def remove_all_objects(sim):
    for id in sim.get_existing_object_ids():
        sim.remove_object(id)
    for id in sim.get_existing_articulated_object_ids():
        sim.remove_articulated_object(id)


def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = [-0.15, -0.7, 1.0]
    # agent_state.position = [-0.15, -1.6, 1.0]
    agent_state.rotation = np.quaternion(-0.83147, 0, 0.55557, 0)
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()


def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/empty_room.glb"
    backend_cfg.enable_physics = True

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup 2 rgb sensors for 1st and 3rd person views
    camera_resolution = [540, 720]
    sensors = {
        "rgba_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, 0.2, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "depth_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": camera_resolution,
            "position": [0.0, 0.6, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "rgba_camera_3rdperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position":  [-2.0,2.0,0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.SensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.orientation = sensor_params["orientation"]
        sensor_specs.append(sensor_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])

import time
last_render = time.time()
def simulate(sim, dt=1.0, get_frames=True, show=True, text=None):
    global last_render
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 30.0)
        if get_frames:
            observation = sim.get_sensor_observations()
            if show:
                img = cv2.cvtColor(observation['rgba_camera_1stperson'], cv2.COLOR_RGB2BGR)
                

                link_rigid_state = sim.get_articulated_link_rigid_state(0,17)
                # print(list(link_rigid_state.translation))
                text = ' '.join(list(map(lambda x: str(x)[:6],link_rigid_state.translation)))


                if text is not None:
                    height = img.shape[0]
                    cv2.putText(img, text, (0,height-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)

                wait_ms = max(1,int(1000*round(last_render+0.017 - time.time())))
                key = cv2.waitKey(wait_ms)
                if key == ord('q'):
                    exit()
                cv2.imshow('aliengo',img)
                last_render = time.time()
            if show:
                observation['rgba_camera_1stperson'] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            observations.append(observation)

    return observations


# [/setup]

# This is wrapped such that it can be added to a unit test
def main(make_video=True, show_video=True):
    if make_video:
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    # [initialize]
    # create the simulator
    cfg = make_configuration()
    sim = habitat_sim.Simulator(cfg)
    agent_transform = place_agent(sim)
    observations = []

    # [basics]

    # load a URDF file
    robot_file = os.path.join(data_path, "URDF_demo_assets/aliengo/urdf/aliengo.urdf")
    robot_id = sim.add_articulated_object_from_urdf(robot_file, True)
    # robot_id = sim.add_articulated_object_from_urdf(robot_file)

    # place the robot root state relative to the agent
    # local_base_pos = np.array([0.0, -0.45, -2.0])
    local_base_pos = np.array([0, 2, 2])
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    base_transform = mn.Matrix4.rotation(mn.Rad(-1.56), mn.Vector3(1.0, 0, 0))
    base_transform.translation = agent_transform.transform_point(local_base_pos)
    sim.set_articulated_object_root_state(robot_id, base_transform)
    # sim.set_gravity([0.,0.,0.])
    pose = sim.get_articulated_object_positions(robot_id)
    sim.set_articulated_object_positions(robot_id, pose)

    pose = sim.get_articulated_object_positions(robot_id)
    calfDofs = [2, 5, 8, 11]
    for dof in calfDofs:
        pose[dof] = -1.0
        pose[dof - 1] = 0.45
        # also set a thigh
    sim.set_articulated_object_positions(robot_id, pose)

    observations += simulate(sim, dt=0.5, get_frames=make_video,show=False, text='Gravity Only Fixed Base')
    # sim.step_physics()
    # motor_settings.position_gain   = 0.03
    # motor_settings.velocity_target = 0.0
    # motor_settings.velocity_gain   = 0.8
    # motor_settings.max_impulse     = 0.1

    for _ in range(30*10):
        pose = sim.get_articulated_object_positions(robot_id)
        for idx,angle in enumerate(pose):
            scale = np.pi/180*10
            if angle > 0:
                target = max(0,angle-scale)
            else:
                target = min(0,angle+scale)
            import random
            # target = angle+scale if random.random() < 0.5 else angle-scale
            motor_settings = sim.get_joint_motor_settings(robot_id, idx)
            motor_settings.position_target = target
            # motor_settings.position_gain   = 0.8
            motor_settings.position_gain   = 0.6
            motor_settings.velocity_target = 0.0
            # motor_settings.velocity_gain   = 1.5
            motor_settings.velocity_gain   = 1.5
            # motor_settings.max_impulse     = 0.1
            motor_settings.max_impulse     = 10
            sim.update_joint_motor(robot_id, idx, motor_settings)
        # simulate
        observations += simulate(sim, dt=1./30., get_frames=make_video,show=False, text='PD Control Straight Legs')

    if make_video:
        vut.make_video(
            observations,
            "rgba_camera_3rdperson",
            "color",
            output_path + "locate_link" + datetime.now().strftime("%d%m%y_%H_%M"),
            open_vid=show_video,
        )

if __name__ == "__main__":
    main(make_video=True, show_video=False)
