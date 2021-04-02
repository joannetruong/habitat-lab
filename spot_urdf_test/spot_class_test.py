# [setup]
import os

import magnum as mn
import numpy as np
from datetime import datetime
import habitat_sim

# import habitat_sim.utils.common as ut
import habitat_sim.utils.viz_utils as vut
from utilities.spot_env import Spot
from utilities.raibert_controller import Raibert_controller
from utilities.raibert_controller import Raibert_controller_turn



dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../habitat-sim/data")
output_path = os.path.join(dir_path, "URDF_robotics_tutorial_output/")


def remove_all_objects(sim):
    for ob_id in sim.get_existing_object_ids():
        sim.remove_object(ob_id)
    for ob_id in sim.get_existing_articulated_object_ids():
        sim.remove_articulated_object(ob_id)


def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = [-0.15, -0.7, 1.0]
    #agent_state.position = [-0.15, -0.7, 1.0]
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
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "depth_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": camera_resolution,
            "position": [0,0,0.0],#[0.0, 0.6, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "rgba_camera_3rdperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [-2.0,2.0,0.0],#[0.0, 1.0, 0.3],
            "orientation": [0.0,0.0,0.0],#[-45, 0.0, 0.0],
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


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())

    return observations


# [/setup]

# This is wrapped such that it can be added to a unit test
def main(make_video=True, show_video=True):
    if make_video and not os.path.exists(output_path):
        os.mkdir(output_path)

    # [initialize]
    # create the simulator
    cfg = make_configuration()
    sim = habitat_sim.Simulator(cfg)
    agent_transform = place_agent(sim)
    observations = []

    # [/initialize]

    urdf_files = {
        "aliengo": os.path.join(
            data_path, "URDF_demo_assets/aliengo/urdf/aliengo.urdf"
        ),
        "iiwa": os.path.join(
            data_path, "test_assets/urdf/kuka_iiwa/model_free_base.urdf"
        ),
        "locobot": os.path.join(
            data_path, "URDF_demo_assets/aliengo/urdf/aliengo.urdf"
        ),
        "locobot_light": os.path.join(
            data_path, "URDF_demo_assets/aliengo/urdf/aliengo.urdf"
        ),
        "spot": os.path.join(
            data_path, "URDF_demo_assets/spot/urdf/spot.urdf.xacro"
        ),
        "spot_bd": os.path.join(
            data_path, "URDF_demo_assets/spot_bd/model_massless.urdf"
        )
    }

    # [basics]


    # load a URDF file
    robot_file_name = "spot_bd"
    robot_file = urdf_files[robot_file_name]
    robot_id = sim.add_articulated_object_from_urdf(robot_file, False)
    turn_controller = False
    
    # place the robot root state relative to the agent
    #local_base_pos = np.array([-4, 2, -4.0])


    # local_base_pos = np.array([-2.0,1.2,-2.0])
    local_base_pos = np.array([-2.0,1.3,-2.0])
    agent_transform1 = sim.agents[0].scene_node.transformation_matrix()
    
    base_transform = mn.Matrix4.rotation(mn.Rad(-1.57), mn.Vector3(1, 0, 0).normalized())
    transform2 = mn.Matrix4.rotation(mn.Rad(-.01), mn.Vector3(0, 0, 1).normalized())
    base_transform = base_transform.__matmul__(transform2)
    inverse_transform = base_transform.inverted()
    base_transform.translation = agent_transform.transform_point(local_base_pos)
    
    
    
    sim.set_articulated_object_root_state(robot_id, base_transform)
    
    # # set a better initial joint state for the aliengo
    # if (robot_file == urdf_files["spot_bd"]) or (robot_file == urdf_files["aliengo"]):
    #     pose = sim.get_articulated_object_positions(robot_id)
    #     calfDofs = [2, 5, 8, 11]
    #     for dof in calfDofs:
    #         pose[dof] = -.8
    #         pose[dof - 1] = 0.35
    #         #pose[dof] = 0.1
    #         #pose[dof-1] = 0
    #         # also set a thigh
    #     sim.set_articulated_object_positions(robot_id, pose)
    
    
    existing_joint_motors = sim.get_existing_joint_motors(robot_id)
    
    agent_config = habitat_sim.AgentConfiguration()
    scene_graph = habitat_sim.SceneGraph()
    agent = habitat_sim.Agent(scene_graph.get_root_node().create_child(), agent_config)
    
    ctrl_freq = 240
    spot = Spot({}, urdf_file=urdf_files[robot_file_name], sim=sim, agent=agent, robot_id=robot_id, dt=1/ctrl_freq, inverse_transform=inverse_transform)
    spot.robot_specific_reset()
    
    time_per_step = 72

    action_limit = np.zeros((12, 2))
    action_limit[:, 0] = np.zeros(12) + np.pi / 2
    action_limit[:, 1] = np.zeros(12) - np.pi / 2


    if turn_controller:
        raibert_controller = Raibert_controller_turn(control_frequency=ctrl_freq, num_timestep_per_HL_action=time_per_step, action_limit=action_limit, robot="Spot")
    else:
        raibert_controller = Raibert_controller(control_frequency=ctrl_freq, num_timestep_per_HL_action=time_per_step, action_limit=action_limit, robot="Spot")
    


    lin = np.array([0, 0]) 
    ang = 0
    target_speed = lin
    target_ang_vel = ang
    state = spot.calc_state()

    init_state = state
    raibert_controller.set_init_state(init_state)


    
 
    if turn_controller:
        latent_action = raibert_controller.plan_latent_action(state, target_speed, target_ang_vel=target_ang_vel)
    else:
        latent_action = raibert_controller.plan_latent_action(state, target_speed, target_ori=0)

    
    for i in range(20):

        if turn_controller:
            latent_action = raibert_controller.plan_latent_action(state, target_speed, target_ang_vel=target_ang_vel)
        else:    
            latent_action = raibert_controller.plan_latent_action(state, target_speed, target_ori=0)

        raibert_controller.update_latent_action(state, latent_action)
        
        for j in range(time_per_step):
   
            action = raibert_controller.get_action(state, j+1)
            
            cur_obs = spot.step(action, dt=1/ctrl_freq, follow_robot=False)
            observations += cur_obs
            state = spot.calc_state(prev_state=state)
            
            
            



    

    
    if make_video:
        time_str = datetime.now().strftime("_%d%m%y_%H_%M")
        sensor_dims = (
            sim.get_agent(0).agent_config.sensor_specifications[0].resolution
        )
        overlay_dims = (int(sensor_dims[1] / 4), int(sensor_dims[0] / 4))
        overlay_settings = [
            {
                "obs": "rgba_camera_1stperson",
                "type": "color",
                "dims": overlay_dims,
                "pos": (10, 10),
                "border": 2,
            },
            {
                "obs": "depth_camera_1stperson",
                "type": "depth",
                "dims": overlay_dims,
                "pos": (10, 30 + overlay_dims[1]),
                "border": 2,
            },
        ]

        # vut.make_video(
        #     observations=observations,
        #     primary_obs="depth_camera_1stperson",
        #     primary_obs_type="depth",
        #     video_file=output_path + "depth_1st_" + time_str,
        #     fps=60,
        #     open_vid=show_video,
        #     overlay_settings=overlay_settings,
        #     depth_clip=10.0,
        # )
        
        vut.make_video(
            observations=observations,
            primary_obs="rgba_camera_3rdperson",
            primary_obs_type="color",
            video_file=output_path + "color_3rd_" + time_str,
            open_vid=show_video,
            fps=ctrl_freq,
            overlay_settings=overlay_settings,
            depth_clip=10.0,
        )
        # vut.make_video(
        #     observations=observations,
        #     primary_obs="rgba_camera_1stperson",
        #     primary_obs_type="color",
        #     video_file=output_path + "color_1st_" + time_str,
        #     fps=60,
        #     open_vid=show_video,
        #     overlay_settings=overlay_settings,
        #     depth_clip=10.0,
        # )
    # if make_video:
    #     vut.make_video(
    #         observations,
    #         "rgba_camera_3rdperson",
    #         "color",
    #         output_path + "URDF_basics" + datetime.now().strftime("%d%m%y_%H_%M"),
    #         open_vid=show_video,
    #         fps=ctrl_freq,
    #     )


if __name__ == "__main__":
    main(make_video=True, show_video=False)
