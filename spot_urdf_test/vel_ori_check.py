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
import cv2



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
            "sensor_subtype": habitat_sim.SensorSubType.ORTHOGRAPHIC,
        },
        "depth_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": camera_resolution,
            "position": [0,0,0.0],#[0.0, 0.6, 0.0],
            "orientation": [0.0, 0.0, 0.0],
            "sensor_subtype": habitat_sim.SensorSubType.ORTHOGRAPHIC,
        },
        "rgba_camera_3rdperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [-2.0,2.0,0.0],#[0.0, 1.0, 0.3],
            "orientation": [0.0,0.0,0.0],#[-45, 0.0, 0.0],
            "sensor_subtype": habitat_sim.SensorSubType.ORTHOGRAPHIC,
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.CameraSensorSpec() #habitat_sim.SensorSpec()
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

def make_video_cv2(observations, ds=1, output_path = None, fps=60, pov="rgba_camera_3rdperson", text=None):
    if output_path is None:
        return False

    shp = observations[0][pov].shape
    
    videodims = (shp[1]//ds, shp[0]//ds)
    
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    vid_name = output_path + ".mp4"
    rate = fps // 30
    observations = observations[1::rate]
    if text is not None:
        text= text[1::rate]
    video = cv2.VideoWriter(vid_name, fourcc, 30, videodims)
    print('Formatting Video')
    for count, ob in enumerate(observations):
        if 'depth' in pov:
            
            ob[pov] = ob[pov][:,:,np.newaxis] / 10 * 255
            bgr_im_3rd_person = ob[pov] * np.ones((shp[0],shp[1], 3))

        else:
            bgr_im_3rd_person = ob[pov][...,0:3]
        
        frame =  cv2.cvtColor(np.uint8(bgr_im_3rd_person),cv2.COLOR_RGB2BGR)
        if text is not None:
            
            for i, line in enumerate(text[count]):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, line, (20, 100 + i*30), font, 0.5, (0, 0, 0), 2)
          
        video.write(frame)
    video.release()


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
            data_path, "URDF_demo_assets/spot_bd/model.urdf"
        ),
        "spot_alex": os.path.join(
            data_path, "URDF_demo_assets/spot_alex/habitat_spot_urdf/urdf/spot.urdf"
        ),
        "spot_akshara": os.path.join(
            data_path, "URDF_demo_assets/spot_akshara/spot_new.urdf"
        ),
        "spot_hybrid": os.path.join(
            data_path, "URDF_demo_assets/spot_hybrid_urdf/habitat_spot_urdf/urdf/spot.urdf"
        ),
    }

    # [basics]


    # load a URDF file
    robot_file_name = "spot_hybrid"
    robot_file = urdf_files[robot_file_name]
    robot_id = sim.add_articulated_object_from_urdf(robot_file, fixed_base=False)
    turn_controller = True

    local_base_pos = np.array([0.0,0.0,0.0])
    inverse = mn.Matrix4.rotation(mn.Rad(0.0), mn.Vector3(1, 0, 0).normalized())
    inverse.translation = agent_transform.transform_point(local_base_pos)
    
    existing_joint_motors = sim.get_existing_joint_motors(robot_id)
    
    agent_config = habitat_sim.AgentConfiguration()
    scene_graph = habitat_sim.SceneGraph()
    agent = habitat_sim.Agent(scene_graph.get_root_node().create_child(), agent_config)
    
    ctrl_freq = 120
    spot = Spot({}, urdf_file=urdf_files[robot_file_name], sim=sim, agent=agent, robot_id=robot_id, dt=1/ctrl_freq, inverse_transform=inverse)
    spot.robot_specific_reset()
    
    time_per_step = 72

    action_limit = np.zeros((12, 2))
    action_limit[:, 0] = np.zeros(12) + np.pi / 2
    action_limit[:, 1] = np.zeros(12) - np.pi / 2


    if turn_controller:
        raibert_controller = Raibert_controller_turn(control_frequency=ctrl_freq, num_timestep_per_HL_action=time_per_step, action_limit=action_limit, robot="Spot")
    else:
        raibert_controller = Raibert_controller(control_frequency=ctrl_freq, num_timestep_per_HL_action=time_per_step, action_limit=action_limit, robot="Spot")
    


    lin = np.array([0.5, 0]) 
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

    text = []
    
    for i in range(30):

        local_base_pos = np.array([0.0,0.0,0.0])
        base_transform = mn.Matrix4.rotation(mn.Rad(0.0), mn.Vector3(1, 0, 0).normalized())
        base_transform.translation = agent_transform.transform_point(local_base_pos)
        print(base_transform)
        sim.set_articulated_object_root_state(robot_id, base_transform)
        state = spot.calc_state(prev_state=state)



if __name__ == "__main__":
    main(make_video=True, show_video=False)
