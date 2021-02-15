from gibson2.core.physics.robot_locomotors import Locobot, Turtlebot, JR2_Kinova, Fetch
from gibson2.core.physics.a1_environment import A1, AlienGo, Laikago, Spot, SpotTurn, A1Turn
# from gibson2.core.physics.daisy_environment import DaisyTurn as Daisy
from gibson2.utils.utils import parse_config
#import daisy_hardware.motion_library as motion_library
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image
from iGibson.examples.demo.raibert_walking_controller import Raibert_controller_turn as Raibert_controller


def main():
    hz = 240
    scene = p.connect(p.DIRECT)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./hz)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    config = parse_config('../configs/locobot_p2p_nav.yaml')

    spot = SpotTurn(config)
    
    time_per_step = 72
    raibert_controller = Raibert_controller(control_frequency=240, num_timestep_per_HL_action=time_per_step, robot="SpotTurn")
    target_speed = np.array([0.2, 0.0])

    spot.load()
    spot.set_position([0, 0, 0.65])
    spot.set_orientation([0, 0, 0, 1.0])
    spot.robot_specific_reset()
    # init_state = motion_library.exp_standing(spot)
    state = spot.calc_state()
    # spot.reset(state['j_pos'])
    init_state = state
    raibert_controller.set_init_state(init_state)

    num_steps = 5
    print("base pos ", state['base_pos_x'], " ", state['base_pos_y'], state['base_pos_z'])
    print(spot.control)
    # last_ori = state['base_ori_euler'][2]
    
    for i in range(num_steps):
        lin = 0.35 # np.random.uniform(-0.75, 0.75, 1)
        ang = 0.15 #np.random.uniform(-0.5, 0.5, 1)
        hl_action = np.append(lin, ang)
        target_speed = np.array([hl_action[0], 0])
        target_ang_vel = hl_action[1]
        latent_action = raibert_controller.plan_latent_action(state, target_speed, target_ang_vel=target_ang_vel)
        raibert_controller.update_latent_action(state, latent_action)
        
        for j in range(time_per_step):
            # action = expert_control(phase=i, offset=1, const=constants)+action_init
            # action = np.random.rand(12)
            action = raibert_controller.get_action(state, j+1)
            spot.step(action)
            state = spot.calc_state()
            print('Action')
            print( action)
            print('State')
            print(state)
            # time.sleep(1/hz)

        if i == 2:
            im = get_camera_image(scene).astype(np.uint8)
            print(im.shape)
            print(im)
            im = Image.fromarray(im)
            im.save("test.jpeg")
        print("commanded ", hl_action, "local velocity ", state['base_velocity'][:2])
        vel = np.linalg.norm(state['base_velocity'])
        # if vel < 0.01:
            # import pdb; pdb.set_trace()
        # print("local yaw rate ", state['base_ang_vel'][2])
        # print("delta ori ", state['base_ori_euler'][2] - last_ori)
        # last_ori = state['base_ori_euler'][2]
    '''
    p.disconnect()


if __name__ == '__main__':
    main()

