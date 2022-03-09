'''
Script to automate stuff.
Makes a new directory, and stores the two yaml files that generate the config.
Replaces the yaml file content with the location of the new directory.
'''

HABITAT_LAB = "/nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-lab"
RESULTS = "/srv/share3/mrudolph8/develop/habitat_spot_results/dan_kinematic"
EXP_YAML  = "habitat_baselines/config/pointnav/ddppo_pointnav_quadruped.yaml"
EVAL_EXP_YAML  = "habitat_baselines/config/pointnav/ddppo_pointnav_quadruped_eval.yaml"
TASK_YAML = "configs/tasks/pointnav_quadruped.yaml"
EVAL_YAML = "configs/tasks/pointnav_quadruped_eval.yaml"
SLURM_TEMPLATE = "/nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-lab/slurm_job_template_max.sh"
EVAL_SLURM_TEMPLATE = "/nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-lab/eval_slurm_template_max.sh"

import os
import argparse
import shutil
import subprocess
import ast

parser = argparse.ArgumentParser()
parser.add_argument('experiment_name')

# Training
parser.add_argument('-sd','--seed', type=int, default=100)
parser.add_argument('-r','--robots', nargs='+', required=True)
parser.add_argument('-c','--control-type', type=str, required=True)
parser.add_argument('-vx','--lin_vel_ranges', nargs='+', required=True)
parser.add_argument('-vy','--hor_vel_ranges', nargs='+', required=True)
parser.add_argument('-vt','--ang_vel_ranges', nargs='+', required=True)
parser.add_argument('-p','--partition', type=str, default='long')
# Evaluation
parser.add_argument('-e','--eval', default=False, action='store_true')
parser.add_argument('-cpt','--ckpt', type=int, default=-1)
parser.add_argument('-v','--video', default=False, action='store_true')
parser.add_argument('-ms', '--max_steps', type=int, default=150)

parser.add_argument('-d','--debug', default=False, action='store_true')
args = parser.parse_args()
experiment_name = args.experiment_name


dst_dir = os.path.join(RESULTS, experiment_name)
exp_yaml_path  = os.path.join(HABITAT_LAB, EXP_YAML)
eval_exp_yaml_path  = os.path.join(HABITAT_LAB, EVAL_EXP_YAML)
task_yaml_path = os.path.join(HABITAT_LAB, TASK_YAML)
eval_yaml_path = os.path.join(HABITAT_LAB, EVAL_YAML)
new_task_yaml_path = os.path.join(dst_dir, os.path.basename(task_yaml_path))
new_eval_yaml_path = os.path.join(dst_dir, os.path.basename(eval_yaml_path))
new_exp_yaml_path  = os.path.join(dst_dir, os.path.basename(exp_yaml_path))
new_eval_exp_yaml_path  = os.path.join(dst_dir, os.path.basename(eval_exp_yaml_path))

# Training
if not args.eval:

    # Create directory
    if os.path.isdir(dst_dir):
        response = input("'{}' already exists. Delete or abort? [d/A]: ".format(dst_dir))
        if response == 'd':
            print('Deleting {}'.format(dst_dir))
            shutil.rmtree(dst_dir)
        else:
            print('Aborting.')
            exit()
    os.mkdir(dst_dir)
    print("Created "+dst_dir)

    # Create task yaml file, using file within Habitat Lab repo as a template
    with open(task_yaml_path) as f:
        task_yaml_data = f.read().splitlines()

    robot_urdfs_dict = {'A1': "/srv/share3/mrudolph8/data/data/URDF_demo_assets/a1/a1.urdf",
                        'AlienGo': "/coc/testnvme/jtruong33/data/URDF_demo_assets/aliengo/urdf/aliengo.urdf",
                        'Daisy': "/coc/testnvme/jtruong33/data/URDF_demo_assets/daisy/daisy_advanced_akshara.urdf",
                        'Spot': "/coc/testnvme/jtruong33/data/URDF_demo_assets/spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid.urdf",
                        'Locobot': "/srv/share3/mrudolph8/data/data/URDF_demo_assets/locobot/urdf/locobot_description2.urdf"
                        }

    robot_heights_dict = {'A1': [0.0, 0.18, -0.24],
                          'AlienGo': [0.0, 0.25, -0.3235],
                          'Daisy': [0.0, 0.42, -0.27],
                          'Spot': [0.0, 0.425, -0.425],
                          'Locobot': [0.0, 0.31, -0.55]
                         }
    robot_length_dict = {'A1': 0.2,
                         'AlienGo': 0.25,
                         'Spot': 0.44}

                          
    robots = args.robots
    num_robots = len(robots)
    robots_urdfs = [robot_urdfs_dict[robot] for robot in robots]
    robots_heights = [robot_heights_dict[robot] for robot in robots]
    robot_lengh = [robot_length_dict[robot] for robot in robots]
    for idx, i in enumerate(task_yaml_data):
        if i.startswith('  TYPE: Nav-v0'):
            task_yaml_data[idx] = "  TYPE: MultiNav-v0"
        elif i.startswith('  ROBOTS:'):
            task_yaml_data[idx] = "  ROBOTS: {}".format(robots)
        elif i.startswith('  ROBOT_URDFS:'):
            task_yaml_data[idx] = "  ROBOT_URDFS: {}".format(robots_urdfs)
        elif i.startswith('  POSSIBLE_ACTIONS:'):
            if args.control_type == 'dynamic':
                control_type = "DYNAMIC_VELOCITY_CONTROL"
            else:
                control_type = "VELOCITY_CONTROL"
            task_yaml_data[idx] = "  POSSIBLE_ACTIONS: [\"{}\"]".format(control_type)
        elif i.startswith('    VELOCITY_CONTROL:'):
            if args.control_type == 'dynamic':
                task_yaml_data[idx] = "    DYNAMIC_VELOCITY_CONTROL:"
        elif i.startswith('    DYNAMIC_VELOCITY_CONTROL:'):
            if not args.control_type == 'dynamic':
                task_yaml_data[idx] = "    VELOCITY_CONTROL:"
        elif i.startswith('      LIN_VEL_RANGES:'):
            lin_vel_ranges = [ast.literal_eval(n) for n in args.lin_vel_ranges]
            task_yaml_data[idx] = "      LIN_VEL_RANGES: {}".format(lin_vel_ranges)
        elif i.startswith('      HOR_VEL_RANGES:'):
            hor_vel_ranges = [ast.literal_eval(n) for n in args.hor_vel_ranges]
            task_yaml_data[idx] = "      HOR_VEL_RANGES: {}".format(hor_vel_ranges)
        elif i.startswith('      ANG_VEL_RANGES:'):
            ang_vel_ranges = [ast.literal_eval(n) for n in args.ang_vel_ranges]
            task_yaml_data[idx] = "      ANG_VEL_RANGES: {}".format(ang_vel_ranges)
        elif i.startswith('    POSITION:'):
            task_yaml_data[idx] = "    POSITION: {}".format(robots_heights[0])

        # elif i.startswith('    POSITION: '):
            # task_yaml_data[idx] = "    POSITION: [0.0, {}, -0.1778]".format(robot_camera_pos)
        elif i.startswith('SEED:'):
            task_yaml_data[idx] = "SEED: {}".format(args.seed)
        elif i.startswith('  MAX_EPISODE_STEPS'):
            task_yaml_data[idx] = "  MAX_EPISODE_STEPS: {}".format(args.max_steps)
        elif i.startswith('    SUCCESS_DISTANCE'):
            task_yaml_data[idx] = "    SUCCESS_DISTANCE: {}".format(robot_lengh[0])
        elif i.startswith('  SUCCESS_DISTANCE:'):
            task_yaml_data[idx] = "  SUCCESS_DISTANCE: {}".format(robot_lengh[0])

    with open(new_task_yaml_path,'w') as f:
        f.write('\n'.join(task_yaml_data))
    print("Created "+new_task_yaml_path)

    # Create experiment yaml file, using file within Habitat Lab repo as a template
    with open(exp_yaml_path) as f:
        exp_yaml_data = f.read().splitlines()

    for idx, i in enumerate(exp_yaml_data):
        if i.startswith('BASE_TASK_CONFIG_PATH:'):
            exp_yaml_data[idx] = "BASE_TASK_CONFIG_PATH: '{}'".format(new_task_yaml_path)
        elif i.startswith('TENSORBOARD_DIR:'):
            exp_yaml_data[idx] = "TENSORBOARD_DIR:    '{}'".format(os.path.join(dst_dir,'tb'))
        elif i.startswith('VIDEO_DIR:'):
            exp_yaml_data[idx] = "VIDEO_DIR:          '{}'".format(os.path.join(dst_dir,'video_dir'))
        elif i.startswith('EVAL_CKPT_PATH_DIR:'):
            exp_yaml_data[idx] = "EVAL_CKPT_PATH_DIR: '{}'".format(os.path.join(dst_dir,'checkpoints'))
        elif i.startswith('CHECKPOINT_FOLDER:'):
            exp_yaml_data[idx] = "CHECKPOINT_FOLDER:  '{}'".format(os.path.join(dst_dir,'checkpoints'))
        elif i.startswith('TXT_DIR:'):
            exp_yaml_data[idx] = "TXT_DIR:            '{}'".format(os.path.join(dst_dir,'txts'))

    with open(new_exp_yaml_path,'w') as f:
        f.write('\n'.join(exp_yaml_data))
    print("Created "+new_exp_yaml_path)

    # Create slurm job
    with open(SLURM_TEMPLATE) as f:
        slurm_data = f.read()
        slurm_data = slurm_data.replace('TEMPLATE', experiment_name)
        slurm_data = slurm_data.replace('HABITAT_REPO_PATH', HABITAT_LAB)
        slurm_data = slurm_data.replace('CONFIG_YAML', new_exp_yaml_path)
        slurm_data = slurm_data.replace('PARTITION', args.partition)
    if args.debug:
        slurm_data = slurm_data.replace('GPUS', '1')
    else:
        slurm_data = slurm_data.replace('GPUS', '8')
    slurm_path = os.path.join(dst_dir, experiment_name+'.sh')
    with open(slurm_path,'w') as f:
        f.write(slurm_data)
    print("Generated slurm job: "+slurm_path)

    # Submit slurm job
    cmd = 'sbatch '+slurm_path
    subprocess.check_call(cmd.split(), cwd=dst_dir)

    print('\nSee output with:\ntail -F {}'.format(os.path.join(dst_dir, experiment_name+'.err')))

# Evaluation
else:

    # Make sure folder exists
    assert os.path.isdir(dst_dir), "{} directory does not exist".format(dst_dir)

        # Create task yaml file, using file within Habitat Lab repo as a template
    with open(eval_yaml_path) as f:
        eval_yaml_data = f.read().splitlines()

    robot_urdfs_dict = {'A1': "/srv/share3/mrudolph8/data/data/URDF_demo_assets/a1/a1.urdf",
                        'AlienGo': "/coc/testnvme/jtruong33/data/URDF_demo_assets/aliengo/urdf/aliengo.urdf",
                        'Daisy': "/coc/testnvme/jtruong33/data/URDF_demo_assets/daisy/daisy_advanced_akshara.urdf",
                        'Spot': "/coc/testnvme/jtruong33/data/URDF_demo_assets/spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid.urdf",
                        'Locobot': "/srv/share3/mrudolph8/data/data/URDF_demo_assets/locobot/urdf/locobot_description2.urdf"
                        }

    robot_heights_dict = {'A1': [0.0, 0.18, -0.24],
                          'AlienGo': [0.0, 0.25, -0.3235],
                          'Daisy': [0.0, 0.42, -0.27],
                          'Spot': [0.0, 0.425, -0.425],
                          'Locobot': [0.0, 0.31, -0.7]
                         }


    robots = args.robots
    num_robots = len(robots)
    robots_urdfs = [robot_urdfs_dict[robot] for robot in robots]
    robots_heights = [robot_heights_dict[robot] for robot in robots]
    camera_count = 0
    for idx, i in enumerate(eval_yaml_data):
        if i.startswith('  TYPE: Nav-v0'):
            eval_yaml_data[idx] = "  TYPE: MultiNav-v0"
        elif i.startswith('  ROBOTS:'):
            eval_yaml_data[idx] = "  ROBOTS: {}".format(robots)
        elif i.startswith('  ROBOT_URDFS:'):
            eval_yaml_data[idx] = "  ROBOT_URDFS: {}".format(robots_urdfs)
        elif i.startswith('  POSSIBLE_ACTIONS:'):
            if args.control_type == 'dynamic':
                control_type = "DYNAMIC_VELOCITY_CONTROL"
            else:
                control_type = "VELOCITY_CONTROL"
            eval_yaml_data[idx] = "  POSSIBLE_ACTIONS: [\"{}\"]".format(control_type)
        elif i.startswith('    VELOCITY_CONTROL:'):
            if args.control_type == 'dynamic':
                eval_yaml_data[idx] = "    DYNAMIC_VELOCITY_CONTROL:"
        elif i.startswith('    DYNAMIC_VELOCITY_CONTROL:'):
            if not args.control_type == 'dynamic':
                eval_yaml_data[idx] = "    VELOCITY_CONTROL:"
        elif i.startswith('      LIN_VEL_RANGES:'):
            lin_vel_ranges = [ast.literal_eval(n) for n in args.lin_vel_ranges]
            eval_yaml_data[idx] = "      LIN_VEL_RANGES: {}".format(lin_vel_ranges)
        elif i.startswith('      HOR_VEL_RANGES:'):
            hor_vel_ranges = [ast.literal_eval(n) for n in args.hor_vel_ranges]
            eval_yaml_data[idx] = "      HOR_VEL_RANGES: {}".format(hor_vel_ranges)
        elif i.startswith('      ANG_VEL_RANGES:'):
            ang_vel_ranges = [ast.literal_eval(n) for n in args.ang_vel_ranges]
            eval_yaml_data[idx] = "      ANG_VEL_RANGES: {}".format(ang_vel_ranges)
        elif i.startswith('      TIME_STEP:'):
            eval_yaml_data[idx] = "      TIME_STEP: 100.0"
        elif i.startswith('SEED:'):
            eval_yaml_data[idx] = "SEED: {}".format(args.seed)
        elif i.startswith('    POSITION:'):   
            if camera_count == 0:
                eval_yaml_data[idx] = "    POSITION: {}".format([0.0, 1.1, 1.0])
                camera_count += 1
            else:
                eval_yaml_data[idx] = "    POSITION: {}".format(robots_heights[0])
        elif i.startswith('  MAX_EPISODE_STEPS'):
            eval_yaml_data[idx] = "  MAX_EPISODE_STEPS: {}".format(args.max_steps)

    with open(new_eval_yaml_path,'w') as f:
        f.write('\n'.join(eval_yaml_data))
    print("Created "+new_eval_yaml_path)

    # Edit the stored experiment yaml file
    with open(eval_exp_yaml_path) as f:
        exp_yaml_data = f.read().splitlines()

    robots_underscore = '_'.join(robots)
    for idx, i in enumerate(exp_yaml_data):
        if i.startswith('BASE_TASK_CONFIG_PATH:'):
            exp_yaml_data[idx] = "BASE_TASK_CONFIG_PATH: '{}'".format(new_eval_yaml_path)
        elif i.startswith('TENSORBOARD_DIR:'):
            tb_dir = 'tb_eval_{}'.format(robots_underscore) if args.ckpt == -1 else 'tb_eval_{}_ckpt={}'.format(robots_underscore, args.ckpt)
            if args.video:
                tb_dir = 'tb_eval_{}_video'.format(robots_underscore) if args.ckpt == -1 else 'tb_eval_{}_ckpt={}_video'.format(robots_underscore, args.ckpt)
            tb_dir = os.path.join(dst_dir, tb_dir)
            exp_yaml_data[idx] = "TENSORBOARD_DIR:    '{}'".format(tb_dir)
        elif i.startswith('NUM_PROCESSES:'):
            exp_yaml_data[idx] = "NUM_PROCESSES: 13"
        elif i.startswith('CHECKPOINT_FOLDER:'):
            ckpt_dir = os.path.join(dst_dir,'checkpoints')
        elif i.startswith('EVAL_CKPT_PATH_DIR:'):
            if args.ckpt == -1:
                exp_yaml_data[idx] = "EVAL_CKPT_PATH_DIR: '{}'".format(os.path.join(dst_dir,'checkpoints'))
            else:
                exp_yaml_data[idx] = "EVAL_CKPT_PATH_DIR: '{}/ckpt.{}.pth'".format(os.path.join(dst_dir,'checkpoints'), args.ckpt)
        elif i.startswith('TXT_DIR:'):
            exp_yaml_data[idx] = "TXT_DIR:            '{}'".format(os.path.join(dst_dir,'txts'))
        elif i.startswith('VIDEO_OPTION:'):
            if args.video:
                exp_yaml_data[idx] = "VIDEO_OPTION: ['disk']"
            else:
                exp_yaml_data[idx] = "VIDEO_OPTION: []"
        elif i.startswith('SENSORS:'):
            if args.video:
                exp_yaml_data[idx] = "SENSORS: ['RGB_SENSOR','DEPTH_SENSOR']"
        elif i.startswith('VIDEO_DIR:'):
            if args.video:
                exp_yaml_data[idx] = "VIDEO_DIR:          '{}_{}'".format(os.path.join(dst_dir,'video_dir'),robots_underscore)
        elif args.ckpt != -1:
            if i.startswith('EVAL_CKPT_PATH_DIR:'):
                new_ckpt_dir = os.path.join(dst_dir, f'checkpoints/ckpt.{args.ckpt}.pth')
                exp_yaml_data[idx] = f"EVAL_CKPT_PATH_DIR: '{new_ckpt_dir}'"
            elif i.startswith('TXT_DIR:'):
                exp_yaml_data[idx] = "TXT_DIR:            '{}'".format(os.path.join(dst_dir,'txts_{}'.format(args.ckpt)))

    if os.path.isdir(tb_dir):
        response = input('{} directory already exists. Delete, continue, or abort? [d/c/A]: '.format(tb_dir))
        if response == 'd':
            print('Deleting {}'.format(tb_dir))
            shutil.rmtree(tb_dir)
        elif response == 'c':
            print('Continuing.')
        else:
            print('Aborting.')
            exit()

    if args.ckpt != -1:
        ckpt_file = os.path.join(ckpt_dir, 'ckpt.{}.pth'.format(args.ckpt))
        assert os.path.isfile(ckpt_file), '{} does not exist'.format(ckpt_file)

    new_exp_eval_yaml_path = new_exp_yaml_path[:-len('.yaml')]+'_eval.yaml'
    with open(new_exp_eval_yaml_path,'w') as f:
        f.write('\n'.join(exp_yaml_data))

    # Create slurm job
    with open(EVAL_SLURM_TEMPLATE) as f:
        slurm_data = f.read()
        slurm_data = slurm_data.replace('TEMPLATE', f'eval_{experiment_name}_{robots_underscore}')
        slurm_data = slurm_data.replace('HABITAT_REPO_PATH', HABITAT_LAB)
        slurm_data = slurm_data.replace('CONFIG_YAML', new_exp_eval_yaml_path)
        slurm_data = slurm_data.replace('PARTITION', args.partition)
        if args.partition == 'overcap':
            slurm_data = slurm_data.replace('# ACCOUNT', '#SBATCH --account overcap')

    slurm_path = os.path.join(dst_dir, experiment_name+'_eval.sh')
    with open(slurm_path,'w') as f:
        f.write(slurm_data)
    print("Generated slurm job: "+slurm_path)

    # Submit slurm job
    cmd = 'sbatch '+slurm_path
    subprocess.check_call(cmd.split(), cwd=dst_dir)
    print('\nSee output with:\ntail -F {}'.format(os.path.join(dst_dir, 'eval_'+experiment_name+'_'+robots_underscore+'.err')))

    # cmd = 'tmuxs eval_{}\n'.format(experiment_name)
    # cmd += 'srun --gres gpu:1 --nodes 1 --partition long --job-name eval_{} --exclude calculon --pty bash \n'.format(experiment_name)
    # cmd += 'aconda aug26n\n'
    # cmd += 'cd {}\n'.format(HABITAT_LAB)
    # cmd += 'python -u -m habitat_baselines.run --exp-config {} --run-type eval\n '.format(new_exp_eval_yaml_path)
    # print('\nCopy-paste and run the following:\n{}'.format(cmd))
    # subprocess.check_call(cmd.split(), cwd=HABITAT_LAB)