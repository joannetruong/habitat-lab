'''
Script to automate stuff.
Makes a new directory, and stores the two yaml files that generate the config.
Replaces the yaml file content with the location of the new directory.
'''

import os
import argparse
import shutil
import subprocess
import ast

parser = argparse.ArgumentParser()
parser.add_argument('experiment_name')

# Training
parser.add_argument('-sd','--seed', type=int, default=100)
parser.add_argument('-u', '--user', type=str, default='max')
parser.add_argument('-r','--robots', nargs='+', required=True)
parser.add_argument('-c','--control-type', type=str, required=True)
parser.add_argument('-vx','--lin_vel_ranges', nargs='+', required=True)
parser.add_argument('-vy','--hor_vel_ranges', nargs='+', required=True)
parser.add_argument('-vt','--ang_vel_ranges', nargs='+', required=True)
parser.add_argument('-p','--partition', type=str, default='long')
parser.add_argument('-ds','--dataset', type=str, default='gibson')
parser.add_argument('-z','--use_z', default=False, action='store_true')
# Evaluation
parser.add_argument('-e','--eval', default=False, action='store_true')
parser.add_argument('-cpt','--ckpt', type=int, default=-1)
parser.add_argument('-v','--video', default=False, action='store_true')

parser.add_argument('-d','--debug', default=False, action='store_true')
args = parser.parse_args()

if args.user == 'joanne':
    HABITAT_LAB = "/coc/testnvme/jtruong33/habitat_spot/habitat-lab"
    RESULTS = "/coc/pskynet3/jtruong33/develop/flash_results/dan_res"
    SLURM_TEMPLATE = "/coc/testnvme/jtruong33/habitat_spot/habitat-lab/slurm_job_template.sh"
    EVAL_SLURM_TEMPLATE = "/coc/testnvme/jtruong33/habitat_spot/habitat-lab/eval_slurm_template.sh"
elif args.user == 'max':
    HABITAT_LAB = "/nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-lab"
    RESULTS = "/srv/share3/mrudolph8/develop/habitat_spot_results/dan_kinematic"
    SLURM_TEMPLATE = "/nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-lab/slurm_job_template_max.sh"
    EVAL_SLURM_TEMPLATE = "/nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-lab/eval_slurm_template_max.sh"


if args.dataset == 'hm3d':
    EXP_YAML  = "habitat_baselines/config/pointnav/ddppo_pointnav_quadruped_train.yaml" #"habitat_baselines/config/pointnav/ddppo_pointnav_quadruped_train_hm3d_gibson.yaml"
    EVAL_EXP_YAML  = "habitat_baselines/config/pointnav/ddppo_pointnav_quadruped_eval.yaml" #"habitat_baselines/config/pointnav/ddppo_pointnav_quadruped_eval_hm3d_gibson.yaml"
    TASK_YAML = "configs/tasks/pointnav_quadruped_train_hm3d_gibson.yaml"
    EVAL_YAML = "configs/tasks/pointnav_quadruped_eval_hm3d_gibson.yaml"
else:
    EXP_YAML  = "habitat_baselines/config/pointnav/ddppo_pointnav_quadruped_train.yaml"
    EVAL_EXP_YAML  = "habitat_baselines/config/pointnav/ddppo_pointnav_quadruped_eval.yaml"
    TASK_YAML = "configs/tasks/pointnav_quadruped_train.yaml"
    EVAL_YAML = "configs/tasks/pointnav_quadruped_eval.yaml"

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

robot_urdfs_dict = {'A1': "/coc/testnvme/jtruong33/data/URDF_demo_assets/a1/a1.urdf",
                    'AlienGo': "/coc/testnvme/jtruong33/data/URDF_demo_assets/aliengo/urdf/aliengo.urdf",
                    'Daisy': "/coc/testnvme/jtruong33/data/URDF_demo_assets/daisy/daisy_advanced_akshara.urdf",
                    'Spot': "/coc/testnvme/jtruong33/data/URDF_demo_assets/spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid.urdf",
                    'Locobot': "/coc/testnvme/jtruong33/data/URDF_demo_assets/locobot/urdf/locobot_description2.urdf"
                    }

robot_goal_dict = {'A1': 0.24,
                   'AlienGo': 0.3235,
                   'Locobot': 0.20,
                   'Spot': 0.425
                  }
robots = args.robots
num_robots = len(robots)
robots_urdfs = [robot_urdfs_dict[robot] for robot in robots]
# print('robots: ', robots)
# print('robot_goal_dict: ', robot_goal_dict)
if num_robots > 1:
    robot_goal = min([robot_goal_dict[robot] for robot in robots])
else: 
    robot_goal = robot_goal_dict[robots[0]]
robots_underscore = '_'.join(robots)
if args.control_type == 'dynamic':
    robots_underscore += '_dynamic'

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
                          
    # robots_heights = [robot_heights_dict[robot] for robot in robots]

    for idx, i in enumerate(task_yaml_data):
        if i.startswith('  TYPE: Nav-v0'):
            task_yaml_data[idx] = "  TYPE: MultiNav-v0"
        elif i.startswith('  ROBOTS:'):
            task_yaml_data[idx] = "  ROBOTS: {}".format(robots)
        elif i.startswith('  ROBOT_URDFS:'):
            task_yaml_data[idx] = "  ROBOT_URDFS: {}".format(robots_urdfs)
        elif i.startswith('  USE_Z:'):
            task_yaml_data[idx] = "  USE_Z: {}".format(args.use_z)
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
        elif i.startswith('  SUCCESS_DISTANCE:'):
            task_yaml_data[idx] = "  SUCCESS_DISTANCE: {}".format(robot_goal)
        elif i.startswith('    SUCCESS_DISTANCE:'):
            task_yaml_data[idx] = "    SUCCESS_DISTANCE: {}".format(robot_goal)
        # elif i.startswith('    POSITION:'):
            # task_yaml_data[idx] = "    POSITION: {}".format(robots_heights[0])

        # elif i.startswith('    POSITION: '):
            # task_yaml_data[idx] = "    POSITION: [0.0, {}, -0.1778]".format(robot_camera_pos)
        elif i.startswith('SEED:'):
            task_yaml_data[idx] = "SEED: {}".format(args.seed)

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

    for idx, i in enumerate(eval_yaml_data):
        if i.startswith('  TYPE: Nav-v0'):
            eval_yaml_data[idx] = "  TYPE: MultiNav-v0"
        elif i.startswith('  ROBOTS:'):
            eval_yaml_data[idx] = "  ROBOTS: {}".format(robots)
        elif i.startswith('  ROBOT_URDFS:'):
            eval_yaml_data[idx] = "  ROBOT_URDFS: {}".format(robots_urdfs)
        elif i.startswith('  USE_Z:'):
            eval_yaml_data[idx] = "  USE_Z: {}".format(args.use_z)
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
        elif i.startswith('  SUCCESS_DISTANCE:'):
            eval_yaml_data[idx] = "  SUCCESS_DISTANCE: {}".format(robot_goal)
        elif i.startswith('    SUCCESS_DISTANCE:'):
            eval_yaml_data[idx] = "    SUCCESS_DISTANCE: {}".format(robot_goal)
        elif i.startswith('SEED:'):
            eval_yaml_data[idx] = "SEED: {}".format(args.seed)

    new_eval_yaml_path = new_eval_yaml_path[:-len('.yaml')]+'_'+robots_underscore+'.yaml'
    with open(new_eval_yaml_path,'w') as f:
        f.write('\n'.join(eval_yaml_data))
    print("Created "+new_eval_yaml_path)

    # Edit the stored experiment yaml file
    with open(eval_exp_yaml_path) as f:
        exp_yaml_data = f.read().splitlines()

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
                if args.ckpt:
                    exp_yaml_data[idx] = "VIDEO_DIR:          '{}_{}_ckpt={}'".format(os.path.join(dst_dir,'video_dir'),robots_underscore, args.ckpt)
                else:
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

    if args.video:
        new_exp_eval_yaml_path = new_exp_yaml_path[:-len('.yaml')]+'_eval_'+robots_underscore+'_video.yaml'
    else:
        new_exp_eval_yaml_path = new_exp_yaml_path[:-len('.yaml')]+'_eval_'+robots_underscore+'.yaml'
    with open(new_exp_eval_yaml_path,'w') as f:
        f.write('\n'.join(exp_yaml_data))

    if args.video:
        robots_underscore += '_video'
    # Create slurm job
    with open(EVAL_SLURM_TEMPLATE) as f:
        slurm_data = f.read()
        slurm_data = slurm_data.replace('TEMPLATE', f'eval_{experiment_name}_{robots_underscore}')
        slurm_data = slurm_data.replace('HABITAT_REPO_PATH', HABITAT_LAB)
        slurm_data = slurm_data.replace('CONFIG_YAML', new_exp_eval_yaml_path)
        slurm_data = slurm_data.replace('PARTITION', args.partition)
        if args.partition == 'overcap':
            slurm_data = slurm_data.replace('# ACCOUNT', '#SBATCH --account overcap')

    slurm_path = os.path.join(dst_dir, experiment_name+'_eval_'+robots_underscore+'.sh')
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