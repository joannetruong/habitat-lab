'''
Script to automate stuff.
Makes a new directory, and stores the two yaml files that generate the config.
Replaces the yaml file content with the location of the new directory.
'''

HABITAT_LAB = "/coc/testnvme/jtruong33/habitat_spot/habitat-lab"
RESULTS = "/coc/pskynet3/jtruong33/develop/flash_results/dan_res"
SLURM_TEMPLATE = "/coc/testnvme/jtruong33/habitat_spot/habitat-lab/slurm_job_template.sh"
EVAL_SLURM_TEMPLATE = "/coc/testnvme/jtruong33/habitat_spot/habitat-lab/eval_slurm_template.sh"

import os
import argparse
import shutil
import subprocess
import ast

parser = argparse.ArgumentParser()
parser.add_argument('experiment_name')

# Training
parser.add_argument('-u', '--user', type=str, default='joanne')
parser.add_argument('-sd','--seed', type=int, default=100)
parser.add_argument('-r','--robots', nargs='+', required=True)
parser.add_argument('-c','--control-type', type=str, required=True)
parser.add_argument('-mvx','--max_lin_vel', type=float, default=0.5, required=False)
parser.add_argument('-mvy','--max_hor_vel', type=float, default=0.5, required=False)
parser.add_argument('-mvt','--max_ang_vel', type=float, default=17.19, required=False)
parser.add_argument('-vx','--lin_vel_ranges', nargs='+', required=True)
parser.add_argument('-vy','--hor_vel_ranges', nargs='+', required=True)
parser.add_argument('-vt','--ang_vel_ranges', nargs='+', required=True)
parser.add_argument('-nd','--noisy_depth', default=False, action='store_true')
parser.add_argument('-p','--partition', type=str, default='long')
## options for dataset are hm3d_gibson, hm3d, gibson
parser.add_argument('-ds','--dataset', type=str, default='gibson')
parser.add_argument('-z','--use_z', default=False, action='store_true')
# Evaluation
parser.add_argument('-e','--eval', default=False, action='store_true')
parser.add_argument('-cpt','--ckpt', type=int, default=-1)
parser.add_argument('-v','--video', default=False, action='store_true')

parser.add_argument('-d','--debug', default=False, action='store_true')
args = parser.parse_args()

if args.user == 'max':
    HABITAT_LAB = "/nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-lab"
    RESULTS = "/srv/share3/mrudolph8/develop/habitat_spot_results/dan_kinematic"
    SLURM_TEMPLATE = "/nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-lab/slurm_job_template_max.sh"
    EVAL_SLURM_TEMPLATE = "/nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-lab/eval_slurm_template_max.sh"
else:
    EXP_YAML  = "habitat_baselines/config/pointnav/ddppo_pointnav_quadruped_train.yaml"
    EVAL_EXP_YAML  = "habitat_baselines/config/pointnav/ddppo_pointnav_quadruped_eval.yaml"
    if args.noisy_depth:
        TASK_YAML = "configs/tasks/pointnav_quadruped_train_noisy.yaml"
        EVAL_YAML = "configs/tasks/pointnav_quadruped_eval_noisy.yaml"
    else:
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

# Threshold to subtract from training goal success radius 
GOAL_THRESH = 0.05

robot_data_dict = {'A1': 'pointnav_a1_0.2',
                   'AlienGo': 'pointnav_aliengo_0.22',
                   'Locobot': 'pointnav_locobot_0.23',
                   'Spot': 'pointnav_spot_0.3'
                  }

robot_radius_dict = {'A1': 0.2,
                   'AlienGo': 0.22,
                   'Locobot': 0.23,
                   'Spot': 0.3
                  }

robots = args.robots
num_robots = len(robots)
robots_urdfs = [robot_urdfs_dict[robot] for robot in robots]

if num_robots > 1:
    robot_goal = min([robot_goal_dict[robot] for robot in robots])
else: 
    robot_goal = robot_goal_dict[robots[0]]

if num_robots > 1:
    largest_robot_value = max([robot_goal_dict[robot] for robot in robots])
    largest_robot = [k for k,v in robot_goal_dict.items() if v == largest_robot_value][0]
else: 
    largest_robot = robots[0]

largest_robot_radius = robot_radius_dict[largest_robot]
robot_data_path = 'pointnav_{}_{}'.format(largest_robot.lower(), largest_robot_radius)


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
        elif i.startswith('    RADIUS:'):
            task_yaml_data[idx] = "    RADIUS: {}".format(largest_robot_radius)
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
        elif i.startswith('      POLICY_LIN_VEL_RANGE:'):
            task_yaml_data[idx] = "      POLICY_LIN_VEL_RANGE: [{}, {}]".format(-args.max_lin_vel, args.max_lin_vel)
        elif i.startswith('      POLICY_HOR_VEL_RANGE:'):
            task_yaml_data[idx] = "      POLICY_HOR_VEL_RANGE: [{}, {}]".format(-args.max_hor_vel, args.max_hor_vel)
        elif i.startswith('      POLICY_ANG_VEL_RANGE:'):
            task_yaml_data[idx] = "      POLICY_ANG_VEL_RANGE: [{}, {}]".format(-args.max_ang_vel, args.max_ang_vel)
        elif i.startswith('      ROBOT_LIN_VEL_RANGES:'):
            lin_vel_ranges = [ast.literal_eval(n) for n in args.lin_vel_ranges]
            task_yaml_data[idx] = "      ROBOT_LIN_VEL_RANGES: {}".format(lin_vel_ranges)
        elif i.startswith('      ROBOT_HOR_VEL_RANGES:'):
            hor_vel_ranges = [ast.literal_eval(n) for n in args.hor_vel_ranges]
            task_yaml_data[idx] = "      ROBOT_HOR_VEL_RANGES: {}".format(hor_vel_ranges)
        elif i.startswith('      ROBOT_ANG_VEL_RANGES:'):
            ang_vel_ranges = [ast.literal_eval(n) for n in args.ang_vel_ranges]
            task_yaml_data[idx] = "      ROBOT_ANG_VEL_RANGES: {}".format(ang_vel_ranges)
        elif i.startswith('  SUCCESS_DISTANCE:'):
            task_yaml_data[idx] = "  SUCCESS_DISTANCE: {}".format(robot_goal-GOAL_THRESH)
        elif i.startswith('    SUCCESS_DISTANCE:'):
            task_yaml_data[idx] = "    SUCCESS_DISTANCE: {}".format(robot_goal-GOAL_THRESH)
        # elif i.startswith('    POSITION:'):
            # task_yaml_data[idx] = "    POSITION: {}".format(robots_heights[0])

        # elif i.startswith('    POSITION: '):
            # task_yaml_data[idx] = "    POSITION: [0.0, {}, -0.1778]".format(robot_camera_pos)
        elif i.startswith('SEED:'):
            task_yaml_data[idx] = "SEED: {}".format(args.seed)
        elif i.startswith('  DATA_PATH:'):
            data_path = '/coc/testnvme/jtruong33/data/datasets/pointnav_{}/{}/{{split}}/{{split}}.json.gz'.format(args.dataset, robot_data_path)
            task_yaml_data[idx] = "  DATA_PATH: {}".format(data_path)
        elif i.startswith('  SPLIT:'):
            if args.dataset == 'hm3d':
                task_yaml_data[idx] = "  SPLIT: train"

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
        elif i.startswith('    RADIUS:'):
            eval_yaml_data[idx] = "    RADIUS: {}".format(largest_robot_radius)
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
        elif i.startswith('      ROBOT_LIN_VEL_RANGES:'):
            lin_vel_ranges = [ast.literal_eval(n) for n in args.lin_vel_ranges]
            eval_yaml_data[idx] = "      ROBOT_LIN_VEL_RANGES: {}".format(lin_vel_ranges)
        elif i.startswith('      ROBOT_HOR_VEL_RANGES:'):
            hor_vel_ranges = [ast.literal_eval(n) for n in args.hor_vel_ranges]
            eval_yaml_data[idx] = "      ROBOT_HOR_VEL_RANGES: {}".format(hor_vel_ranges)
        elif i.startswith('      ROBOT_ANG_VEL_RANGES:'):
            ang_vel_ranges = [ast.literal_eval(n) for n in args.ang_vel_ranges]
            eval_yaml_data[idx] = "      ROBOT_ANG_VEL_RANGES: {}".format(ang_vel_ranges)
        elif i.startswith('  SUCCESS_DISTANCE:'):
            eval_yaml_data[idx] = "  SUCCESS_DISTANCE: {}".format(robot_goal)
        elif i.startswith('    SUCCESS_DISTANCE:'):
            eval_yaml_data[idx] = "    SUCCESS_DISTANCE: {}".format(robot_goal)
        elif i.startswith('SEED:'):
            eval_yaml_data[idx] = "SEED: {}".format(args.seed)
        elif i.startswith('  DATA_PATH:'):
            if args.dataset == 'hm3d':
                data_path = '/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d_gibson/pointnav_spot_0.6/{split}/{split}.json.gz'
                eval_yaml_data[idx] = "  DATA_PATH: {}".format(data_path)
        elif i.startswith('  SPLIT:'):
            if args.dataset == 'hm3d':
                eval_yaml_data[idx] = "  SPLIT: val"

    new_eval_yaml_path = new_eval_yaml_path[:-len('.yaml')]+'_'+ args.dataset + '_' + robots_underscore+'.yaml'
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
            tb_dir = 'tb_eval_{}_{}'.format(args.dataset, robots_underscore) if args.ckpt == -1 else 'tb_eval_{}_{}_ckpt={}'.format(args.dataset, robots_underscore, args.ckpt)
            if args.video:
                tb_dir = 'tb_eval_{}_{}_video'.format(args.dataset, robots_underscore) if args.ckpt == -1 else 'tb_eval_{}_{}_ckpt={}_video'.format(args.dataset, robots_underscore, args.ckpt)
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
        new_exp_eval_yaml_path = new_exp_yaml_path[:-len('.yaml')]+'_eval_'+ args.dataset + '_' +robots_underscore+'_video.yaml'
    else:
        new_exp_eval_yaml_path = new_exp_yaml_path[:-len('.yaml')]+'_eval_'+ args.dataset + '_' +robots_underscore+'.yaml'
    with open(new_exp_eval_yaml_path,'w') as f:
        f.write('\n'.join(exp_yaml_data))

    if args.video:
        robots_underscore += '_video'
    # Create slurm job
    with open(EVAL_SLURM_TEMPLATE) as f:
        slurm_data = f.read()
        slurm_data = slurm_data.replace('TEMPLATE', f'eval_{experiment_name}_{args.dataset}_{robots_underscore}')
        slurm_data = slurm_data.replace('HABITAT_REPO_PATH', HABITAT_LAB)
        slurm_data = slurm_data.replace('CONFIG_YAML', new_exp_eval_yaml_path)
        slurm_data = slurm_data.replace('PARTITION', args.partition)
        if args.partition == 'overcap':
            slurm_data = slurm_data.replace('# ACCOUNT', '#SBATCH --account overcap')

    slurm_path = os.path.join(dst_dir, experiment_name + '_eval_' + args.dataset + '_' + robots_underscore+'.sh')
    with open(slurm_path,'w') as f:
        f.write(slurm_data)
    print("Generated slurm job: "+slurm_path)

    # Submit slurm job
    cmd = 'sbatch '+slurm_path
    subprocess.check_call(cmd.split(), cwd=dst_dir)
    print('\nSee output with:\ntail -F {}'.format(os.path.join(dst_dir, 'eval_'+experiment_name+'_'+ args.dataset + '_' +robots_underscore+'.err')))

    # cmd = 'tmuxs eval_{}\n'.format(experiment_name)
    # cmd += 'srun --gres gpu:1 --nodes 1 --partition long --job-name eval_{} --exclude calculon --pty bash \n'.format(experiment_name)
    # cmd += 'aconda aug26n\n'
    # cmd += 'cd {}\n'.format(HABITAT_LAB)
    # cmd += 'python -u -m habitat_baselines.run --exp-config {} --run-type eval\n '.format(new_exp_eval_yaml_path)
    # print('\nCopy-paste and run the following:\n{}'.format(cmd))
    # subprocess.check_call(cmd.split(), cwd=HABITAT_LAB)