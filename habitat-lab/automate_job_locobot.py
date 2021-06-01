'''
Script to automate stuff.
Makes a new directory, and stores the two yaml files that generate the config.
Replaces the yaml file content with the location of the new directory.
'''

HABITAT_LAB = "/coc/testnvme/jtruong33/habitat-cont-v2/habitat-lab"
RESULTS = "/coc/pskynet3/jtruong33/develop/flash_results/cont_ctrl_results_v2"
EXP_YAML  = "habitat_baselines/config/pointnav/ddppo_pointnav.yaml"
TASK_YAML = "configs/tasks/pointnav_locobot.yaml"
EVAL_YAML = "configs/tasks/pointnav_locobot_eval.yaml"
SLURM_TEMPLATE = "/coc/testnvme/jtruong33/habitat-cont-v2/habitat-lab/slurm_job_tempate.sh"
EVAL_SLURM_TEMPLATE = "/coc/testnvme/jtruong33/habitat-cont-v2/habitat-lab/eval_slurm_template.sh"

import os
import argparse
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('experiment_name')

# Training
parser.add_argument('-y','--decay', type=int, default=-1)
parser.add_argument('-pn','--penalty', type=float, default=-0.1)
parser.add_argument('-skr','--slack_reward', type=float, default=-0.01)
parser.add_argument('-ty','--proximity_penalty_type', type=str, default='None')
parser.add_argument('-s','--sliding_on', default=False, action='store_true')
parser.add_argument('-pe','--use_pretrained_encoder', default=False, action='store_true')
parser.add_argument('-mc','--max_collisions', type=int, default=-1)
parser.add_argument('-b','--allow_backwards', default=False, action='store_true')
parser.add_argument('-sf','--use_strafe', default=False, action='store_true')
parser.add_argument('-as','--auto_stop', default=False, action='store_true')
parser.add_argument('-l','--max_linear', type=float, default=0.35)
parser.add_argument('-ms','--max_strafe', type=float, default=0.15)
parser.add_argument('-g','--max_angular', type=int, default=10)
parser.add_argument('-p','--partition', type=str, default='long')
# Evaluation
parser.add_argument('-e','--eval', default=False, action='store_true')
parser.add_argument('-c','--ckpt', type=int, default=-1)
parser.add_argument('-v','--video', default=False, action='store_true')

parser.add_argument('-d','--debug', default=False, action='store_true')
args = parser.parse_args()
experiment_name = args.experiment_name


dst_dir = os.path.join(RESULTS, experiment_name)
exp_yaml_path  = os.path.join(HABITAT_LAB, EXP_YAML)
task_yaml_path = os.path.join(HABITAT_LAB, TASK_YAML)
eval_yaml_path = os.path.join(HABITAT_LAB, EVAL_YAML)
new_task_yaml_path = os.path.join(dst_dir, os.path.basename(task_yaml_path))
new_eval_yaml_path = os.path.join(dst_dir, os.path.basename(eval_yaml_path))
new_exp_yaml_path  = os.path.join(dst_dir, os.path.basename(exp_yaml_path))

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

    for idx, i in enumerate(task_yaml_data):
        if i.startswith('    ALLOW_SLIDING:'):
            if args.sliding_on:
                task_yaml_data[idx] = "    ALLOW_SLIDING: True"
            else:
                task_yaml_data[idx] = "    ALLOW_SLIDING: False"
        elif i.startswith('      LIN_VEL_RANGE:'):
            if args.allow_backwards:
                task_yaml_data[idx] = "      LIN_VEL_RANGE: [{}, {}]".format(-args.max_linear, args.max_linear)
            else:
                task_yaml_data[idx] = "      LIN_VEL_RANGE: [{}, {}]".format(str(0.0), args.max_linear)
        elif i.startswith('      STRAFE_VEL_RANGE:'):
            task_yaml_data[idx] = "      STRAFE_VEL_RANGE: [{}, {}]".format(-args.max_strafe, args.max_strafe)
        elif i.startswith('      ANG_VEL_RANGE:'):
            task_yaml_data[idx] = "      ANG_VEL_RANGE: [{}, {}]".format(-args.max_angular, args.max_angular)
        elif i.startswith('      USE_STRAFE:'):
            if args.use_strafe:
                task_yaml_data[idx] = "      USE_STRAFE: True"
            else:
                task_yaml_data[idx] = "      USE_STRAFE: False"
        elif i.startswith('      AUTO_STOP:'):
            if args.auto_stop:
                task_yaml_data[idx] = "      AUTO_STOP: True"
            else:
                task_yaml_data[idx] = "      AUTO_STOP: False"
        elif i.startswith('      MAX_COLLISIONS:'):
            task_yaml_data[idx] = "      MAX_COLLISIONS: {}".format(args.max_collisions)

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
        elif i.startswith('  STEP_REWARD_DECAY:'):
            exp_yaml_data[idx] = "  STEP_REWARD_DECAY: {}".format(args.decay)
        elif i.startswith('  SLACK_REWARD:'):
            exp_yaml_data[idx] = "  SLACK_REWARD: {}".format(args.slack_reward)
        elif i.startswith('  STEP_PROXIMITY_PENALTY:'):
            exp_yaml_data[idx] = "  STEP_PROXIMITY_PENALTY: {}".format(args.penalty)
        elif i.startswith('  PROXIMITY_PENALTY_TYPE:'):
            exp_yaml_data[idx] = "  PROXIMITY_PENALTY_TYPE: {}".format(args.proximity_penalty_type)
        elif i.startswith('    pretrained_encoder:'):
            if args.use_pretrained_encoder:
                exp_yaml_data[idx] = "    pretrained_encoder: True"
            else:
                exp_yaml_data[idx] = "    pretrained_encoder: False"
        elif i.startswith('    train_encoder:'):
            if args.use_pretrained_encoder:
                exp_yaml_data[idx] = "    train_encoder: False"
            else:
                exp_yaml_data[idx] = "    train_encoder: True"
        elif i.startswith('NUM_PROCESSES:'):
            if args.debug:
                exp_yaml_data[idx] = "NUM_PROCESSES: 1"
            else:
                exp_yaml_data[idx] = "NUM_PROCESSES: 8"

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

    with open(new_eval_yaml_path,'w') as f:
        f.write('\n'.join(eval_yaml_data))
    print("Created "+new_eval_yaml_path)
    
    # Edit the stored experiment yaml file
    with open(new_exp_yaml_path) as f:
        exp_yaml_data = f.read().splitlines()

    for idx, i in enumerate(exp_yaml_data):
        if i.startswith('BASE_TASK_CONFIG_PATH:'):
            exp_yaml_data[idx] = "BASE_TASK_CONFIG_PATH: '{}'".format(new_eval_yaml_path)
        elif i.startswith('TENSORBOARD_DIR:'):
            tb_dir = 'tb_eval' if args.ckpt == -1 else 'tb_eval_{}'.format(args.ckpt)
            tb_dir = os.path.join(dst_dir, tb_dir)
            exp_yaml_data[idx] = "TENSORBOARD_DIR:    '{}'".format(tb_dir)
        elif i.startswith('NUM_PROCESSES:'):
            exp_yaml_data[idx] = "NUM_PROCESSES: 13"
        elif i.startswith('CHECKPOINT_FOLDER:'):
            ckpt_dir = i.split()[-1][1:-1]
        elif i.startswith('VIDEO_OPTION:'):
            if args.video:
                exp_yaml_data[idx] = "VIDEO_OPTION: ['disk']"
            else:
                exp_yaml_data[idx] = "VIDEO_OPTION: []"
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
        slurm_data = slurm_data.replace('YAML_PATH', new_exp_eval_yaml_path)
        slurm_data = slurm_data.replace('JOB_NAME',  f'eval_{experiment_name}')
    slurm_path = os.path.join(dst_dir, experiment_name+'_eval.sh')
    with open(slurm_path,'w') as f:
        f.write(slurm_data)
    print("Generated slurm job: "+slurm_path)

    # Submit slurm job
    cmd = 'sbatch '+slurm_path
    subprocess.check_call(cmd.split(), cwd=dst_dir)
    print('\nSee output with:\ntail -F {}'.format(os.path.join(dst_dir, 'eval_'+experiment_name+'_eval.err')))

    # cmd = 'tmuxs eval_{}\n'.format(experiment_name)
    # cmd += 'srun --gres gpu:1 --nodes 1 --partition long --job-name eval_{} --exclude calculon --pty bash \n'.format(experiment_name)
    # cmd += 'aconda aug26n\n'
    # cmd += 'cd {}\n'.format(HABITAT_LAB)
    # cmd += 'python -u -m habitat_baselines.run --exp-config {} --run-type eval\n '.format(new_exp_eval_yaml_path)
    # print('\nCopy-paste and run the following:\n{}'.format(cmd))
    # subprocess.check_call(cmd.split(), cwd=HABITAT_LAB)