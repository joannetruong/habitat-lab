"""
Script to automate stuff.
Makes a new directory, and stores the two yaml files that generate the config.
Replaces the yaml file content with the location of the new directory.
"""
import argparse
import os
import shutil
import subprocess

HABITAT_LAB = "/private/home/akshararai/kin2dyn/habitat-lab"
CONDA_ENV = "/private/home/akshararai/.conda/envs/kin2dyn/bin/python"
RESULTS = "/checkpoint/akshararai/kin2dyn"
URDFS = "/private/home/akshararai/kin2dyn/habitat-lab/data/URDF_demo_assets"

SLURM_TEMPLATE = os.path.join(HABITAT_LAB, "slurm_template_ar.sh")
EVAL_SLURM_TEMPLATE = os.path.join(HABITAT_LAB, "eval_slurm_template_ar.sh")

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")

# Training
parser.add_argument("-sd", "--seed", type=int, default=100)
parser.add_argument("-r", "--robot", default="Spot")
parser.add_argument("-c", "--control-type", required=True)
parser.add_argument("-p", "--partition", default="long")
## options for dataset are hm3d_gibson, hm3d, gibson
parser.add_argument("-ds", "--dataset", default="hm3d_gibson")

## using spot's camera instead of intel realsense
parser.add_argument("-sc", "--spot_cameras", default=False, action="store_true")
parser.add_argument("-g", "--use_gray", default=False, action="store_true")
parser.add_argument("-gd", "--use_gray_depth", default=False, action="store_true")
parser.add_argument("-o", "--outdoor", default=False, action="store_true")

parser.add_argument("-curr", "--curriculum", default=False, action="store_true")

# Evaluation
parser.add_argument("-e", "--eval", default=False, action="store_true")
parser.add_argument("-cpt", "--ckpt", type=int, default=-1)
parser.add_argument("-v", "--video", default=False, action="store_true")

parser.add_argument("-d", "--debug", default=False, action="store_true")
parser.add_argument("--ext", default="")
args = parser.parse_args()

EXP_YAML = "habitat_baselines/config/pointnav/ddppo_pointnav_quadruped.yaml"
TASK_YAML = "configs/tasks/pointnav_quadruped.yaml"

experiment_name = args.experiment_name

dst_dir = os.path.join(RESULTS, experiment_name)
eval_dst_dir = os.path.join(RESULTS, experiment_name, "eval", args.control_type)

exp_yaml_path = os.path.join(HABITAT_LAB, EXP_YAML)
task_yaml_path = os.path.join(HABITAT_LAB, TASK_YAML)

new_task_yaml_path = os.path.join(dst_dir, os.path.basename(task_yaml_path))
new_exp_yaml_path = os.path.join(dst_dir, os.path.basename(exp_yaml_path))

exp_name = "_{}".format(args.control_type)

if args.eval:
    exp_name += "_eval"
if args.ckpt != -1:
    exp_name += "_ckpt_{}".format(args.ckpt)
if args.video:
    exp_name += "_video"
if args.ext != "":
    exp_name += "_" + args.ext
    eval_dst_dir += "_" + args.ext
if args.outdoor:
    exp_name += "_ferst"

new_eval_task_yaml_path = (
    os.path.join(eval_dst_dir, os.path.basename(task_yaml_path)).split(".yaml")[0]
    + exp_name
    + ".yaml"
)
new_eval_exp_yaml_path = (
    os.path.join(eval_dst_dir, os.path.basename(exp_yaml_path)).split(".yaml")[0]
    + exp_name
    + ".yaml"
)

robot_urdfs_dict = {
    "A1": os.path.join(URDFS, "a1/a1.urdf"),
    "AlienGo":  os.path.join(URDFS, "aliengo/urdf/aliengo.urdf"),
    "Daisy":  os.path.join(URDFS, "daisy/daisy_advanced_akshara.urdf"),
    "Spot":  os.path.join(URDFS, "spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid.urdf"),
    "Locobot":  os.path.join(URDFS, "locobot/urdf/locobot_description2.urdf"),
}

num_steps_dict = {
    "a1": 326,
    "aliengo": 268,
    "spot": 150,
}

robot_goal_dict = {
    "A1": 0.24,
    "AlienGo": 0.3235,
    "Locobot": 0.20,
    "Spot": 0.425,
}

robot_vel_dict = {
    "A1": [[-0.23, 0.23], [-8.02, 8.02]],
    "AlienGo": [[-0.28, 0.28], [-9.74, 9.74]],
    "Spot": [[-0.5, 0.5], [-17.19, 17.19]],
}
robot_radius_dict = {"A1": 0.2, "AlienGo": 0.22, "Locobot": 0.23, "Spot": 0.3}

robot = args.robot
robot_urdf = robot_urdfs_dict[robot]

robot_lin_vel, robot_ang_vel = robot_vel_dict[robot]
succ_radius = robot_goal_dict[robot]
robot_radius = robot_radius_dict[robot]

robot_num_steps = num_steps_dict[args.robot.lower()]

# Training
if not args.eval:
    # Create directory
    if os.path.isdir(dst_dir):
        response = input(
            "'{}' already exists. Delete or abort? [d/A]: ".format(dst_dir)
        )
        if response == "d":
            print("Deleting {}".format(dst_dir))
            shutil.rmtree(dst_dir)
        else:
            print("Aborting.")
            exit()
    os.mkdir(dst_dir)
    print("Created " + dst_dir)

    # Create task yaml file, using file within Habitat Lab repo as a template
    with open(task_yaml_path) as f:
        task_yaml_data = f.read().splitlines()

    # robots_heights = [robot_heights_dict[robot] for robot in robots]

    for idx, i in enumerate(task_yaml_data):
        if i.startswith("  MAX_EPISODE_STEPS:"):
            task_yaml_data[idx] = "  MAX_EPISODE_STEPS: {}".format(robot_num_steps)
        elif i.startswith("  CURRICULUM:"):
            task_yaml_data[idx] = "  CURRICULUM: {}".format(args.curriculum)
        elif i.startswith("    RADIUS:"):
            task_yaml_data[idx] = "    RADIUS: {}".format(robot_radius)
        elif i.startswith("  ROBOT:"):
            task_yaml_data[idx] = "  ROBOT: '{}'".format(robot)
        elif i.startswith("  ROBOT_URDF:"):
            task_yaml_data[idx] = "  ROBOT_URDF: {}".format(robot_urdf)
        elif i.startswith("  POSSIBLE_ACTIONS:"):
            if args.control_type == "dynamic":
                control_type = "DYNAMIC_VELOCITY_CONTROL"
            else:
                control_type = "VELOCITY_CONTROL"
            task_yaml_data[idx] = '  POSSIBLE_ACTIONS: ["{}"]'.format(control_type)
        elif i.startswith("    VELOCITY_CONTROL:"):
            if args.control_type == "dynamic":
                task_yaml_data[idx] = "    DYNAMIC_VELOCITY_CONTROL:"
        elif i.startswith("    DYNAMIC_VELOCITY_CONTROL:"):
            if not args.control_type == "dynamic":
                task_yaml_data[idx] = "    VELOCITY_CONTROL:"
        elif i.startswith("      TIME_STEP:"):
            if args.control_type == "dynamic":
                task_yaml_data[idx] = "      TIME_STEP: 0.33"
        elif i.startswith("      LIN_VEL_RANGE:"):
            task_yaml_data[idx] = "      LIN_VEL_RANGE: [{}, {}]".format(
                robot_lin_vel[0], robot_lin_vel[1]
            )
        elif i.startswith("      ANG_VEL_RANGE:"):
            task_yaml_data[idx] = "      ANG_VEL_RANGE: [{}, {}]".format(
                robot_ang_vel[0], robot_ang_vel[1]
            )
        elif i.startswith("      HOR_VEL_RANGE:"):
            task_yaml_data[idx] = "      HOR_VEL_RANGE: [{}, {}]".format(
                robot_lin_vel[0], robot_lin_vel[1]
            )
        elif i.startswith("  SUCCESS_DISTANCE:"):
            task_yaml_data[idx] = "  SUCCESS_DISTANCE: {}".format(succ_radius - 0.05)
        elif i.startswith("    SUCCESS_DISTANCE:"):
            task_yaml_data[idx] = "    SUCCESS_DISTANCE: {}".format(succ_radius - 0.05)
        elif i.startswith("SEED:"):
            task_yaml_data[idx] = "SEED: {}".format(args.seed)
        elif i.startswith("  DATA_PATH:"):
            if args.outdoor:
                data_path = "/coc/testnvme/jtruong33/data/datasets/ferst/{split}/{split}.json.gz"

    with open(new_task_yaml_path, "w") as f:
        f.write("\n".join(task_yaml_data))
    print("Created " + new_task_yaml_path)

    # Create experiment yaml file, using file within Habitat Lab repo as a template
    with open(exp_yaml_path) as f:
        exp_yaml_data = f.read().splitlines()

    for idx, i in enumerate(exp_yaml_data):
        if i.startswith("BASE_TASK_CONFIG_PATH:"):
            exp_yaml_data[idx] = "BASE_TASK_CONFIG_PATH: '{}'".format(
                new_task_yaml_path
            )
        elif i.startswith("TOTAL_NUM_STEPS:"):
            max_num_steps = 5e8 if args.control_type == "kinematic" else 5e7
            exp_yaml_data[idx] = "TOTAL_NUM_STEPS: {}".format(max_num_steps)
        elif i.startswith("TENSORBOARD_DIR:"):
            exp_yaml_data[idx] = "TENSORBOARD_DIR:    '{}'".format(
                os.path.join(dst_dir, "tb")
            )
        elif i.startswith("NUM_ENVIRONMENTS:"):
            if args.use_gray or args.use_gray_depth:
                exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 8"
            if args.outdoor:
                exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 1"
        elif i.startswith("SENSORS:"):
            if args.spot_cameras and args.use_gray:
                exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
            elif args.spot_cameras and args.use_gray_depth:
                exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR', 'SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
            elif args.spot_cameras:
                exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR']"
            else:
                exp_yaml_data[idx] = "SENSORS: ['DEPTH_SENSOR']"
        elif i.startswith("VIDEO_DIR:"):
            exp_yaml_data[idx] = "VIDEO_DIR:          '{}'".format(
                os.path.join(dst_dir, "video_dir")
            )
        elif i.startswith("EVAL_CKPT_PATH_DIR:"):
            exp_yaml_data[idx] = "EVAL_CKPT_PATH_DIR: '{}'".format(
                os.path.join(dst_dir, "checkpoints")
            )
        elif i.startswith("CHECKPOINT_FOLDER:"):
            exp_yaml_data[idx] = "CHECKPOINT_FOLDER:  '{}'".format(
                os.path.join(dst_dir, "checkpoints")
            )
        elif i.startswith("TXT_DIR:"):
            exp_yaml_data[idx] = "TXT_DIR:            '{}'".format(
                os.path.join(dst_dir, "txts")
            )

    with open(new_exp_yaml_path, "w") as f:
        f.write("\n".join(exp_yaml_data))
    print("Created " + new_exp_yaml_path)

    # Create slurm job
    with open(SLURM_TEMPLATE) as f:
        slurm_data = f.read()
        slurm_data = slurm_data.replace("$TEMPLATE", experiment_name)
        slurm_data = slurm_data.replace("$CONDA_ENV", CONDA_ENV)
        slurm_data = slurm_data.replace("$HABITAT_REPO_PATH", HABITAT_LAB)
        slurm_data = slurm_data.replace("$LOG", os.path.join(dst_dir, experiment_name))
        slurm_data = slurm_data.replace("$CONFIG_YAML", new_exp_yaml_path)
        slurm_data = slurm_data.replace("$PARTITION", args.partition)
    if args.debug:
        slurm_data = slurm_data.replace("$GPUS", "1")
    else:
        slurm_data = slurm_data.replace("$GPUS", "8")

    slurm_path = os.path.join(dst_dir, experiment_name + ".sh")
    with open(slurm_path, "w") as f:
        f.write(slurm_data)
    print("Generated slurm job: " + slurm_path)

    # Submit slurm job
    cmd = "sbatch " + slurm_path
    subprocess.check_call(cmd.split(), cwd=dst_dir)

    print(
        "\nSee output with:\ntail -F {}".format(
            os.path.join(dst_dir, experiment_name + ".err")
        )
    )

# Evaluation
else:
    # Make sure folder exists
    assert os.path.isdir(dst_dir), "{} directory does not exist".format(dst_dir)
    os.makedirs(eval_dst_dir, exist_ok=True)
    # Create task yaml file, using file within Habitat Lab repo as a template
    with open(task_yaml_path) as f:
        eval_yaml_data = f.read().splitlines()

    for idx, i in enumerate(eval_yaml_data):
        if i.startswith("  MAX_EPISODE_STEPS:"):
            eval_yaml_data[idx] = "  MAX_EPISODE_STEPS: {}".format(robot_num_steps)
        elif i.startswith("    RADIUS:"):
            eval_yaml_data[idx] = "    RADIUS: {}".format(robot_radius)
        elif i.startswith("  ROBOT:"):
            eval_yaml_data[idx] = "  ROBOT: '{}'".format(robot)
        elif i.startswith("  ROBOT_URDF:"):
            eval_yaml_data[idx] = "  ROBOT_URDF: {}".format(robot_urdf)
        elif i.startswith("  POSSIBLE_ACTIONS:"):
            if args.control_type == "dynamic":
                control_type = "DYNAMIC_VELOCITY_CONTROL"
            else:
                control_type = "VELOCITY_CONTROL"
            eval_yaml_data[idx] = '  POSSIBLE_ACTIONS: ["{}"]'.format(control_type)
        elif i.startswith("    VELOCITY_CONTROL:"):
            if args.control_type == "dynamic":
                eval_yaml_data[idx] = "    DYNAMIC_VELOCITY_CONTROL:"
        elif i.startswith("    DYNAMIC_VELOCITY_CONTROL:"):
            if not args.control_type == "dynamic":
                eval_yaml_data[idx] = "    VELOCITY_CONTROL:"
        elif i.startswith("      TIME_STEP:"):
            if args.control_type == "dynamic":
                eval_yaml_data[idx] = "      TIME_STEP: 0.33"
        elif i.startswith("      LIN_VEL_RANGE:"):
            eval_yaml_data[idx] = "      LIN_VEL_RANGE: [{}, {}]".format(
                robot_lin_vel[0], robot_lin_vel[1]
            )
        elif i.startswith("      ANG_VEL_RANGE:"):
            eval_yaml_data[idx] = "      ANG_VEL_RANGE: [{}, {}]".format(
                robot_ang_vel[0], robot_ang_vel[1]
            )
        elif i.startswith("      HOR_VEL_RANGE:"):
            eval_yaml_data[idx] = "      HOR_VEL_RANGE: [{}, {}]".format(
                robot_lin_vel[0], robot_lin_vel[1]
            )
        elif i.startswith("  SUCCESS_DISTANCE:"):
            eval_yaml_data[idx] = "  SUCCESS_DISTANCE: {}".format(succ_radius)
        elif i.startswith("    SUCCESS_DISTANCE:"):
            eval_yaml_data[idx] = "    SUCCESS_DISTANCE: {}".format(succ_radius)
        elif i.startswith("SEED:"):
            eval_yaml_data[idx] = "SEED: {}".format(args.seed)

    with open(new_eval_task_yaml_path, "w") as f:
        f.write("\n".join(eval_yaml_data))
    print("Created " + new_eval_task_yaml_path)

    # Edit the stored experiment yaml file
    with open(exp_yaml_path) as f:
        eval_exp_yaml_data = f.read().splitlines()

    for idx, i in enumerate(eval_exp_yaml_data):
        if i.startswith("BASE_TASK_CONFIG_PATH:"):
            eval_exp_yaml_data[idx] = "BASE_TASK_CONFIG_PATH: '{}'".format(
                new_eval_task_yaml_path
            )
        elif i.startswith("TENSORBOARD_DIR:"):
            tb_dir = "tb_eval_{}".format(args.control_type)
            if args.ckpt != -1:
                tb_dir += "_ckpt_{}".format(args.ckpt)
            if args.video:
                tb_dir += "_video"
            eval_exp_yaml_data[idx] = "TENSORBOARD_DIR:    '{}'".format(
                os.path.join(eval_dst_dir, "tb_evals", tb_dir)
            )
        elif i.startswith("NUM_PROCESSES:"):
            eval_exp_yaml_data[idx] = "NUM_PROCESSES: 13"
        elif i.startswith("CHECKPOINT_FOLDER:"):
            eval_exp_yaml_data[idx] = "CHECKPOINT_FOLDER:  '{}'".format(
                os.path.join(dst_dir, "checkpoints")
            )
        elif i.startswith("EVAL_CKPT_PATH_DIR:"):
            if args.ckpt == -1:
                eval_exp_yaml_data[idx] = "EVAL_CKPT_PATH_DIR: '{}'".format(
                    os.path.join(dst_dir, "checkpoints")
                )
            else:
                eval_exp_yaml_data[idx] = "EVAL_CKPT_PATH_DIR: '{}/ckpt.{}.pth'".format(
                    os.path.join(dst_dir, "checkpoints"), args.ckpt
                )
        elif i.startswith("TXT_DIR:"):
            txt_dir = "txts_eval_{}".format(args.control_type)
            if args.ckpt != -1:
                txt_dir += "_ckpt_{}".format(args.ckpt)
            eval_exp_yaml_data[idx] = "TXT_DIR:            '{}'".format(
                os.path.join(eval_dst_dir, "txts", txt_dir)
            )
        elif i.startswith("VIDEO_OPTION:"):
            if args.video:
                eval_exp_yaml_data[idx] = "VIDEO_OPTION: ['disk']"
            else:
                eval_exp_yaml_data[idx] = "VIDEO_OPTION: []"
        elif i.startswith("NUM_ENVIRONMENTS:"):
            if args.use_gray or args.use_gray_depth:
                eval_exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 8"
            if args.outdoor:
                eval_exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 1"
        elif i.startswith("SENSORS:"):
            if args.spot_cameras and args.use_gray:
                eval_exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
                if args.video:
                    eval_exp_yaml_data[
                        idx
                    ] = "SENSORS: ['RGB_SENSOR', 'SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
            elif args.spot_cameras:
                eval_exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR']"
                if args.video:
                    eval_exp_yaml_data[
                        idx
                    ] = "SENSORS: ['RGB_SENSOR', 'SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR']"
            else:
                eval_exp_yaml_data[idx] = "SENSORS: ['DEPTH_SENSOR']"
                if args.video:
                    eval_exp_yaml_data[idx] = "SENSORS: ['RGB_SENSOR','DEPTH_SENSOR']"
        elif i.startswith("VIDEO_DIR:"):
            video_dir = (
                "video_dir"
                if args.ckpt == -1
                else "video_dir_{}_ckpt_{}".format(args.control_type, args.ckpt)
            )
            eval_exp_yaml_data[idx] = "VIDEO_DIR:          '{}'".format(
                os.path.join(eval_dst_dir, "videos", video_dir)
            )
        elif i.startswith("    SPLIT:"):
            if args.dataset == "hm3d":
                eval_exp_yaml_data[idx] = "    SPLIT: val"
            elif args.dataset == "ferst_20m":
                eval_exp_yaml_data[idx] = "    SPLIT: val_20m"

    if os.path.isdir(tb_dir):
        response = input(
            "{} directory already exists. Delete, continue, or abort? [d/c/A]: ".format(
                tb_dir
            )
        )
        if response == "d":
            print("Deleting {}".format(tb_dir))
            shutil.rmtree(tb_dir)
        elif response == "c":
            print("Continuing.")
        else:
            print("Aborting.")
            exit()

    with open(new_eval_exp_yaml_path, "w") as f:
        f.write("\n".join(eval_exp_yaml_data))
    print("Created " + new_eval_exp_yaml_path)

    eval_experiment_name = experiment_name + exp_name

    # Create slurm job
    with open(EVAL_SLURM_TEMPLATE) as f:
        slurm_data = f.read()
        slurm_data = slurm_data.replace("$TEMPLATE", eval_experiment_name)
        slurm_data = slurm_data.replace("$CONDA_ENV", CONDA_ENV)
        slurm_data = slurm_data.replace("$HABITAT_REPO_PATH", HABITAT_LAB)
        slurm_data = slurm_data.replace(
            "$LOG", os.path.join(eval_dst_dir, eval_experiment_name)
        )
        slurm_data = slurm_data.replace("$CONFIG_YAML", new_eval_exp_yaml_path)
        slurm_data = slurm_data.replace("$PARTITION", args.partition)
        if args.partition == "overcap":
            slurm_data = slurm_data.replace("# ACCOUNT", "#SBATCH --account overcap")
        elif args.partition == "user-overcap":
            slurm_data = slurm_data.replace(
                "# ACCOUNT", "#SBATCH --account user-overcap"
            )
    slurm_path = os.path.join(eval_dst_dir, eval_experiment_name + ".sh")
    with open(slurm_path, "w") as f:
        f.write(slurm_data)
    print("Generated slurm job: " + slurm_path)

    # Submit slurm job
    cmd = "sbatch " + slurm_path
    subprocess.check_call(cmd.split(), cwd=dst_dir)
    print(
        "\nSee output with:\ntail -F {}".format(
            os.path.join(eval_dst_dir, eval_experiment_name + ".err")
        )
    )
