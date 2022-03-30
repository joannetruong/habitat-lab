"""
Script to automate stuff.
Makes a new directory, and stores the two yaml files that generate the config.
Replaces the yaml file content with the location of the new directory.
"""
import argparse
import os
import shutil
import subprocess
import sys

automate_command = "python " + " ".join(sys.argv)
HABITAT_LAB = "/coc/testnvme/jtruong33/google_nav/habitat-lab"
CONDA_ENV = "/nethome/jtruong33/miniconda3/envs/habitat-outdoor/bin/python"
RESULTS = "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results"
SLURM_TEMPLATE = os.path.join(HABITAT_LAB, "slurm_template.sh")
EVAL_SLURM_TEMPLATE = os.path.join(HABITAT_LAB, "eval_slurm_template.sh")

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")

# Training
parser.add_argument("-sd", "--seed", type=int, default=100)
parser.add_argument("-r", "--robot", default="Spot")
parser.add_argument("-c", "--control-type", default="kinematic")
parser.add_argument("-p", "--partition", default="long")
parser.add_argument("--constraint", default=False, action="store_true")

## options for dataset are hm3d_gibson, hm3d, gibson
parser.add_argument("-ds", "--dataset", default="hm3d_gibson")
parser.add_argument("-ne", "--num_environments", type=int, default=8)

parser.add_argument("-g", "--use_gray", default=False, action="store_true")
parser.add_argument("-gd", "--use_gray_depth", default=False, action="store_true")
parser.add_argument("-o", "--outdoor", default=False, action="store_true")
parser.add_argument("--coda", default=False, action="store_true")

parser.add_argument("--splitnet", default=False, action="store_true")
parser.add_argument("-sn", "--surface_normal", default=False, action="store_true")

parser.add_argument("-2cnn", "--two_cnns", default=False, action="store_true")

parser.add_argument("-pn", "--pepper_noise", default=False, action="store_true")
parser.add_argument("-rn", "--redwood_noise", default=False, action="store_true")
parser.add_argument("-mb", "--median_blur", default=False, action="store_true")
parser.add_argument("-ks", "--kernel_size", type=float, default=9)
parser.add_argument("-ni", "--num_iters", type=int, default=1)
parser.add_argument("-np", "--noise_percent", type=float, default=0.4)
parser.add_argument("-cn", "--cutout_noise", default=False, action="store_true")
parser.add_argument("-curr", "--curriculum", default=False, action="store_true")

# Evaluation
parser.add_argument("-e", "--eval", default=False, action="store_true")
parser.add_argument("-cpt", "--ckpt", type=int, default=-1)
parser.add_argument("-v", "--video", default=False, action="store_true")

parser.add_argument("-d", "--debug", default=False, action="store_true")
parser.add_argument("-x", default=False, action="store_true")
parser.add_argument("--ext", default="")
args = parser.parse_args()

if args.splitnet:
    EXP_YAML = "habitat_baselines/config/pointnav/ddppo_pointnav_spot_splitnet.yaml"
else:
    EXP_YAML = "habitat_baselines/config/pointnav/ddppo_pointnav_spot.yaml"

if args.redwood_noise:
    TASK_YAML = "configs/tasks/pointnav_spot_redwood.yaml"
else:
    TASK_YAML = "configs/tasks/pointnav_spot.yaml"

experiment_name = args.experiment_name

dst_dir = os.path.join(RESULTS, experiment_name)
eval_dst_dir = os.path.join(RESULTS, experiment_name, "eval", args.control_type)

exp_yaml_path = os.path.join(HABITAT_LAB, EXP_YAML)
task_yaml_path = os.path.join(HABITAT_LAB, TASK_YAML)

new_task_yaml_path = os.path.join(dst_dir, os.path.basename(task_yaml_path))
new_exp_yaml_path = os.path.join(dst_dir, os.path.basename(exp_yaml_path))

exp_name = f"_{args.control_type}"

if args.eval:
    exp_name += "_eval"
if args.ckpt != -1:
    exp_name += f"_ckpt_{args.ckpt}"
    eval_dst_dir += f"_ckpt_{args.ckpt}"
if args.video:
    exp_name += "_video"
    eval_dst_dir += "_video"
if args.ext != "":
    exp_name += "_" + args.ext
    eval_dst_dir += "_" + args.ext
if args.outdoor:
    exp_name += "_ferst"
    eval_dst_dir += "_ferst"
if args.dataset == "coda":
    exp_name += "_coda"
    eval_dst_dir += "_coda"
if args.pepper_noise:
    exp_name += f"_pepper_noise_{args.noise_percent}"
    eval_dst_dir += f"_pepper_noise_{args.noise_percent}"
if args.redwood_noise:
    exp_name += f"_redwood_noise_{args.noise_percent}"
    eval_dst_dir += f"_redwood_noise_{args.noise_percent}"
if args.median_blur:
    exp_name += f"_median_blur_{args.kernel_size}"
    eval_dst_dir += f"_median_blur_{args.kernel_size}"

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
    "A1": "/coc/testnvme/jtruong33/data/URDF_demo_assets/a1/a1.urdf",
    "AlienGo": "/coc/testnvme/jtruong33/data/URDF_demo_assets/aliengo/urdf/aliengo.urdf",
    "Daisy": "/coc/testnvme/jtruong33/data/URDF_demo_assets/daisy/daisy_advanced_akshara.urdf",
    "Spot": "/coc/testnvme/jtruong33/data/URDF_demo_assets/spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid.urdf",
    "Locobot": "/coc/testnvme/jtruong33/data/URDF_demo_assets/locobot/urdf/locobot_description2.urdf",
}

robot_goal_dict = {
    "A1": 0.24,
    "AlienGo": 0.3235,
    "Locobot": 0.20,
    "Spot": 0.425,
}
robot_radius_dict = {"A1": 0.2, "AlienGo": 0.22, "Locobot": 0.23, "Spot": 0.3}

robot = args.robot
robot_goal = robot_goal_dict[robot]
robot_urdf = robot_urdfs_dict[robot]

succ_radius = robot_goal_dict[robot]
robot_radius = robot_radius_dict[robot]

# Training
if not args.eval:
    # Create directory
    if os.path.isdir(dst_dir):
        response = input(f"'{dst_dir}' already exists. Delete or abort? [d/A]: ")
        if response == "d":
            print(f"Deleting {dst_dir}")
            shutil.rmtree(dst_dir)
        else:
            print("Aborting.")
            exit()
    os.mkdir(dst_dir)
    print("Created " + dst_dir)

    with open(os.path.join(dst_dir, "automate_job_cmd.txt"), "w") as f:
        f.write(automate_command)
    print("Saved automate command: " + os.path.join(dst_dir, "automate_job_cmd.txt"))

    # Create task yaml file, using file within Habitat Lab repo as a template
    with open(task_yaml_path) as f:
        task_yaml_data = f.read().splitlines()

    # robots_heights = [robot_heights_dict[robot] for robot in robots]

    for idx, i in enumerate(task_yaml_data):
        if i.startswith("  CURRICULUM:"):
            task_yaml_data[idx] = f"  CURRICULUM: {args.curriculum}"
        elif i.startswith("    RADIUS:"):
            task_yaml_data[idx] = f"    RADIUS: {robot_radius}"
        elif i.startswith("  ROBOT:"):
            task_yaml_data[idx] = f"  ROBOT: '{robot}'"
        elif i.startswith("      ROBOT_URDF:"):
            task_yaml_data[idx] = f"      ROBOT_URDF: {robot_urdf}"
        elif i.startswith("  POSSIBLE_ACTIONS:"):
            if args.control_type == "dynamic":
                control_type = "DYNAMIC_VELOCITY_CONTROL"
            else:
                control_type = "VELOCITY_CONTROL"
            task_yaml_data[idx] = f'  POSSIBLE_ACTIONS: ["{control_type}"]'
        elif i.startswith("    VELOCITY_CONTROL:"):
            if args.control_type == "dynamic":
                task_yaml_data[idx] = "    DYNAMIC_VELOCITY_CONTROL:"
        elif i.startswith("    DYNAMIC_VELOCITY_CONTROL:"):
            if not args.control_type == "dynamic":
                task_yaml_data[idx] = "    VELOCITY_CONTROL:"
        elif i.startswith("      TIME_STEP:"):
            if args.control_type == "dynamic":
                task_yaml_data[idx] = "      TIME_STEP: 0.33"
        elif i.startswith("  SUCCESS_DISTANCE:"):
            task_yaml_data[idx] = f"  SUCCESS_DISTANCE: {succ_radius - 0.05}"
        elif i.startswith("    SUCCESS_DISTANCE:"):
            task_yaml_data[idx] = f"    SUCCESS_DISTANCE: {succ_radius - 0.05}"
        elif i.startswith("SEED:"):
            task_yaml_data[idx] = f"SEED: {args.seed}"
        elif i.startswith("  DATA_PATH:"):
            if args.outdoor:
                data_path = "/coc/testnvme/jtruong33/data/datasets/ferst/{split}/{split}.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            if args.dataset == "coda":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test/coda_spot_test.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
        elif i.startswith("      noise_multiplier:"):
            task_yaml_data[idx] = f"      noise_multiplier: {args.noise_percent}"
    with open(new_task_yaml_path, "w") as f:
        f.write("\n".join(task_yaml_data))
    print("Created " + new_task_yaml_path)

    # Create experiment yaml file, using file within Habitat Lab repo as a template
    with open(exp_yaml_path) as f:
        exp_yaml_data = f.read().splitlines()

    for idx, i in enumerate(exp_yaml_data):
        if i.startswith("BASE_TASK_CONFIG_PATH:"):
            exp_yaml_data[idx] = f"BASE_TASK_CONFIG_PATH: '{new_task_yaml_path}'"
        elif i.startswith("TOTAL_NUM_STEPS:"):
            max_num_steps = 5e8 if args.control_type == "kinematic" else 5e7
            exp_yaml_data[idx] = f"TOTAL_NUM_STEPS: {max_num_steps}"
        elif i.startswith("TENSORBOARD_DIR:"):
            exp_yaml_data[idx] = f"TENSORBOARD_DIR:    '{os.path.join(dst_dir, 'tb')}'"
        elif i.startswith("NUM_ENVIRONMENTS:"):
            exp_yaml_data[idx] = f"NUM_ENVIRONMENTS: {args.num_environments}"
            if args.use_gray or args.use_gray_depth:
                exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 8"
            if args.outdoor or args.dataset == "coda":
                exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 1"
        elif i.startswith("SENSORS:"):
            if args.use_gray:
                exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
            elif args.use_gray_depth:
                exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR', 'SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
            if args.surface_normal:
                exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR', 'SPOT_SURFACE_NORMAL_SENSOR']"
        elif i.startswith("VIDEO_DIR:"):
            exp_yaml_data[
                idx
            ] = f"VIDEO_DIR:          '{os.path.join(dst_dir, 'video_dir')}'"
        elif i.startswith("EVAL_CKPT_PATH_DIR:"):
            exp_yaml_data[
                idx
            ] = f"EVAL_CKPT_PATH_DIR: '{os.path.join(dst_dir, 'checkpoints')}'"
        elif i.startswith("CHECKPOINT_FOLDER:"):
            exp_yaml_data[
                idx
            ] = f"CHECKPOINT_FOLDER:  '{os.path.join(dst_dir, 'checkpoints')}'"
        elif i.startswith("TXT_DIR:"):
            exp_yaml_data[
                idx
            ] = f"TXT_DIR:            '{os.path.join(dst_dir, 'txts')}'"
        elif i.startswith("    num_cnns:"):
            if args.two_cnns:
                exp_yaml_data[idx] = "    num_cnns: 2"
        elif i.startswith("      ENABLED_TRANSFORMS: [ ]"):
            if args.pepper_noise:
                exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['PEPPER_NOISE']"
            elif args.cutout_noise:
                exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['CUTOUT']"
            elif args.median_blur:
                exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['MEDIAN_BLUR']"
        elif i.startswith("        NOISE_PERCENT:"):
            exp_yaml_data[idx] = f"        NOISE_PERCENT: {args.noise_percent}"
        elif i.startswith("        KERNEL_SIZE:"):
            exp_yaml_data[idx] = f"        KERNEL_SIZE: {args.kernel_size}"
        elif i.startswith("        NUM_ITERS:"):
            exp_yaml_data[idx] = f"        NUM_ITERS: {args.num_iters}"
        elif i.startswith("    decoder_output:"):
            if args.surface_normal:
                exp_yaml_data[idx] = "    decoder_output: ['depth', 'surface_normals']"

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
    if args.constraint:
        slurm_data = slurm_data.replace(
            "# CONSTRAINT", "#SBATCH --constraint rtx_6000|a40"
        )
    slurm_path = os.path.join(dst_dir, experiment_name + ".sh")
    with open(slurm_path, "w") as f:
        f.write(slurm_data)
    print("Generated slurm job: " + slurm_path)

    if not args.x:
        # Submit slurm job
        cmd = "sbatch " + slurm_path
        subprocess.check_call(cmd.split(), cwd=dst_dir)
    else:
        print(slurm_data)

    print(
        f"\nSee output with:\ntail -F {os.path.join(dst_dir, experiment_name + '.err')}"
    )

# Evaluation
else:
    # Make sure folder exists
    assert os.path.isdir(dst_dir), f"{dst_dir} directory does not exist"
    os.makedirs(eval_dst_dir, exist_ok=True)
    with open(os.path.join(eval_dst_dir, "automate_job_cmd.txt"), "w") as f:
        f.write(automate_command)
    print(
        "Saved automate command: " + os.path.join(eval_dst_dir, "automate_job_cmd.txt")
    )

    # Create task yaml file, using file within Habitat Lab repo as a template
    with open(task_yaml_path) as f:
        eval_yaml_data = f.read().splitlines()

    for idx, i in enumerate(eval_yaml_data):
        if i.startswith("  CURRICULUM:"):
            eval_yaml_data[idx] = f"  CURRICULUM: {args.curriculum}"
        elif i.startswith("    RADIUS:"):
            eval_yaml_data[idx] = f"    RADIUS: {robot_radius}"
        elif i.startswith("  ROBOT:"):
            eval_yaml_data[idx] = f"  ROBOT: '{robot}'"
        elif i.startswith("      ROBOT_URDF:"):
            eval_yaml_data[idx] = f"      ROBOT_URDF: {robot_urdf}"
        elif i.startswith("  POSSIBLE_ACTIONS:"):
            if args.control_type == "dynamic":
                control_type = "DYNAMIC_VELOCITY_CONTROL"
            else:
                control_type = "VELOCITY_CONTROL"
            eval_yaml_data[idx] = f'  POSSIBLE_ACTIONS: ["{control_type}"]'
        elif i.startswith("    VELOCITY_CONTROL:"):
            if args.control_type == "dynamic":
                eval_yaml_data[idx] = "    DYNAMIC_VELOCITY_CONTROL:"
        elif i.startswith("    DYNAMIC_VELOCITY_CONTROL:"):
            if not args.control_type == "dynamic":
                eval_yaml_data[idx] = "    VELOCITY_CONTROL:"
        elif i.startswith("      TIME_STEP:"):
            if args.control_type == "dynamic":
                eval_yaml_data[idx] = "      TIME_STEP: 0.33"
        elif i.startswith("  SUCCESS_DISTANCE:"):
            eval_yaml_data[idx] = f"  SUCCESS_DISTANCE: {robot_goal}"
        elif i.startswith("    SUCCESS_DISTANCE:"):
            eval_yaml_data[idx] = f"    SUCCESS_DISTANCE: {robot_goal}"
        elif i.startswith("SEED:"):
            eval_yaml_data[idx] = f"SEED: {args.seed}"
        elif i.startswith("  DATA_PATH:"):
            if args.outdoor:
                data_path = "/coc/testnvme/jtruong33/data/datasets/ferst/{split}/{split}.json.gz"
                eval_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            if args.dataset == "coda":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test/coda_spot_test.json.gz"
                eval_yaml_data[idx] = "  DATA_PATH: {data_path}"
        elif i.startswith("      noise_multiplier:"):
            eval_yaml_data[idx] = f"      noise_multiplier: {args.noise_percent}"
    with open(new_eval_task_yaml_path, "w") as f:
        f.write("\n".join(eval_yaml_data))
    print("Created " + new_eval_task_yaml_path)

    # Edit the stored experiment yaml file
    with open(exp_yaml_path) as f:
        eval_exp_yaml_data = f.read().splitlines()

    for idx, i in enumerate(eval_exp_yaml_data):
        if i.startswith("BASE_TASK_CONFIG_PATH:"):
            eval_exp_yaml_data[
                idx
            ] = f"BASE_TASK_CONFIG_PATH: '{new_eval_task_yaml_path}'"
        elif i.startswith("TENSORBOARD_DIR:"):
            tb_dir = f"tb_eval_{args.control_type}"
            if args.ckpt != -1:
                tb_dir += f"_ckpt_{args.ckpt}"
            if args.video:
                tb_dir += "_video"
            eval_exp_yaml_data[
                idx
            ] = f"TENSORBOARD_DIR:    '{os.path.join(eval_dst_dir, 'tb_evals', tb_dir)}'"
        elif i.startswith("NUM_PROCESSES:"):
            eval_exp_yaml_data[idx] = "NUM_PROCESSES: 13"
        elif i.startswith("CHECKPOINT_FOLDER:"):
            eval_exp_yaml_data[
                idx
            ] = f"CHECKPOINT_FOLDER:  '{os.path.join(dst_dir, 'checkpoints')}'"
        elif i.startswith("EVAL_CKPT_PATH_DIR:"):
            if args.ckpt == -1:
                eval_exp_yaml_data[
                    idx
                ] = f"EVAL_CKPT_PATH_DIR: '{os.path.join(dst_dir, 'checkpoints')}'"
            else:
                eval_exp_yaml_data[
                    idx
                ] = f"EVAL_CKPT_PATH_DIR: '{os.path.join(dst_dir, 'checkpoints')}/ckpt.{args.ckpt}.pth'"
        elif i.startswith("TXT_DIR:"):
            txt_dir = f"txts_eval_{args.control_type}"
            if args.ckpt != -1:
                txt_dir += f"_ckpt_{args.ckpt}"
            eval_exp_yaml_data[
                idx
            ] = f"TXT_DIR:            '{os.path.join(eval_dst_dir, 'txts', txt_dir)}'"
        elif i.startswith("VIDEO_OPTION:"):
            if args.video:
                eval_exp_yaml_data[idx] = "VIDEO_OPTION: ['disk']"
            else:
                eval_exp_yaml_data[idx] = "VIDEO_OPTION: []"
        elif i.startswith("NUM_ENVIRONMENTS:"):
            if args.use_gray or args.use_gray_depth:
                eval_exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 8"
            if args.outdoor or args.dataset == "coda":
                eval_exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 1"
        elif i.startswith("SENSORS:"):
            if args.video:
                eval_exp_yaml_data[
                    idx
                ] = "SENSORS: ['RGB_SENSOR', 'SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR']"
            if args.use_gray:
                eval_exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
                if args.video:
                    eval_exp_yaml_data[
                        idx
                    ] = "SENSORS: ['RGB_SENSOR', 'SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
        elif i.startswith("VIDEO_DIR:"):
            video_dir = (
                "video_dir"
                if args.ckpt == -1
                else f"video_dir_{args.control_type}_ckpt_{args.ckpt}"
            )
            eval_exp_yaml_data[
                idx
            ] = f"VIDEO_DIR:          '{os.path.join(eval_dst_dir, 'videos', video_dir)}'"
        elif i.startswith("    SPLIT:"):
            if args.dataset == "hm3d":
                eval_exp_yaml_data[idx] = "    SPLIT: val"
            elif args.dataset == "ferst_20m":
                eval_exp_yaml_data[idx] = "    SPLIT: val_20m"
        elif i.startswith("    num_cnns:"):
            if args.two_cnns:
                eval_exp_yaml_data[idx] = "    num_cnns: 2"
        elif i.startswith("      ENABLED_TRANSFORMS: [ ]"):
            if args.pepper_noise:
                eval_exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['PEPPER_NOISE']"
            elif args.cutout_noise:
                eval_exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['CUTOUT']"
            elif args.median_blur:
                eval_exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['MEDIAN_BLUR']"
        elif i.startswith("        NOISE_PERCENT:"):
            eval_exp_yaml_data[idx] = f"        NOISE_PERCENT: {args.noise_percent}"
        elif i.startswith("        KERNEL_SIZE:"):
            eval_exp_yaml_data[idx] = f"        KERNEL_SIZE: {args.kernel_size}"
        elif i.startswith("        NUM_ITERS:"):
            eval_exp_yaml_data[idx] = f"        NUM_ITERS: {args.num_iters}"

    if os.path.isdir(tb_dir):
        response = input(
            f"{tb_dir} directory already exists. Delete, continue, or abort? [d/c/A]: "
        )
        if response == "d":
            print(f"Deleting {tb_dir}")
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
    slurm_path = os.path.join(eval_dst_dir, eval_experiment_name + ".sh")
    with open(slurm_path, "w") as f:
        f.write(slurm_data)
    print("Generated slurm job: " + slurm_path)

    if not args.x:
        # Submit slurm job
        cmd = "sbatch " + slurm_path
        subprocess.check_call(cmd.split(), cwd=dst_dir)
    else:
        print(slurm_data)
    print(
        f"\nSee output with:\ntail -F {os.path.join(eval_dst_dir, eval_experiment_name + '.err')}"
    )
