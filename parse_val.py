"""
Given an agent, return stats on both the validation set and the test set
Return:
- The best checkpoint from the validation set, in terms of SCT
- Stats on the test episodes corresponding to the best checkpoint
"""
import argparse
import glob
import json
import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.stats
import tqdm


# NUM_EPISODES = 1070
# CSV header: id,reward,distance_to_goal,success,spl,sct,steps
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_dir", help="dir of .csvs of a particular agent")
    parser.add_argument("-n", "--num-episodes", type=int, default=1070)
    parser.add_argument("-c", "--checkpoint", type=int, default=-1)
    parser.add_argument("-l", "--limit", type=int, default=1e9, help="step limit")
    args = parser.parse_args()
    global NUM_EPISODES
    NUM_EPISODES = args.num_episodes
    S = Stats()
    pprint("Determining best checkpoint...")
    best_csv_succ, best_csv_spl, last_csv = S.get_best_checkpoint(
        args.agent_dir,
        split="all",
        ckpt=args.checkpoint,
        step_limit=args.limit,
    )

    S.print_stats(best_csv_succ, "SUCC")
    S.print_stats(best_csv_spl, "SPL")
    S.print_stats(last_csv, "LAST")


class Stats(object):
    def return_successful_eps(self, csv, split):
        SPLIT = list(range(NUM_EPISODES))
        with open(csv) as f:
            df = pandas.read_csv(f)
        ret = [
            episode_id
            for episode_id, success in zip(df.id, df.success)
            if float(success) == 1.0 and episode_id in SPLIT
        ]
        return ret

    def return_unsuccessful_eps(self, csv, split):
        SPLIT = list(range(NUM_EPISODES))
        with open(csv) as f:
            df = pandas.read_csv(f)
        ret = [
            episode_id
            for episode_id, success in zip(df.id, df.success)
            if float(success) == 0.0 and episode_id in SPLIT
        ]
        return ret

    def plot_episode_distance(self, df):
        episode_dist = df.episode_distance
        # print("episode distance: ", episode_dist)
        fig, ax = plt.subplots()
        df.episode_distance.hist(legend=True)
        df["episode_distance_success"] = df[df.success == 1.0].episode_distance
        df.episode_distance_success.hist(legend=True)
        plt.xlabel("episode distance (m)")
        plt.ylabel("frequency")
        plt.title("Outdoor Env Episode Distance Frequency")
        fig.savefig("/coc/testnvme/jtruong33/habitat_spot/habitat-lab/hist.png")
        print("min episode dist: ", df["episode_distance"].min())
        print("max episode dist: ", df["episode_distance"].max())
        print(" ")
        print(
            "min successful episode dist: ",
            df["episode_distance_success"].min(),
        )
        print(
            "max successful episode dist: ",
            df["episode_distance_success"].max(),
        )

    """
    Given a CSV DataFrame for a checkpoint and a given split,
    calculate stats.
    """

    def calculate_stats(self, csv, split, eps=None, eps_ignore=None):
        num_lines = sum(1 for line in open(csv))
        # print(f'{csv} has {num_lines} unique episode ids.')
        print("NUM_LINES: ", num_lines, "CKPT: ", csv)
        # assert num_lines == NUM_EPISODES + 1
        with open(csv) as f:
            df = pandas.read_csv(f)
        SPLIT = list(range(NUM_EPISODES))
        if eps is not None:
            SPLIT = [i for i in SPLIT if i in eps]
        if eps_ignore is not None:
            SPLIT = [i for i in SPLIT if i not in eps_ignore]
        df_split = df[[ep_id in SPLIT for ep_id in df.id]]
        # self.plot_episode_distance(df_split)
        stats = {}
        # id,reward,distance_to_goal,success,spl,steps,collisions,soft_spl,episode_distance,num_actions
        stats["spl_stats"] = self.mean_confidence_interval(np.array(df_split.spl))
        stats["dist_stats"] = self.mean_confidence_interval(df_split.distance_to_goal)
        stats["success_stats"] = self.mean_confidence_interval(
            np.array(df_split.success)
        )
        stats["episode_distance_stats"] = self.mean_confidence_interval(
            np.array(df_split.episode_distance)
        )
        stats["success_episode_distance_stats"] = self.mean_confidence_interval(
            np.array(df_split[df_split.success == 1.0].episode_distance)
        )
        stats["fail_episode_distance_stats"] = self.mean_confidence_interval(
            np.array(df_split[df_split.success == 0.0].episode_distance)
        )
        try:
            stats["num_actions_stats"] = self.mean_confidence_interval(
                np.array(df_split.steps)
            )
        except:
            stats["num_actions_stats"] = self.mean_confidence_interval(
                np.array(df_split.steps0)
            )
        return stats

    """
    Get csvs that correspond to a step amount <= step_limit
    """

    def get_valid_csvs(self, agent_dir, ckpt=None, step_limit=2e8):
        # ckpt_num = int(step_limit / 5e6)
        ckpt_num = np.inf
        agent_name = os.path.basename(agent_dir)
        csvs = []
        ckpt_to_idx = lambda x: int(x.split("_")[-1][:-4])
        for csv in sorted(glob.glob(os.path.join(agent_dir, "*csv")), key=ckpt_to_idx):
            ckpt_ind = int(os.path.basename(csv).split(".")[0][len("ckpt_") :])
            if ckpt != -1:
                if ckpt_ind == ckpt:
                    csvs.append(csv)
            else:
                if ckpt_ind <= ckpt_num:
                    csvs.append(csv)
        return csvs

    """
    Given path to agent directory of csvs, return the checkpoint
    index corresponding to highest.
    Only consider csvs corresponding to ckpts trained on <= 200M steps.
    """

    def get_best_checkpoint(self, agent_dir, split="all", ckpt=-1, step_limit=2e8):
        agent_name = os.path.basename(agent_dir)
        csvs = self.get_valid_csvs(agent_dir, ckpt=ckpt, step_limit=step_limit)
        mean_successes, mean_spls, mean_colls = [], [], []
        valid_csvs = []
        csv_nums = []
        invalid_csv_nums = []
        for csv in tqdm.tqdm(csvs):
            ckpt_ind = int(os.path.basename(csv).split(".")[0][len("ckpt_") :])
            try:
                stats = self.calculate_stats(csv, split=split)
                csv_nums.append(ckpt_ind)
                valid_csvs.append(csv)
            except AssertionError:
                pprint(
                    f"[ERROR]: {csv} which does not have {NUM_EPISODES} num episodes. Skipping."
                )
                invalid_csv_nums.append(ckpt_ind)
                continue
            mean_successes.append((stats["success_stats"][0], csv))
            mean_spls.append((stats["spl_stats"][0], csv))
        best_csv_succ = max(mean_successes)[1]
        best_csv_spl = max(mean_spls)[1]
        last_csv = valid_csvs[-1]
        print("valid csvs: ", sorted(csv_nums))
        print("invalid csvs: ", sorted(invalid_csv_nums))
        return best_csv_succ, best_csv_spl, last_csv

    """
    Return the mean and 95% confidence interval
    """

    def mean_confidence_interval(self, data):
        a = np.array(data, dtype=np.float32).copy()
        mu, std = np.mean(a), np.std(a)
        interval = 1.960 * std / np.sqrt(len(a))
        # interval = std
        assert not np.isnan(mu)
        return mu, interval

    def print_stats(self, csv, ty):
        all_stats = self.calculate_stats(csv, split="all")
        best_ckpt_idx = csv.split("/")[-1].split("_")[-1][:-4]
        ckpt_pth = os.path.abspath(
            csv.split("eval/")[0] + f"checkpoints/ckpt.{best_ckpt_idx}.pth"
        )
        pprint(f"Best checkpoint [{ty}] stats: {ckpt_pth}")
        print(
            "Best Ckpt, Success, SPL, # Steps, # Collisions, Dist2Goal, Episode Distance"
        )
        print(
            ("{}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t").format(
                best_ckpt_idx,
                all_stats["success_stats"][0],
                all_stats["spl_stats"][0],
                all_stats["num_actions_stats"][0],
                0.00,
                all_stats["dist_stats"][0],
                all_stats["episode_distance_stats"][0],
            )
        )
        print("")


if __name__ == "__main__":
    main()
