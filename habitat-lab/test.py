import os
import argparse
import ast

parser = argparse.ArgumentParser()

# Training
parser.add_argument('-r','--robots', nargs='+', required=True)
parser.add_argument('-vx','--lin_vel_ranges', nargs='+', required=False)
args = parser.parse_args()

robots = args.robots
print(robots)

robot_heights_dict = {'A1': [0.0, 0.40, -0.1335],
                          'AlienGo': [0.0, 0.47, -0.3235],
                          'Daisy': [0.0, 0.425, -0.1778],
                          'Spot': [0.0, 0.30, -0.09]
                         }

robots_heights = [robot_heights_dict[robot] for robot in robots]
print("    POSITION: {}".format(robots_heights[0]))

# lin_vel_ranges = [ast.literal_eval(n) for n in args.lin_vel_ranges]

# print(lin_vel_ranges)