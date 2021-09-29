#!/usr/bin/env python3

import argparse
import glob
import gzip
import json
import multiprocessing
import os
from os import path as osp

import tqdm

import habitat
from habitat.datasets.pointnav.pointnav_generator import (
    generate_pointnav_episode,
)

parser = argparse.ArgumentParser()
parser.add_argument('config_yaml')
parser.add_argument('out_dir')
parser.add_argument('num_episodes', type=int)
parser.add_argument('glb_dir')
parser.add_argument('-v', '--val', action='store_true')
args = parser.parse_args()

# Task config yaml file. Needed for agent radius and height, not sure what else...
CONFIG_YAML = args.config_yaml

# Folder where json.gz files will be saved:
TRAIN_EP_DIR = args.out_dir

NUM_EPISODES_PER_SCENE = args.num_episodes

VAL = [
    'Cantwell', 'Denmark', 'Eastville', 'Edgemere', 'Elmira', 'Eudora', 'Greigsville', 'Mosquito', 'Pablo',
    'Ribera', 'Sands', 'Scioto', 'Sisters', 'Swormville'
]

TRAIN = [
    'Adrian', 'Albertville', 'Anaheim', 'Andover', 'Angiola', 'Annawan', 'Applewold', 'Arkansaw',
    'Avonia', 'Azusa', 'Ballou', 'Beach', 'Bolton', 'Bowlus', 'Brevort', 'Capistrano',
    'Colebrook', 'Convoy', 'Cooperstown', 'Crandon', 'Delton', 'Dryville', 'Dunmor', 'Eagerville', 'Goffs',
    'Hainesburg', 'Hambleton', 'Haxtun', 'Hillsdale', 'Hometown', 'Hominy', 'Kerrtown', 'Maryhill', 'Mesic',
    'Micanopy', 'Mifflintown', 'Mobridge', 'Monson', 'Mosinee', 'Nemacolin', 'Nicut', 'Nimmons',
    'Nuevo', 'Oyens', 'Parole', 'Pettigrew', 'Placida', 'Pleasant', 'Quantico', 'Rancocas',
    'Reyno', 'Roane', 'Roeville', 'Rosser', 'Roxboro', 'Sanctuary', 'Sasakwa', 'Sawpit',
    'Seward', 'Shelbiana', 'Silas', 'Sodaville', 'Soldier', 'Spencerville', 'Spotswood', 'Springhill',
    'Stanleyville', 'Stilwell', 'Stokes', 'Sumas', 'Superior', 'Woonsocket',
]

split = VAL if args.val else TRAIN

# List of paths to glb files
SCENES = list(filter(
    lambda glb_path: any(list(
        map(
            lambda scene: scene in osp.basename(glb_path),
            split,
        )
    )),
    glob.glob(osp.abspath(osp.join(args.glb_dir, '*.glb'))),
))

# Pass absolute path to the scene glb
def _generate_fn(scene):
    scene_key = osp.basename(scene)
    out_file = osp.join(
        TRAIN_EP_DIR, scene_key.split(".")[0]+".json.gz"
    )

    if osp.isfile(out_file):
        return

    cfg = habitat.get_config(CONFIG_YAML)
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim, NUM_EPISODES_PER_SCENE,
            is_gen_shortest_path=False,
            geodesic_to_euclid_min_ratio=1.1
        )
    )
    for ep in dset.episodes:
        ep.scene_id = osp.join('gibson', scene_key)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())

def generate_gibson_large_dataset():
    scenes = SCENES

    with multiprocessing.Pool(20) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        for _ in pool.imap_unordered(_generate_fn, scenes):
            pbar.update()

if __name__ == "__main__":
    # print('\n'.join(SCENES))
    generate_gibson_large_dataset()