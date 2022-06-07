# -*-coding:utf-8 -*-

from genericpath import exists
import numpy as np
import open3d as o3d
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--ply_path', type=str, default='outputs/xx/yy_fusion_plys')
args = parser.parse_args()

if not os.path.exists(f"{args.ply_path}/filtered"):
    os.makedirs(f"{args.ply_path}/filtered")


scans_with_ratio = {
    '[scene-name]': 1.5,
}


for scan, ratio in scans_with_ratio.items():
    if os.path.exists(f"{args.ply_path}/{scan}.ply"):
        print(f"Starting filtering: {scan}...")
        
        ply = o3d.io.read_point_cloud(f"{args.ply_path}/{scan}.ply")

        ply_filtered, ind = ply.remove_statistical_outlier(nb_neighbors=20, std_ratio=ratio)
        # ply_filtered, ind = ply.remove_radius_outlier(nb_points=16, radius=5)

        o3d.io.write_point_cloud(f"{args.ply_path}/filtered/{scan}.ply", ply_filtered)

        print(f"Statistical Filter for: {scan} done!")
