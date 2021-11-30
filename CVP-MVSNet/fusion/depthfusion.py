# Modified version of the original depthfusion.py from Yao Yao.
# Convert our output to Gipuma format for post-processing.
# By: Jiayu Yang
# Date: 2019-11-05


from __future__ import print_function

import argparse
import os
import time
import glob
import random
import math
import re
import sys
import shutil
from struct import *

import cv2
import numpy as np

import pylab as plt
from preprocess import * 

import pdb

def read_gipuma_dmb(path):
    '''read Gipuma .dmb format image'''

    with open(path, "rb") as fid:
        
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]
        
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channel), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def write_gipuma_dmb(path, image):
    '''write Gipuma .dmb format image'''
    
    image_shape = np.shape(image)
    width = image_shape[1]
    height = image_shape[0]
    if len(image_shape) == 3:
        channels = image_shape[2]
    else:
        channels = 1

    if len(image_shape) == 3:
        image = np.transpose(image, (2, 0, 1)).squeeze()

    with open(path, "wb") as fid:
        # fid.write(pack(1))
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)
    return 

def mvsnet_to_gipuma_dmb(in_path, out_path):
    '''convert mvsnet .pfm output to Gipuma .dmb format'''
    
    image = load_pfm(open(in_path))
    write_gipuma_dmb(out_path, image)

    return 

def mvsnet_to_gipuma_cam(in_path, out_path):
    '''convert mvsnet camera to gipuma camera format'''

    cam = load_cam(open(in_path))

    extrinsic = cam[0:4][0:4][0]
    intrinsic = cam[0:4][0:4][1]
    intrinsic[3][0] = 0
    intrinsic[3][1] = 0
    intrinsic[3][2] = 0
    intrinsic[3][3] = 0

    projection_matrix = np.matmul(intrinsic, extrinsic)
    projection_matrix = projection_matrix[0:3][:]
    
    f = open(out_path, "w")
    for i in range(0, 3):
        for j in range(0, 4):
            f.write(str(projection_matrix[i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()

    return

def fake_gipuma_normal(in_depth_path, out_normal_path):
    
    depth_image = read_gipuma_dmb(in_depth_path)
    image_shape = np.shape(depth_image)

    normal_image = np.ones_like(depth_image)
    normal_image = np.reshape(normal_image, (image_shape[0], image_shape[1], 1))
    normal_image = np.tile(normal_image, [1, 1, 3])
    normal_image = normal_image / 1.732050808

    mask_image = np.squeeze(np.where(depth_image > 0, 1, 0))
    mask_image = np.reshape(mask_image, (image_shape[0], image_shape[1], 1))
    mask_image = np.tile(mask_image, [1, 1, 3])
    mask_image = np.float32(mask_image)

    normal_image = np.multiply(normal_image, mask_image)
    normal_image = np.float32(normal_image)

    write_gipuma_dmb(out_normal_path, normal_image)
    return 

def mvsnet_to_gipuma(scan_folder, scan, dtu_test_root, gipuma_point_folder):
    
    image_folder = os.path.join(dtu_test_root, 'Rectified', scan)
    cam_folder = os.path.join(dtu_test_root, 'Cameras')
    depth_folder = os.path.join(scan_folder, 'depth_est')

    gipuma_cam_folder = os.path.join(gipuma_point_folder, 'cams')
    gipuma_image_folder = os.path.join(gipuma_point_folder, 'images')
    if not os.path.isdir(gipuma_point_folder):
        os.mkdir(gipuma_point_folder)
    if not os.path.isdir(gipuma_cam_folder):
        os.mkdir(gipuma_cam_folder)
    if not os.path.isdir(gipuma_image_folder):
        os.mkdir(gipuma_image_folder)

    # convert cameras 
    for view in range(0,49):
        in_cam_file = os.path.join(cam_folder, "{:08d}_cam.txt".format(view))
        out_cam_file = os.path.join(gipuma_cam_folder, "{:08d}.png.P".format(view))
        mvsnet_to_gipuma_cam(in_cam_file, out_cam_file)

    # copy images to gipuma image folder    
    for view in range(0,49):
        in_image_file = os.path.join(image_folder, "rect_{:03d}_3_r5000.png".format(view+1))# Our image start from 1
        out_image_file = os.path.join(gipuma_image_folder, "{:08d}.png".format(view))
        shutil.copy(in_image_file, out_image_file)

    # convert depth maps and fake normal maps
    gipuma_prefix = '2333__'
    for view in range(0,49):

        sub_depth_folder = os.path.join(gipuma_point_folder, gipuma_prefix+"{:08d}".format(view))
        if not os.path.isdir(sub_depth_folder):
            os.mkdir(sub_depth_folder)
        in_depth_pfm = os.path.join(depth_folder, "{:08d}_prob_filtered.pfm".format(view))
        out_depth_dmb = os.path.join(sub_depth_folder, 'disp.dmb')
        fake_normal_dmb = os.path.join(sub_depth_folder, 'normals.dmb')
        mvsnet_to_gipuma_dmb(in_depth_pfm, out_depth_dmb)
        fake_gipuma_normal(out_depth_dmb, fake_normal_dmb)

def probability_filter(scan_folder, prob_threshold):
    depth_folder = os.path.join(scan_folder, 'depth_est')
    prob_folder = os.path.join(scan_folder, 'confidence')
    
    # convert cameras 
    for view in range(0,49):
        init_depth_map_path = os.path.join(depth_folder, "{:08d}.pfm".format(view)) # New dataset outputs depth start from 0.
        prob_map_path = os.path.join(prob_folder, "{:08d}.pfm".format(view)) # Same as above
        out_depth_map_path = os.path.join(depth_folder, "{:08d}_prob_filtered.pfm".format(view)) # Gipuma start from 0

        depth_map = load_pfm(open(init_depth_map_path))
        prob_map = load_pfm(open(prob_map_path))
        depth_map[prob_map < prob_threshold] = 0
        write_pfm(out_depth_map_path, depth_map)

def depth_map_fusion(point_folder, fusibile_exe_path, disp_thresh, num_consistent):

    cam_folder = os.path.join(point_folder, 'cams')
    image_folder = os.path.join(point_folder, 'images')
    depth_min = 0.001
    depth_max = 100000
    normal_thresh = 360

    cmd = fusibile_exe_path
    cmd = cmd + ' -input_folder ' + point_folder + '/'
    cmd = cmd + ' -p_folder ' + cam_folder + '/'
    cmd = cmd + ' -images_folder ' + image_folder + '/'
    cmd = cmd + ' --depth_min=' + str(depth_min)
    cmd = cmd + ' --depth_max=' + str(depth_max)
    cmd = cmd + ' --normal_thresh=' + str(normal_thresh)
    cmd = cmd + ' --disp_thresh=' + str(disp_thresh)
    cmd = cmd + ' --num_consistent=' + str(num_consistent)
    print (cmd)
    os.system(cmd)

    return 

if __name__ == '__main__':

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtu_test_root', type=str, default = '')
    parser.add_argument('--depth_folder', type=str, default = '')
    parser.add_argument('--out_folder', type=str, default = '')
    parser.add_argument('--fusibile_exe_path', type=str, default = './fusibile')
    parser.add_argument('--prob_threshold', type=float, default = '0.8')
    parser.add_argument('--disp_threshold', type=float, default = '0.13')
    parser.add_argument('--num_consistent', type=float, default = '3')
    args = parser.parse_args()

    dtu_test_root = args.dtu_test_root
    depth_folder = args.depth_folder
    out_folder = args.out_folder
    fusibile_exe_path = args.fusibile_exe_path
    prob_threshold = args.prob_threshold
    disp_threshold = args.disp_threshold
    num_consistent = args.num_consistent

    # Read test list
    testlist = "./scan_list_test.txt"
    with open(testlist) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]

    # Fusion
    for scan in scans:

        scan_folder = os.path.join(depth_folder, scan)
        fusibile_workspace = os.path.join(depth_folder, out_folder, scan)

        if not os.path.isdir(os.path.join(depth_folder, out_folder)):
            os.mkdir(os.path.join(depth_folder, out_folder))

        if not os.path.isdir(fusibile_workspace):
            os.mkdir(fusibile_workspace)

        # probability filtering
        print ('filter depth map with probability map')
        probability_filter(scan_folder, prob_threshold)

        # convert to gipuma format
        print ('Convert mvsnet output to gipuma input')
        mvsnet_to_gipuma(scan_folder, scan, dtu_test_root, fusibile_workspace)

        # depth map fusion with gipuma 
        print ('Run depth map fusion & filter')
        depth_map_fusion(fusibile_workspace, fusibile_exe_path, disp_threshold, num_consistent)
