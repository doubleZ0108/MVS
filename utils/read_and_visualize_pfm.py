# -*- coding: utf-8 -*-
# @Description: read depth(.pfm) and visualized
# @Author: doubleZ, Peking University
# @Date: 2022-10-31 19:04:43
# @LastEditTime: 2022-10-31 19:04:43


import numpy as np
import re
import sys, os
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def visual_depth(depth, color_reverse=False):
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.xticks([]), plt.yticks([]), plt.axis('off')
    if color_reverse:
        plt.imshow(depth, 'viridis_r', vmin=500, vmax=830)
    else:
        plt.imshow(depth, 'viridis')


def read_depth(filename):
    depth = read_pfm(filename)[0]
    return np.array(depth, dtype=np.float32)


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale