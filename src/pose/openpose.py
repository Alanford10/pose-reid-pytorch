import os
import re
import sys
import cv2
import shutil
import glob
import math
import time
import scipy
import sqlite3
import argparse
import matplotlib
import numpy as np
import time
import pylab as plt
import torchvision
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter
from src.pose.network.rtpose_vgg import get_model
from src.pose.network.post import *
from src.pose.training.datasets.coco_data.preprocessing import (inception_preprocess,
                                     rtpose_preprocess,
                                     ssd_preprocess, vgg_preprocess)
from src.pose.network import im_transform
from src.pose.evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat
from src.pose.preprocessing import *

LEN = 32


def pose_batch_processing(extract_folder='./extract_folder/',
                          pose_folder='./pose_result/'):

    weight_name = './network/weight/pose_model.pth'
    model = get_model('vgg19').cuda()
    model.load_state_dict(torch.load(weight_name))
    # model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()

    img_name = sorted(os.listdir(extract_folder))

    try:
        shutil.rmtree(pose_folder)
    except OSError:
        pass
    os.mkdir(pose_folder)

    with torch.no_grad():
        for i in img_name:
            start_time = time.time()
            img_path = extract_folder + i
            frame_counter = 1
            oriImg = cv2.imread(img_path)  # B,G,R order
            shape_dst = np.min(oriImg.shape[0:2])
            # Get results of original image
            multiplier = get_multiplier(oriImg)

            orig_paf, orig_heat = get_outputs(
                multiplier, oriImg, model,  'rtpose')
            # Get results of flipped image
            swapped_img = oriImg[:, ::-1, :]
            flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
                                                model, 'rtpose')
            # compute averaged heatmap and paf
            paf, heatmap = handle_paf_and_heat(
                orig_heat, flipped_heat, orig_paf, flipped_paf)

            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            canvas, to_plot, candidate, subset = decode_pose(
                oriImg, param, heatmap, paf)

            cv2.imwrite(pose_folder+i, to_plot)

            people_num = len(subset)
            people_list = cut_joint_image(oriImg, 16, 32, candidate, subset)
            PIL_to_plot = Image.fromarray(cv2.cvtColor(to_plot, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(PIL_to_plot)
            draw.text((0, 0), "People number: " + str(people_num), (255, 255, 255))
            draw.text((0, 20), "Frame: " + str(frame_counter) + "/" + str(len(img_name)), (255, 255, 255))
            draw.text((0, 40), "Time: " + str(round(time.time() - start_time, 2)) + "sec", (255, 255, 255))
            PIL_to_plot.save(pose_folder + i)
            print(i + " completed in " + str(round(time.time() - start_time, 2)), " s...")
            frame_counter += 1


def pose_si_processing(model, frame, pose_val, plot_skeleton):
    oriImg = frame
    # shape_dst = np.min(oriImg.shape[0:2])
    # Get results of original image
    multiplier = get_multiplier(oriImg)
    orig_paf, orig_heat = get_outputs(
        multiplier, oriImg, model,  'rtpose')
    # Get results of flipped image
    swapped_img = oriImg[:, ::-1, :]
    flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
                                        model, 'rtpose')
    # compute averaged heatmap and paf
    paf, heatmap = handle_paf_and_heat(
        orig_heat, flipped_heat, orig_paf, flipped_paf)
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}

    to_plot, candidate, subset = decode_pose(
        oriImg, param, heatmap, paf, pose_val, plot_skeleton)
    # canvas, to_plot, candidate, subset = decode_pose(
    #     oriImg, param, heatmap, paf, pose_val, plot_skeleton)
    # candidate, subset = decode_pose(
    #     oriImg, param, heatmap, paf, pose_val)
    # print("heatmap: ", heatmap.shape)
    posebox, coord, joint_coords_db, missing_part = cut_joint_image(oriImg, candidate, subset, cut_size=LEN)
    return to_plot, posebox, coord, missing_part
