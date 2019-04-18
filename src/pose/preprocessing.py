# coding=utf-8
# Created by utils on 2019-03-05 11:02
# Copyright © 2019 Alan. All rights reserved.
import cv2
import os
import shutil
import numpy as np
from torchvision import datasets, models, transforms


def video_extract(video_path='./IMG_3885.mov', extract_folder='./extract_folder', frequency=1):
    """
    extract the video from the given video path
    :param video_path:
    :param extract_folder: the image folder for the snapped image
    :param frequency: for a certain frequency, a image is snapped from the video
    """
    print("Exracting the pictures from the video...")
    index = 1
    try:
        shutil.rmtree(extract_folder)
    except OSError:
        pass
    os.mkdir(extract_folder)

    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        exit(1)
    count = 1
    while True:
        _, frame = video.read()
        if frame is None:
            break
        if count % frequency == 0:
            save_path = "{}/{:>03d}.jpg".format(extract_folder, index)
            cv2.imwrite(save_path, frame)
            index += 1
        count += 1
    video.release()
    print("Totally save {:d} pics".format(index-1))
    return 0


def cut_joint_image(img_orig, joint_list, person_to_joint_assoc, cut_size):
    """
    cut joint image to form a posebox
    :param img_orig:
    :param joint_list:
    :param person_to_joint_assoc:
    :param cut_size: default to be 32 defined in LEN
    :return: posebox, coordinators, missing_part(for posebox refine)
    """
    # data normalization
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    cut_size = cut_size // 2
    height, width, channel = img_orig.shape
    # initialize the information recorder
    people_num = len(person_to_joint_assoc)
    coord = np.zeros((people_num, 4))
    coord[:, 0] = 3840
    coord[:, 2] = 3840
    new_imag = np.zeros((people_num, cut_size * 2 * 6, cut_size * 2 * 3, 3))
    missing_part = []
    index = 0
    # compose poseboxes
    for person_joint_info in person_to_joint_assoc:
        temp_missing = []
        for i in range(18):
            joint_indices = person_joint_info[i].astype(int)
            if joint_indices == -1:
                temp_missing.append(i)
                continue

            # get coordinate for pose box and square
            else:
                joint_coords = joint_list[joint_indices, 0:3]
                #
                if int(joint_coords[1] - cut_size) >= 0:
                    A = int(joint_coords[1] - cut_size)
                else:
                    A = 0
                if int(joint_coords[1] + cut_size) <= height:
                    B = int(joint_coords[1] + cut_size)
                else:
                    B = height
                if int(joint_coords[0] - cut_size) >= 0:
                    C = int(joint_coords[0] - cut_size)
                else:
                    C = 0
                if int(joint_coords[0] + cut_size) <= width:
                    D = int(joint_coords[0] + cut_size)
                else:
                    D = width

                # get periphery square
                if A+cut_size < coord[index, 0]:
                    coord[index, 0] = A+cut_size

                if B-cut_size > coord[index, 1]:
                    coord[index, 1] = B-cut_size

                if C+cut_size < coord[index, 2]:
                    coord[index, 2] = C+cut_size

                if D-cut_size > coord[index, 3]:
                    coord[index, 3] = D-cut_size
                # 0:x-max,1:y-max,2:x-min,3:y-min
                #
                cropped = img_orig[A:B, C:D]

                #
                x = cropped.shape[0]
                y = cropped.shape[1]
                new_imag[index, (i % 6) * (cut_size * 2):x + (i % 6) * (cut_size * 2),\
                (i//6) * (cut_size * 2):y + (i//6) * (cut_size * 2)] = cropped
        new_imag[index] = np.transpose(data_transforms(new_imag[index]).numpy(), (1 , 2, 0))
        index += 1
        missing_part.append(temp_missing)

    return new_imag, coord.astype(int), missing_part


def denoise(person_to_joint_assoc, val=10):
    """
    only preserve the image less than the valve value skeleton as a human
    :param person_to_joint_assoc:
    :param val: valve value for human recognition
    :return: denoisd person to joint association
    """
    if len(person_to_joint_assoc) == 0:
        return []
    detected_num, dim = person_to_joint_assoc.shape
    noise_list = []
    for i in range(detected_num):
        counter = 0
        for j in range(dim):
            joint_indices = person_to_joint_assoc[i, j].astype(int)
            if joint_indices == -1:
                continue
            else:
                counter += 1
        if counter < val:
            # only have the person_to_joint_assoc when over val joints is detected
            noise_list.append(i)

    person_to_joint_assoc = np.delete(person_to_joint_assoc, noise_list, axis=0)
    return person_to_joint_assoc
