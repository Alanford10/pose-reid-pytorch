# coding=utf-8
# Created by utils on 2019-03-11 21:41
# Copyright © 2019 Alan. All rights reserved.
import numpy as np
import cv2
import math


class Identity:
    def __init__(self, posebox, id_vector, missing_part):
        self.posebox = posebox
        self.id_vector = id_vector
        self.missing_part = missing_part


def match(database, posebox, id_vec, missing_part, val):
    db_len = len(database)
    id_len = len(id_vec)
    id_relation = []
    # id_relation = np.zeros(id_len)
    # id_relation[:] = -1
    # record the currently matched value
    id_pipeline = []
    # list for recording value less than val
    point_distance = []

    for j in range(id_len):
        for i in range(db_len):
            current_distance = np.sum(np.abs(databse[i].id_vector - id_vec[j]))
            point_distance.append((current_distance, i, j))

    point_distance.sort(reverse=True)
    distance_len = len(point_distance)
    # to judge that a picture cannot be two identical people
    for i in range(distance_len):
        distance, db_id, id_id = point_distance[distance_len - 1 - i]
        # distance less than the svd_val value, deemed as the same identity
        if distance < val:
            # id not belonged
            if id_id not in id_pipeline:
                # make sure db_id and id_id are paired only once
                id_pipeline.append(id_id)
                id_relation.append(db_id)
                # posebox success from the database
                # identity vector update
                # default coefficient: 0.5
                # database[db_id].id_vector = (database[db_id].id_vector + id_vec[id_id, :]) / 2
            else:
                continue
        # distance more than then svd_val value, deemed as different identity
        else:
            # id not belonged
            if id_id not in id_pipeline:
                continue
            else:
                id_pipeline.append(id_id)
                database.append(Identity(posebox, id_vec, missing_part[id_id]))
                id_relation.append(len(database) - 1)
    return database, id_relation


def draw_id(skeleton, coords, id_relation):
    scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    index = 0
    for coord in coords:
        color = (227 + 200 * math.sin(id_relation[index]), \
                 227 + 200 * math.sin(3 * id_relation[index] - 2), \
                 227 + 200 * math.cos(2 * id_relation[index]))
        # draw ids
        text_y = (coord[0] + coord[1]) // 2
        text_x = (coord[2] + coord[3]) // 2
        content = "id-" + str(int(id_relation[index]))
        cv2.putText(skeleton, content, (text_x, text_y), font, scale, color, 2)
        half_h = (coord[1] - coord[0])//2
        half_w = (coord[3] - coord[2])//2
        xmin = text_x - half_w
        ymin = text_y - half_h
        xmax = text_x + half_w
        ymax = text_y + half_h
        # draw rectangles
        skeleton = cv2.rectangle(skeleton, (xmin, ymin), (xmax, ymax), color)
        index += 1

    return skeleton


def put_text(skeleton, people_num, frame_counter, pose_time, reid_time):

    scale = 0.5
    color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.LINE_AA
    content = "People number: " + str(people_num)
    cv2.putText(skeleton, content, (10, 10), font, scale, color)

    content = "Frame: " + str(frame_counter)
    cv2.putText(skeleton, content, (10, 30), font, scale, color)

    content = "Pose Time: " + str(round(pose_time, 2)) + 's'
    cv2.putText(skeleton, content, (10, 50), font, scale, color)

    content = "Reid Time: " + str(round(reid_time, 2)) + 's'
    cv2.putText(skeleton, content, (10, 70), font, scale, color)

    return skeleton


def replace(dest_img, ori_img, dest_pos, pos, flip=False):
    cut_size = 32
    dest_x, dest_y = get_position(dest_pos)
    ori_x, ori_y = get_position(pos)
    subimg = ori_img[cut_size*ori_x:cut_size*(ori_x+1), cut_size*ori_y:cut_size*(ori_y+1)]
    if flip:
        # if op is set to be True, then flip the image
        subimg = np.flip(subimg, axis=2)
    dest_img[cut_size*dest_x:cut_size*(dest_x+1), cut_size*dest_y:cut_size*(dest_y+1)] = subimg

    return dest_img


def get_position(pos):
    x = pos % 6
    y = pos // 6
    return x, y
