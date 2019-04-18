# coding=utf-8
# Created by posebox_refine on 2019-04-05 20:20
# Copyright © 2019 Alan. All rights reserved.
from utils import *


def joint_replace(posebox, missing_part):
    # posebox refine: joint replacement
    # in my implementation, I assume:
    # (0,14,15,16,17) to be similar
    # (2,5)(3,6)(4,7)(8,11)(9,12)(10,13) to be symmetric
    length = len(posebox)
    replaced = posebox
    for i in range(length):
        len_miss = len(missing_part)
        for j in range(len_miss):
            sequence = j
            if missing_part[i][j] in [2,3,4,8,9,10] and missing_part[i][j]+3 not in missing_part[i]:
                replaced = replace(replaced[i], posebox[i], sequence, sequence+3, flip=True)
            if missing_part[i][j] in [5,6,7,11,12,13] and missing_part[i][j]-3 not in missing_part[i]:
                replaced = replace(replaced[i], posebox[i], sequence, sequence-3, flip=True)
            if missing_part[i][j] in [0,14,15,16,17]:
                for dest in range([0,14,15,16,17]):
                    if dest not in missing_part:
                        replaced = replace(replaced[i], posebox[i], sequence, dest)
                        continue
    return replaced


def joint_success(db, ori_posebox, missing_part):
    # if posebox preserved in database
    for j in missing_part:
        # joint success, only when observed in the previous frames
        if j not in db.missing_part:
            # update the posebox in database
            db.posebox = replace(db.posebox, ori_posebox, j, j)
            missing_part.remove(j)
    # update the database
    db.missing_part = missing_part
    return db
