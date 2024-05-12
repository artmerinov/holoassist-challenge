# https://github.com/taeinkwon/PyHoloAssist/blob/main/hand_eye_project.py

import numpy as np


def read_hand_pose_txt(hand_path):
    #  The format for each entry is: Time, IsGripped, IsPinched, IsTracked, IsActive, {Joint values}, {Joint valid flags}, {Joint tracked flags}
    hand_array = []
    with open(hand_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            hand = []
            line_data = list(map(float, line.split('\t')))
            line_data_reshape = np.reshape(
                line_data[3:-52], (-1, 4, 4))  # For version2: line_data[5:-52]

            line_data_xyz = []
            for line_data_reshape_elem in line_data_reshape:
                # To get translation of the hand joints
                location = np.dot(line_data_reshape_elem,
                                np.array([[0, 0, 0, 1]]).T)
                line_data_xyz.append(location[:3].T[0])

            line_data_xyz = np.array(line_data_xyz).T
            hand = line_data[:4]
            hand.extend(line_data_xyz.reshape(-1))
            hand_array.append(hand)
        hand_array = np.array(hand_array)
    return hand_array
