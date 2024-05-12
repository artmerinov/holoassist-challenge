import os
import torch
import numpy as np
from typing import List, Any


def read_hand_pose_txt(hand_path):
    """
    # https://github.com/taeinkwon/PyHoloAssist/blob/main/hand_eye_project.py
    """
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


def make_paths_to_hands(
        holoassist_dir: str,
        video_name: str
):
    left_hand_file = os.path.join(holoassist_dir, "hands", video_name, "Export_py/Hands/Left_sync.txt")
    right_hand_file = os.path.join(holoassist_dir, "hands", video_name, "Export_py/Hands/Right_sync.txt")
    return left_hand_file, right_hand_file


def load_hands_coords(
        holoassist_dir: str,
        video_name: str, 
        start_secs: float, 
        end_secs: float
    ) -> torch.Tensor:
    """
    Load hands coordinates between start and end time.
    """
    # Construct paths to hand skeleton data.
    left_hand_path, right_hand_path = make_paths_to_hands(
        holoassist_dir=holoassist_dir,
        video_name=video_name,
    )
    
    # Load row data.
    left_hand_data = read_hand_pose_txt(hand_path=left_hand_path)
    right_hand_data = read_hand_pose_txt(hand_path=right_hand_path)

    # Extract time values from row data.
    times = left_hand_data[:, 0]

    # Select coordinates based on start and end time
    # and reshape arrays to have [-1, 26, 3] size.
    time_ids = np.where((times > start_secs) & (times < end_secs))[0]
    left_hand_data = left_hand_data[time_ids, 4:].reshape(-1, 3, 26).transpose(0, 2, 1)
    right_hand_data = right_hand_data[time_ids, 4:].reshape(-1, 3, 26).transpose(0, 2, 1)

    # Make it torch. 
    left_hand_data = torch.tensor(left_hand_data, dtype=torch.float32)
    right_hand_data = torch.tensor(right_hand_data, dtype=torch.float32)

    return left_hand_data, right_hand_data
