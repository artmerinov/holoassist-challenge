import os
import math
from typing import Dict


def secs_to_pts(
        time_in_seconds: float,
        time_base: float,
        start_pts: int,
        round_mode: str = "floor",
    ) -> int:
    """
    Converts a time (in seconds) to the given time base and start_pts offset
    presentation time. Round_mode specifies the mode of rounding when converting time.

    Returns:
        pts (int): The time in the given time base.
    """
    if time_in_seconds == math.inf:
        return math.inf

    assert round_mode in ["floor", "ceil"], f"round_mode={round_mode} is not supported!"

    if round_mode == "floor":
        return math.floor(time_in_seconds / time_base) + start_pts
    else:
        return math.ceil(time_in_seconds / time_base) + start_pts


def pts_to_secs(
        pts: int, 
        time_base: float,
        start_pts: int,
    ) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base


def make_video_path(
        holoassist_dir: str,
        video_name: str
):
    path = f"{holoassist_dir}/video_pitch_shifted/{video_name}/Export_py/Video_pitchshift.mp4"
    return path


def fganame_to_fgaid(fga_map_file: str) -> Dict[str, int]:
    """
    Args:
        fga_map_file: path to the fine grained action annotation file.
    """
    action_name_to_id_dict = {}

    with open(fga_map_file) as f:
        for line in f.readlines():
            label_id, label_name = line.strip().split(" ")
            action_name_to_id_dict[label_name] = int(label_id)

    return action_name_to_id_dict


def fgaid_to_fganame(fga_map_file: str) -> Dict[str, int]:
    """
    Args:
        fga_map_file: path to the fine grained action annotation file.
    """
    action_name_to_id_dict = fganame_to_fgaid(fga_map_file=fga_map_file)
    id_to_action_name_dict = {}
    for label_name, label_id in action_name_to_id_dict.items():
        id_to_action_name_dict[label_id] = label_name
        
    return id_to_action_name_dict
