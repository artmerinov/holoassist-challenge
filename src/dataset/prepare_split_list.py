import os
from typing import List


def get_video_name_list(
        split_dir: str, 
        holoassist_dir: str,
        mode="train",
    ) -> List[str]:
    """
    Outputs list of video paths.
    """
    tr_list_path = os.path.join(split_dir, "train-v1.txt")
    va_list_path = os.path.join(split_dir, "val-v1.txt")
    te_list_path = os.path.join(split_dir, "test-v1.txt")

    if mode == "train":
        with open(tr_list_path, 'r') as file:
            video_list = [line.strip() for line in file.readlines()]
        video_list = reduce_list(video_name_list=video_list, holoassist_dir=holoassist_dir)
    elif mode == "validation":
        with open(va_list_path, 'r') as file:
            video_list = [line.strip() for line in file.readlines()]
        video_list = reduce_list(video_name_list=video_list, holoassist_dir=holoassist_dir)
    else:
        with open(te_list_path, 'r') as file:
            video_list = [line.strip() for line in file.readlines()]

    return video_list


def reduce_list(
        video_name_list: List[str],
        holoassist_dir: str,
    ) -> List[str]:
    """
    Some video names in the list are actially not present as video files, 
    therefore, we will skip them.
    """
    # Collect all downloaded videos.
    available_video_names = set()
    path = f"{holoassist_dir}/video_pitch_shifted/"
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            available_video_names.add(item)

    # Check how many names from the split list file
    # are actually stored on the disk.
    # Dont use set to preserve order for reproducibility.
    intersection = []
    for item in video_name_list:
        if item in available_video_names:
            intersection.append(item)

    # Count the number of videos that present in the split list 
    # but are missing as videos.
    not_present_count = len(video_name_list) - len(intersection)
    print(f"\nThere are {len(video_name_list)} videos in the split list file.",
          f"\nThere are {len(available_video_names)} videos downloaded and stored on the disk",
          f"\nThere are {not_present_count} videos that present in the list but are missing on the disk.")

    return intersection
