from collections import defaultdict
import json
from typing import Dict, Any


def action_name_to_id(
        fine_grained_actions_map_file: str
    ) -> Dict[str, int]:
    """
    Args:
        fine_grained_actions_map_file: path to the raw annotation file.
    """
    action_name_to_id_dict = {}

    with open(fine_grained_actions_map_file) as f:
        for line in f.readlines():
            label_id, label_name = line.strip().split(" ")
            action_name_to_id_dict[label_name] = int(label_id)

    return action_name_to_id_dict


def id_to_action_name(fine_grained_actions_map_file: str):
    """
    Args:
        fine_grained_actions_map_file: path to the raw annotation file.
    """
    action_name_to_id_dict = action_name_to_id(
        fine_grained_actions_map_file=fine_grained_actions_map_file
    )
    id_to_action_name_dict = {}
    for label_name, label_id in action_name_to_id_dict.items():
        id_to_action_name_dict[label_id] = label_name
        
    return id_to_action_name_dict