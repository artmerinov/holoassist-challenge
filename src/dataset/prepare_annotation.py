from collections import defaultdict
import json
from typing import Dict, Any


def prepare_annotation_action(
        raw_annotation_file: str
    ) -> Dict[str, Dict[str, Any]]:
    """
    Prepare annotation dict from start of fine-grained action
    to the end of this fine-grained action.

    Args:
        raw_annotation_path: path to the raw annotation file.
    """
    with open(raw_annotation_file) as f:
        raw_annt = json.load(f)

    annt = defaultdict(list)
    for video in raw_annt:
        video_name = video["video_name"]
        events = video["events"]
        for event in events:
            if event["label"] == "Fine grained action":
                action_info = {
                    "start": event["start"],
                    "end": event["end"],
                    "label": event["attributes"]["Verb"] + "-" + event["attributes"]["Noun"],
                }
                annt[video_name].append(action_info)
    
    return annt


def prepare_annotation_mistake(
        raw_annotation_file: str
    ) -> Dict[str, Dict[str, Any]]:
    """
    Prepare annotation dict from start of course-grained action
    to the end of each fine-grained action clip. If there is no 
    mistake during this time period, the final label will be 0 
    (no mistakes). If there is at least one mistake during this 
    period, then the final label will be 1 (a mistake).

    Args:
        raw_annotation_path: path to the raw annotation file.
    """
    with open(raw_annotation_file) as f:
        raw_annt = json.load(f)

    annt = defaultdict(list)
    for video in raw_annt:
        video_name = video["video_name"]
        events = video["events"]
        for event in events:
            
            if event["label"] == "Coarse grained action":
                cga_st = event["start"]
                
                # Wherer there is a mistake or not from start 
                # of sequence of course-gained action (CGA) 
                # up to end of current fine-grained action (FGA).
                cga_mistake = 0


            elif event["label"] == "Fine grained action":
                
                fga_end = event["end"]
                fga_mistake = int(event["attributes"]["Action Correctness"] != "Correct Action") # 1 if a mistake 0 otherwise
                cga_mistake = max(cga_mistake, fga_mistake)

                action_info = {
                    "start": cga_st,
                    "end": fga_end,
                    "label": cga_mistake,
                }

                annt[video_name].append(action_info)
    
    return annt