from collections import defaultdict
import json
from typing import Dict, Any


def prepare_annotation(
        raw_annotation_file: str
    ) -> Dict[str, Dict[str, Any]]:
    """
    Args:
        raw_annotation_path: path to the raw annotation file.
    """
    with open(raw_annotation_file) as f:
        raw_annt = json.load(f)

    annt = defaultdict(list)
    for video in raw_annt:
        name = video["video_name"]
        events = video["events"]
        for event in events:
            if event["label"] == "Fine grained action":
                action_info = {
                    "Start": event["start"],
                    "End": event["end"],
                    "Verb": event["attributes"]["Verb"],
                    "Noun": event["attributes"]["Noun"],
                    "Action": event["attributes"]["Verb"] + "-" + event["attributes"]["Noun"],
                    "Action Correctness": int(event["attributes"]["Action Correctness"] == "Correct Action")
                }
                annt[name].append(action_info)
    
    return annt
