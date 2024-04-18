from typing import Dict, Any
import random

from .prepare_labels_mapping import action_name_to_id


class ClipRecord:
    """
    Class to store information from trimmed video 
    (clip).
    """
    def __init__(self,
            holoassist_dir: str,
            video_name: str,
            fine_grained_actions_map_file: str,
            clip: Dict[str, Any]
        ) -> None:
        
        self.holoassist_dir = holoassist_dir
        self._video_name = video_name
        self._clip = clip

        self.action_name_to_id_dict = action_name_to_id(
            fine_grained_actions_map_file=fine_grained_actions_map_file)

    @property
    def untrimmed_video_name(self):
        return self._video_name
    
    @property
    def path_to_video(self):
        return f"{self.holoassist_dir}/video_pitch_shifted/{self._video_name}/Export_py/Video_pitchshift.mp4"
    
    @property
    def start(self):
        return self._clip["Start"]
    
    @property
    def end(self):
        return self._clip["End"]

    @property
    def verb(self):
        return self._clip["Verb"]

    @property
    def noun(self):
        return self._clip["Noun"]
    
    @property
    def action(self):
        return self._clip["Action"]
    
    @property
    def action_id(self):
        return self.action_name_to_id_dict[self.action]

    @property
    def correctness(self):
        return self._clip["Action Correctness"]
