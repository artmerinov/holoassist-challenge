# https://github.com/assembly-101/assembly101-action-recognition/blob/main/MS-G3D-action-recognition/MS_G3D/graph/assembly101_hands.py

import numpy as np

from .tools import get_spatial_graph

num_node = 26
self_link = [(i, i) for i in range(num_node)]
inward = [
    [0, 1],
    [1, 2], [2, 3], [3, 4], [4, 5], # Thumb
    [1, 6], [6, 7], [7, 8], [8, 9], [9, 10], # Index
    [1, 11], [11, 12], [12, 13], [13, 14], [14, 15], # Middle
    [1, 16], [16, 17], [17, 18], [18, 19], [19, 20], # Ring
    [1, 21], [21, 22], [22, 23], [23, 24], [24, 25] # Pinky
]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode="spatial"):
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A