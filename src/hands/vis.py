import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def vis_hands_skeleton(
        left_hand: torch.Tensor,
        right_hand: torch.Tensor,
        joints: List[List[int]]
    ) -> None:
    """
    Visualize 3D plot of hands.
    """
    # Create a 3D plot
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(left_hand[:, 0], left_hand[:, 1], left_hand[:, 2], color="red", label="Left")
    ax.scatter(right_hand[:, 0], right_hand[:, 1], right_hand[:, 2], color="blue", label="Right")

    # Plot the connections
    for joint in joints:
        for points, color in zip([left_hand, right_hand], ["red", "blue"]):
            start = points[joint[0]]
            end = points[joint[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, lw=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def vis_matrix(matrix: np.ndarray) -> None:

    plt.figure(figsize=(5,5))
    plt.imshow(matrix, cmap="Oranges");

    nrows = matrix.shape[0]
    ncols = matrix.shape[1]
    
    # Annotate cells
    for row in range(nrows):
        for col in range(ncols):
            if matrix[row][col] == 0:
                val = ""
            else:
                val = f"{matrix[row][col]:.1f}"
            plt.text(col, row, val, ha="center", va="center", color="white", fontsize=5)

    # Create grid
    for row in range(1, nrows):
        plt.axvline(row - 0.5, color='white', linewidth=1)
    for col in range(1, ncols):
        plt.axhline(col - 0.5, color='white', linewidth=1)

    # Ticks
    plt.xticks(range(ncols), rotation=90, ha="right", fontsize=5) # to
    plt.yticks(range(nrows), fontsize=6) # from
    plt.ylabel("B")
    plt.xlabel("A")

    plt.show()