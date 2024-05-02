import numpy as np
import matplotlib.pyplot as plt


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