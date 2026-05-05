import itertools

import java
from java.lang.reflect import Array

from src.utils.numeric import intr


def init_mat(n, init_value=0):
    a = Array.newInstance(java.lang.Float, [n, n])
    for i, j in itertools.product(range(n), repeat=2):
        a[i][j] = init_value
    return a


def save_mat_to_tsplib(mat, path):
    with open(path, "w") as f:
        f.write(
            (
                "NAME: Section_Similarity_Data\n"
                "TYPE: TSP\n"
                "DIMENSION: {}\n"
                "EDGE_WEIGHT_TYPE: EXPLICIT\n"
                "EDGE_WEIGHT_FORMAT: UPPER_ROW\n"
                "NODE_COORD_TYPE: NO_COORDS\n"
                "DISPLAY_DATA_TYPE: NO_DISPLAY\n"
                "EDGE_WEIGHT_SECTION\n"
            ).format(len(mat) + 1)
        )

        distances = [0] * len(mat)  # dummy city
        for i, j in itertools.combinations(range(len(mat)), 2):
            distances.append(intr(mat[i][j]))
        for id_distance, distance in enumerate(distances):
            f.write(str(distance))
            if (id_distance + 1) % 10 == 0:
                f.write("\n")
            else:
                f.write(" ")
        f.write("EOF\n")
