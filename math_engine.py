import cupy as cp
from itertools import product, combinations

def generate_vertices(n):
    cords_list = list(product([-1.0, 1.0], repeat=n))
    return cp.array(cords_list, dtype=cp.float32)

def generate_edges(vertices):
    dim = vertices.shape[1]
    dot_products = vertices @ vertices.T
    matching_pairs = cp.argwhere(dot_products == (dim - 2))
    matching_pairs_cpu = matching_pairs.get()
    return matching_pairs_cpu

def get_rotation_planes(dimensions):
    if(dimensions < 2):
        print("Minimum 2 dimensions")
        return []
    
    combinations_list = list(combinations(range(dimensions), 2))
    return combinations_list

def create_plane_rotation_matrix(n, plane, angle):
    rotation_matrix = cp.eye(n)
    c = cp.cos(angle)
    s = cp.sin(angle)
    i, j = plane
    rotation_matrix[i, i] = c
    rotation_matrix[j, j] = c
    rotation_matrix[i, j] = -s
    rotation_matrix[j, i] = s
    return rotation_matrix

def get_combined_rotation_matrix(n, active_planes, angles_dict):
    combined = cp.eye(n, dtype=cp.float32)

    for plane in active_planes:
        angle = angles_dict.get(plane, 0.0)

        if angle == 0.0:
            continue

        r_matrix = create_plane_rotation_matrix(n, plane, angle)
        combined = combined @ r_matrix
    return combined

def project_vertices(vertices, scale, screen_center, distance = 3.0):
    coords = vertices.copy()
    n = coords.shape[1]

    while coords.shape[1] > 2:
        depth = coords[:, -1]
        coords = coords[:, :-1]

        coords = coords / (distance + depth)[:, cp.newaxis]
    
    adjusted_scale = scale * (distance ** (n - 2))

    screen_x = coords[:, 0] * adjusted_scale + screen_center[0]
    screen_y = coords[:, 1] * adjusted_scale + screen_center[1]

    return cp.column_stack((screen_x, screen_y))

if __name__ == "__main__":