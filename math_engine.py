import cupy as cp
from itertools import product, combinations

def generate_vertices(n):
    """Generate the vertices of an n-dimensional hypercube.

    Args:
        n (int): The dimension of the hypercube.

    Returns:
        cupy.ndarray: A float32 array of shape (2^n, n) representing the vertices.
    """
    cords_list = list(product([-1.0, 1.0], repeat=n))
    return cp.array(cords_list, dtype=cp.float32)

def generate_edges(vertices):
    """Generate the indices of connecting vertices to form the hypercube's edges.

    Uses GPU matrix multiplication (dot products) to quickly identify 
    neighboring vertices that differ by exactly one coordinate.

    Args:
        vertices (cupy.ndarray): Array of vertices with shape (V, n).

    Returns:
        numpy.ndarray: A CPU-accessible 2D array of index pairs representing edges.
    """
    dim = vertices.shape[1]
    dot_products = vertices @ vertices.T
    matching_pairs = cp.argwhere(dot_products == (dim - 2))
    matching_pairs_cpu = matching_pairs.get()
    return matching_pairs_cpu

def get_rotation_planes(dimensions):
    """Generate all unique 2D coordinate planes of rotation for a given dimension.

    Args:
        dimensions (int): The dimension 'n' of the space.

    Returns:
        list[tuple[int, int]]: A list of unique index pairs representing planes.
                               Returns an empty list if dimensions < 2.
    """
    if dimensions < 2:
        print("Minimum 2 dimensions")
        return []
    
    combinations_list = list(combinations(range(dimensions), 2))
    return combinations_list

def create_plane_rotation_matrix(n, plane, angle):
    """Create an n-dimensional rotation matrix for a single 2D rotation plane.

    Args:
        n (int): The dimension of the space.
        plane (tuple[int, int]): A pair of indices (i, j) defining the rotation plane.
        angle (float): The angle of rotation in radians.

    Returns:
        cupy.ndarray: An (n, n) orthogonal rotation matrix of type float32.
    """
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
    """Chain multiple plane rotations into a single, unified n-dimensional matrix.

    Args:
        n (int): The dimension of the space.
        active_planes (list[tuple[int, int]]): List of all available rotation planes.
        angles_dict (dict[tuple[int, int], float]): Dictionary mapping planes to angles in radians.

    Returns:
        cupy.ndarray: A combined (n, n) transformation matrix.
    """
    combined = cp.eye(n, dtype=cp.float32)

    for plane in active_planes:
        angle = angles_dict.get(plane, 0.0)

        if angle == 0.0:
            continue

        r_matrix = create_plane_rotation_matrix(n, plane, angle)
        combined = combined @ r_matrix
    return combined

def project_vertices(vertices, scale, screen_center, distance=3.0):
    """Project n-dimensional coordinates down to 2D screen space.

    Applies cascading perspective division dimension-by-dimension,
    compensates for scale loss in high dimensions, and offsets to screen center.

    Args:
        vertices (cupy.ndarray): Rotated vertices of shape (V, n).
        scale (float): Base viewport scale factor.
        screen_center (tuple[int, int]): The screen center (X, Y) for translation.
        distance (float, optional): Virtual camera distance for perspective division. Defaults to 3.0.

    Returns:
        cupy.ndarray: Screen-space coordinates with shape (V, 2) on the GPU.
    """
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