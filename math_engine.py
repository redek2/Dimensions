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


if __name__ == "__main__":
    # --- TESTS FOR ISSUE #1 ---
    square_v = generate_vertices(2)
    square_e = generate_edges(square_v)
    cube_v = generate_vertices(3)
    cube_e = generate_edges(cube_v)
    
    print(f"\nVertices for square:\n\n{square_v}")
    print(f"\nEdges for square:\n\n{square_e}")
    print(f"\nVertices for cube:\n\n{cube_v}")
    print(f"\nEdges for cube:\n\n{cube_e}")

    # --- TESTS FOR ISSUE #2 ---
    print("\n" + "="*40)
    print("ROTATION PLANES TESTING")
    print("="*40)
    
    print(f"Planes for 1D: {get_rotation_planes(1)}")
    
    planes_3d = get_rotation_planes(3)
    print(f"Planes for 3D ({len(planes_3d)}): {planes_3d}")
    
    planes_4d = get_rotation_planes(4)
    print(f"Planes for 4D ({len(planes_4d)}): {planes_4d}")