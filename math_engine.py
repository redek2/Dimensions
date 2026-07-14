import cupy as cp
from itertools import product

def generate_vertices(n):
    cords_list = list(product([-1.0, 1.0], repeat=n))
    return cp.array(cords_list, dtype=cp.float32)

def generate_edges(vertices):
    dim = vertices.shape[1]
    dot_products = vertices @ vertices.T
    matching_pairs = cp.argwhere(dot_products == (dim - 2))
    matching_pairs_cpu = matching_pairs.get()
    return matching_pairs_cpu


if __name__ == "__main__":
    square_v = generate_vertices(2)
    square_e = generate_edges(square_v)
    cube_v = generate_vertices(3)
    cube_e = generate_edges(cube_v)
    print(f"\nVertices for square:\n\n{square_v}")
    print(f"\nEdges for square:\n\n{square_e}")
    print(f"\nVertices for cube:\n\n{cube_v}")
    print(f"\nEdges for cube:\n\n{cube_e}")