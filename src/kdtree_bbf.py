# src/kdtree_bbf.py

import numpy as np
import heapq # For priority queue in BBF

class Node:
    """
    Represents a node in the K-D Tree.
    """
    def __init__(self, points_indices=None, axis=None, median_value=None,
                 left_child=None, right_child=None, is_leaf=False,
                 bounding_box=None):
        self.points_indices = points_indices # Indices of points in this leaf node (maps to original dataset)
        self.axis = axis                 # Splitting axis for internal nodes
        self.median_value = median_value # Splitting value for internal nodes
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.bounding_box = bounding_box # [min_coords, max_coords] for this node's region

    def __lt__(self, other):
        # heapq needs a comparison method for items if priorities are equal.
        # This is a placeholder; actual priority is an external value.
        return id(self) < id(other)


def _build_kdtree_recursive(points_data, point_indices, depth, leaf_size):
    """
    Recursive helper to build the K-D Tree.
    points_data: The full dataset array.
    point_indices: Indices of points currently considered for this node.
    depth: Current depth of the tree.
    leaf_size: Maximum number of points in a leaf node.
    """
    n_points = len(point_indices)
    if n_points == 0:
        return None

    # Determine bounding box for the current points
    current_points = points_data[point_indices]
    min_coords = np.min(current_points, axis=0)
    max_coords = np.max(current_points, axis=0)
    bbox = np.array([min_coords, max_coords])

    if n_points <= leaf_size:
        return Node(points_indices=point_indices, is_leaf=True, bounding_box=bbox)

    n_dims = points_data.shape[1]
    axis = depth % n_dims

    # Sort points along the current axis to find the median
    # We sort the *indices* based on the point values to avoid copying point data repeatedly
    sorted_original_indices = sorted(point_indices, key=lambda idx: points_data[idx, axis])
    
    median_split_idx = n_points // 2
    
    # Handle cases where many points have the same coordinate on the splitting axis
    # Ensure the median point chosen for splitting actually divides the set if possible
    # This simple median might lead to imbalanced trees if many points are identical on axis
    median_original_idx = sorted_original_indices[median_split_idx]
    median_value = points_data[median_original_idx, axis]

    left_indices = []
    right_indices = []

    # Partition indices based on the median value
    # Points equal to median can go to either side; here, they go to right for simplicity or based on split point
    # A more robust split would ensure points strictly less go left, others go right.
    # Or handle exact median matches carefully.
    
    # For a robust split that handles duplicates at median:
    left_indices = [idx for idx in sorted_original_indices[:median_split_idx]]
    right_indices = [idx for idx in sorted_original_indices[median_split_idx:]]
    
    # If one side is empty due to all points being same as median or poor split choice, make it a leaf
    if not left_indices or not right_indices:
        # This can happen if all remaining points are identical along the split axis,
        # or if leaf_size is very small relative to duplicates.
        return Node(points_indices=point_indices, is_leaf=True, bounding_box=bbox)

    return Node(
        axis=axis,
        median_value=median_value, # The value at the median point on the axis
        left_child=_build_kdtree_recursive(points_data, left_indices, depth + 1, leaf_size),
        right_child=_build_kdtree_recursive(points_data, right_indices, depth + 1, leaf_size),
        is_leaf=False,
        bounding_box=bbox
    )

def build_kdtree(points_data, leaf_size=10):
    """
    Builds a K-D Tree.
    points_data: numpy array of shape (n_samples, n_features).
    leaf_size: maximum number of points in a leaf node.
    Returns the root node of the K-D Tree.
    """
    n_samples = points_data.shape[0]
    if n_samples == 0:
        return None
    initial_indices = np.arange(n_samples)
    return _build_kdtree_recursive(points_data, initial_indices, 0, leaf_size)

def _squared_euclidean_dist(p1, p2):
    return np.sum((p1 - p2)**2)

def _min_dist_sq_to_node_bbox(query_point, node_bbox):
    """
    Calculates the minimum squared Euclidean distance from a query_point to a node's bounding_box.
    """
    if node_bbox is None: return np.inf
    min_coords, max_coords = node_bbox[0], node_bbox[1]
    dist_sq_sum = 0.0
    for i in range(len(query_point)):
        q_i = query_point[i]
        if q_i < min_coords[i]:
            dist_sq_sum += (min_coords[i] - q_i)**2
        elif q_i > max_coords[i]:
            dist_sq_sum += (q_i - max_coords[i])**2
    return dist_sq_sum

def bbf_knn_search(root_node, full_dataset_points, query_point, k, t_max):
    """
    Best-Bin-First k-NN search.
    root_node: Root of the K-D Tree.
    full_dataset_points: The original NxD array of points.
    query_point: 1D array, the point to search for.
    k: Number of nearest neighbors to find.
    t_max: Maximum number of leaf nodes to inspect.
    Returns: (distances_sq, indices) of the k-nearest neighbors. distances_sq are squared Euclidean.
    """
    if root_node is None:
        return np.array([]), np.array([])

    # Priority queue for nodes to visit: (min_dist_to_node_bbox_sq, node_object)
    nodes_to_visit_pq = []
    
    # Stores K best neighbors found so far: (-dist_sq_to_point, point_original_index)
    # Using negative distance because heapq is a min-heap, effectively making it a max-heap for distances.
    best_k_neighbors_heap = []

    leaves_inspected_count = 0
    
    # Initial dist_sq to root's bbox
    dist_to_root_bbox_sq = _min_dist_sq_to_node_bbox(query_point, root_node.bounding_box)
    heapq.heappush(nodes_to_visit_pq, (dist_to_root_bbox_sq, root_node))

    # Distance to the current k-th furthest neighbor (squared)
    # This is the radius of our current k-NN ball.
    kth_furthest_dist_sq = np.inf

    while nodes_to_visit_pq and leaves_inspected_count < t_max:
        dist_to_node_sq, current_node = heapq.heappop(nodes_to_visit_pq)

        # Pruning step: if the closest point of this node's bbox is further
        # than our current k-th neighbor, we can skip this node and its children.
        if dist_to_node_sq >= kth_furthest_dist_sq and len(best_k_neighbors_heap) == k:
            continue

        if current_node.is_leaf:
            leaves_inspected_count += 1
            for point_original_idx in current_node.points_indices:
                point = full_dataset_points[point_original_idx]
                d_sq = _squared_euclidean_dist(query_point, point)

                if len(best_k_neighbors_heap) < k:
                    heapq.heappush(best_k_neighbors_heap, (-d_sq, point_original_idx))
                    if len(best_k_neighbors_heap) == k: # Heap just filled up to k
                         kth_furthest_dist_sq = -best_k_neighbors_heap[0][0] # Smallest neg_dist is largest pos_dist
                elif d_sq < kth_furthest_dist_sq: # Found a closer point than current k-th
                    heapq.heapreplace(best_k_neighbors_heap, (-d_sq, point_original_idx))
                    kth_furthest_dist_sq = -best_k_neighbors_heap[0][0] # Update k-th distance
        
        else: # Internal node
            axis = current_node.axis
            median_val = current_node.median_value
            
            # Determine which child is "closer" based on query point's position relative to splitting plane
            if query_point[axis] < median_val:
                nearer_child = current_node.left_child
                further_child = current_node.right_child
            else:
                nearer_child = current_node.right_child
                further_child = current_node.left_child

            # Explore nearer child first
            if nearer_child and nearer_child.bounding_box is not None:
                dist_to_nearer_bbox_sq = _min_dist_sq_to_node_bbox(query_point, nearer_child.bounding_box)
                # Only add if it could contain a better point
                if dist_to_nearer_bbox_sq < kth_furthest_dist_sq or len(best_k_neighbors_heap) < k:
                    heapq.heappush(nodes_to_visit_pq, (dist_to_nearer_bbox_sq, nearer_child))
            
            # Explore further child if its bounding box could still contain a better point
            if further_child and further_child.bounding_box is not None:
                dist_to_further_bbox_sq = _min_dist_sq_to_node_bbox(query_point, further_child.bounding_box)
                if dist_to_further_bbox_sq < kth_furthest_dist_sq or len(best_k_neighbors_heap) < k:
                     heapq.heappush(nodes_to_visit_pq, (dist_to_further_bbox_sq, further_child))

    # Extract and sort results
    # best_k_neighbors_heap contains (-d_sq, index)
    final_neighbors_sorted = sorted([(-neg_d_sq, idx) for neg_d_sq, idx in best_k_neighbors_heap], key=lambda x: x[0])
    
    distances_sq_result = np.array([d for d, i in final_neighbors_sorted])
    indices_result = np.array([i for d, i in final_neighbors_sorted], dtype=int)
    
    return distances_sq_result, indices_result
