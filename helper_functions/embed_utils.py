import scipy as sci
import numpy as np
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
from time import time
from helper_functions.utils import replace_nan_inf, print_memory_usage


def column_norm(K):
    if sp.issparse(K):
            # Compute column sums for sparse matrix
            col_sum = np.array(K.sum(axis=0)).flatten()
            # Handle division by zero in col_sum
            col_sum[col_sum == 0] = 1
            # Compute the inverse of column sums
            inv_sum = 1.0 / col_sum
            # remove inf values in inv_sum
            inv_sum[np.isinf(inv_sum)] = 1
            # Create diagonal matrix with inverse column sums
            norm_mat = sp.diags(inv_sum)
            # Normalize columns
            P = K @ norm_mat
    else:
        # Compute column sums for dense matrix
        col_sum = np.sum(K, axis=0)
        # Handle division by zero in col_sum
        col_sum[col_sum == 0] = 1
        # Compute the inverse of column sums
        inv_sum = 1.0 / col_sum
        # remove inf values in inv_sum
        inv_sum[np.isinf(inv_sum)] = 1
        # Create diagonal matrix with inverse column sums
        norm_mat = np.diag(inv_sum)
        # Normalize columns
        P = K @ norm_mat
    return P


def row_norm(K):
    if sp.issparse(K):
            # Compute column sums for sparse matrix
            col_sum = np.array(K.sum(axis=1)).flatten()
            # Handle division by zero in col_sum
            col_sum[col_sum == 0] = 1
            # Compute the inverse of column sums
            inv_sum = 1.0 / col_sum
            # remove inf values in inv_sum
            inv_sum[np.isinf(inv_sum)] = 1
            # Create diagonal matrix with inverse column sums
            norm_mat = sp.diags(inv_sum)
            # Normalize columns
            P = norm_mat @ K
    else:
        # Compute column sums for dense matrix
        col_sum = np.sum(K, axis=1)
        # Handle division by zero in col_sum
        col_sum[col_sum == 0] = 1
        # Compute the inverse of column sums
        inv_sum = 1.0 / col_sum
        # remove inf values in inv_sum
        inv_sum[np.isinf(inv_sum)] = 1
        # Create diagonal matrix with inverse column sums
        norm_mat = np.diag(inv_sum)
        # Normalize columns
        P = norm_mat @ K
    return P


# Function to create Markov Transition matrix for Standard Diffusion Map Algorithm
def Create_Transition_Mat(data_points, scale=2, mode='median'):
    # Calculate the kernel matrix
    start_time = time()
    N = data_points.shape[0]
    dist_mat = sci.spatial.distance.pdist(data_points, metric='euclidean')
    dist_mat = sci.spatial.distance.squareform(dist_mat)

    # Compute Kernel
    if mode == 'median':
        adjusted_scale = (np.median(dist_mat) ** 2) * scale # popular choice for sigma
    elif mode == 'scale':
        adjusted_scale = scale
    else:
        raise ValueError(f'invalid mode {mode}')
    K_mat = np.exp(-dist_mat ** 2 / adjusted_scale)
    # Noramalize kernel matrix columns to create Markov transition matrix
    P = column_norm(K_mat)
    end_time = time()
    print(f' Kernel computation finished, in {end_time - start_time} seconds')
    return K_mat, P


def dist2kernel(dist_mat, scale=2, zero_diag=False, k=None):
    '''
    function that computes a similarity kernel from a distance matrix
    :param dist_mat: the distance matrix
    :param scale: scale factor for the similarity kernel (multiplies the distance median squared)
    :param zero_diag: whether to zero out the diagonal elements of the kernel
    :param k: number of nearest neighbors used in the kernel - only NN have non-zero similarity
        (if None then all elements are used)
    :return: K_mat: the similarity kernel
    '''
    # Ensure dist_mat is a numpy array
    dist_mat = np.array(dist_mat)

    if k is not None:
        # Create a mask for the K nearest neighbors in each row
        mask = np.zeros_like(dist_mat, dtype=bool)
        for i in range(dist_mat.shape[0]):
            nearest_indices = np.argsort(dist_mat[i])[1:k + 1]  # Excluding self
            mask[i, nearest_indices] = True

        # Extract the distances for the K nearest neighbors
        k_nearest_distances = dist_mat[mask]

        # Compute sigma using the median of the non-zero elements of the K nearest distances
        sigma = np.median(k_nearest_distances[k_nearest_distances > 0])
        adjusted_scale = (sigma ** 2) * scale  # popular choice for sigma

        # Initialize the kernel matrix with zeros
        K_mat = np.zeros_like(dist_mat)

        # Compute the kernel for the K nearest points
        for i in range(dist_mat.shape[0]):
            nearest_indices = np.argsort(dist_mat[i])[1:k + 1]  # Excluding self
            K_mat[i, nearest_indices] = np.exp(-dist_mat[i, nearest_indices] ** 2 / (adjusted_scale))

    else:
        # Compute sigma using the median of the non-zero elements in the distance matrix
        sigma = np.median(dist_mat[dist_mat > 0])
        adjusted_scale = (sigma ** 2) * scale  # popular choice for sigma

        # Compute the kernel for all points
        K_mat = np.exp(-dist_mat ** 2 / (adjusted_scale))

    # Zero out the diagonal to eliminate self-loops
    if zero_diag:
        np.fill_diagonal(K_mat, 0)

    return K_mat


# compute kernel with sparsification for better time and memory
def Create_Transition_Mat_Sparse(data_points, k=1000, scale=1):
    # Compute the k-nearest neighbors indicator
    start_time = time()
    knn_graph = kneighbors_graph(data_points, n_neighbors=k, mode='connectivity', metric='euclidean')

    # Get the indices of the k-nearest neighbors for each point
    roots_idx, neighbors_idx = knn_graph.nonzero()

    # Initialize a sparse matrix for the kernel matrix
    n_samples = data_points.shape[0]
    K_mat = sp.lil_matrix((n_samples, n_samples), dtype=float)

    # Compute pairwise distances and median sigma
    distances = np.linalg.norm(data_points[roots_idx] - data_points[neighbors_idx], axis=1)
    sigma = np.median(distances)
    adjusted_scale = (sigma ** 2) * scale  # popular choice for sigma

    # Compute Kernel for KNN
    K_mat[roots_idx, neighbors_idx] = np.exp(-distances ** 2 / (adjusted_scale))
    K_mat[neighbors_idx, roots_idx] = np.exp(-distances ** 2 / (adjusted_scale))

    # Normalize columns
    P = column_norm(K_mat)

    end_time = time()
    print(f'Kernel computation finished, in {end_time - start_time} seconds')

    return K_mat, P


# Function to create asymmetric transition kernel - for out of sample extension
def Create_Asym_Tran_Kernel(data_points1, data_points2, scale=2, mode='median'):
    # data_points2 should be the reference set!
    start_time = time()
    # Calculate the kernel matrix
    dist_mat = sci.spatial.distance.cdist(data_points1, data_points2, metric='euclidean')
    

    if mode == 'median':
        adjusted_scale = (np.median(dist_mat) ** 2) * scale  # popular choice for sigma
    elif mode == 'scale':
        adjusted_scale = scale
    else:
        raise ValueError(f'invalid mode {mode}')
    A = np.exp(-dist_mat ** 2 / (adjusted_scale))
    # Normalize columns of K_mat
    P_L = row_norm(A)

    # Normalize columns of K_mat Transpose
    P_R = row_norm(A.T)
    end_time = time()
    print(f' Kernel computation finished, in {end_time - start_time} seconds')
    return A, P_L, P_R


# Function to create a sparse asymmetric transition kernel - for out of sample extension
def Create_Asym_Tran_Kernel_Sparse(data_points1, data_points2, k=1000, scale=1):
    # Compute the k-nearest neighbors indicator
    start_time = time()
    # Fit NearestNeighbors on data_points2
    nn_model = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    nn_model.fit(data_points2)

    # Find k-nearest neighbors for each point in data_points1
    distances, indices = nn_model.kneighbors(data_points1)

    # compute median for the kernel scale
    sigma = np.median(distances.flatten())
    adjusted_scale = (sigma ** 2) * scale  # popular choice for sigma

    # Initialize a sparse matrix for the bipartite KNN graph
    n_samples_1 = data_points1.shape[0]
    n_samples_2 = data_points2.shape[0]
    A = sp.lil_matrix((n_samples_1, n_samples_2), dtype=float)

    # Set True for nearest neighbors in the bipartite KNN graph
    idx1 = np.arange(n_samples_1)
    for kk in range(k):
        A[idx1, indices[idx1, kk]] = np.exp(-distances[:, kk] ** 2 / (adjusted_scale))

    # Normalize columns of K_mat
    P_L = row_norm(A)

    # Normalize columns of K_mat Transpose
    P_R = row_norm(A.T)
    end_time = time()
    print(f' Kernel computation finished, in {end_time - start_time} seconds')
    return A, P_L, P_R


# implement a metric for embedding quality of each point
def embed_score(embed_orig, embed_new):
    score = np.abs(embed_orig - embed_new)
    return score


# uniqify eigenvectors
def eigvec_uniq(vecs):
    # make vectors unique by ensuring the sum of elements is positive
    sum_sign = np.sum(vecs, axis=0) / (np.abs(np.sum(vecs,axis=0)) + 1e-12)
    return vecs*np.reshape(vecs, newshape=(-1, 1))


def sort_evd_components(vals, vecs):
    srt_idx = np.argsort(np.abs(vals))[::-1]  # sort indecies to use the vectors with the biggest eigenvalues
    vals = vals[srt_idx]
    vecs = vecs[:, srt_idx]
    return vals, vecs


def sort_svd_components(vecs_l, vals, vecs_r):
    srt_idx = np.argsort(np.abs(vals))[::-1]  # sort indecies to use the vectors with the biggest eigenvalues
    vals = vals[srt_idx]
    vecs_l = vecs_l[:, srt_idx]
    vecs_r = vecs_r[srt_idx, :]
    return vecs_l, vals, vecs_r


def prep_kernels(dist_mat1=None, dist_mat2=None, method='forward_only', scale1=2, scale2=None, zero_diag=False, k=None):
    if scale2 is None:
        scale2 = scale1
    # if method in {"forward_only", "forward_only_slow", "alternating_roseland", "ffbb", "fbfb", "ncca",
    #               "kcca", "nystrom"}:
    #     A1 = dist2kernel(dist_mat1, scale=scale1, zero_diag=zero_diag, k=k)
    #     K2_ref = dist2kernel(dist_mat2, scale=scale2, zero_diag=zero_diag, k=k)
    #     return A1, K2_ref
    #
    # elif method in {"ad", 'dm'}:
    #     K1 = dist2kernel(dist_mat1, scale=scale1, zero_diag=zero_diag, k=k)
    #     K2 = dist2kernel(dist_mat2, scale=scale2, zero_diag=zero_diag, k=k)
    #     return K1, K2
    # elif method == "lead":
    #     A1 = dist2kernel(dist_mat1, scale=scale1, zero_diag=zero_diag, k=k)
    #     A2 = dist2kernel(dist_mat2, scale=scale2, zero_diag=zero_diag, k=k)
    #     return A1, A2
    K1 = dist2kernel(dist_mat1, scale=scale1, zero_diag=zero_diag, k=k)
    K2 = dist2kernel(dist_mat2, scale=scale2, zero_diag=zero_diag, k=k)
    return K1, K2

