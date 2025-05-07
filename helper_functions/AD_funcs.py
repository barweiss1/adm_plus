import gc

import scipy as sci
import numpy as np
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
from time import time
from helper_functions.utils import replace_nan_inf, print_memory_usage

# ------------- Functions for alternating diffusion and variants ------------


# Function that column normalizes the kernel
# def column_norm(K):
#     # Noramalize kernel matrix columns to create Markov transition matrix
#     col_sum = np.sum(K, axis=0)
#     inv_sum = col_sum ** (-1)
#     norm_mat = np.diag(inv_sum.T)
#     P = K @ norm_mat  # normalize columns
#     return P

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
        sigma = np.median(dist_mat) * scale # popular choice for sigma
    elif mode == 'scale':
        sigma = scale
    else:
        raise ValueError(f'invalid mode {mode}')
    K_mat = np.exp(-dist_mat ** 2 / (sigma ** 2))
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

        # Initialize the kernel matrix with zeros
        K_mat = np.zeros_like(dist_mat)

        # Compute the kernel for the K nearest points
        for i in range(dist_mat.shape[0]):
            nearest_indices = np.argsort(dist_mat[i])[1:k + 1]  # Excluding self
            K_mat[i, nearest_indices] = np.exp(-dist_mat[i, nearest_indices] ** 2 / (scale * sigma ** 2))

    else:
        # Compute sigma using the median of the non-zero elements in the distance matrix
        sigma = np.median(dist_mat[dist_mat > 0])

        # Compute the kernel for all points
        K_mat = np.exp(-dist_mat ** 2 / (scale * sigma ** 2))

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

    # Compute Kernel for KNN
    K_mat[roots_idx, neighbors_idx] = np.exp(-distances ** 2 / (scale * sigma ** 2))
    K_mat[neighbors_idx, roots_idx] = np.exp(-distances ** 2 / (scale * sigma ** 2))

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

    # Compute Kernel
    sigma = np.median(dist_mat)  # popular choice for sigma

    if mode == 'median':
        sigma = np.median(dist_mat) * scale  # popular choice for sigma
    elif mode == 'scale':
        sigma = scale
    else:
        raise ValueError(f'invalid mode {mode}')
    A = np.exp(-dist_mat ** 2 / (sigma ** 2))
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

    # Initialize a sparse matrix for the bipartite KNN graph
    n_samples_1 = data_points1.shape[0]
    n_samples_2 = data_points2.shape[0]
    A = sp.lil_matrix((n_samples_1, n_samples_2), dtype=float)

    # Set True for nearest neighbors in the bipartite KNN graph
    idx1 = np.arange(n_samples_1)
    for kk in range(k):
        A[idx1, indices[idx1, kk]] = np.exp(-distances[:, kk] ** 2 / (scale * sigma ** 2))

    # Normalize columns of K_mat
    P_L = row_norm(A)

    # Normalize columns of K_mat Transpose
    P_R = row_norm(A.T)
    end_time = time()
    print(f' Kernel computation finished, in {end_time - start_time} seconds')
    return A, P_L, P_R


# Function to create roseland kernel matrix
def Roseland_Asym_Kernel(data_points1, data_points2, scale=2):
    # data_points2 should be the reference set!
    start_time = time()
    # Calculate the kernel matrix
    dist_mat = sci.spatial.distance.cdist(data_points1, data_points2, metric='euclidean')

    # Compute Kernel
    sigma = np.median(dist_mat)  # popular choice for sigma
    K_mat = np.exp(-dist_mat ** 2 / (scale * sigma ** 2))

    end_time = time()
    print(f' Kernel computation finished, in {end_time - start_time} seconds')
    return K_mat


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


def diffusion_map(s=None, embed_dim=2, t=1, K=None, stabilize=False, tol=1e-8, solver='arpack', return_vecs=False):
    if K is None:
        if s is None:
            raise ValueError("s must be provided if Kernel not provided")
        _, P = Create_Transition_Mat(s)
    else:
        P = row_norm(K)
    # replace the NaN and inf vals with 0
    P = replace_nan_inf(P, replacement_value=0)
    if sp.issparse(P):
        P = sp.csr_matrix(P)
    # compute embedding with eigen-decomposition
    return compute_embedding(P, embed_dim, t, stabilize, tol=tol, solver=solver, return_vecs=return_vecs)


def compute_embedding(P, embed_dim, t=1, stabilize=False, max_iter=3000, tol=1e-8, solver='arpack', return_vecs=False):
    # print_memory_usage('entered compute_embedding')
    if stabilize:
        eps = 1e-8
        n = P.shape[0]
        P = P + eps * np.eye(n)
    if solver == 'arpack':
        try:
            vals, vecs = sci.sparse.linalg.eigs(P, k=embed_dim + 1, which='LM', maxiter=max_iter, tol=tol)
        except sci.sparse.linalg.ArpackNoConvergence as e:
            # Handle the convergence issue
            vals = e.eigenvalues
            vecs = e.eigenvectors
            print(f"ARPACK did not converge within {max_iter} iterations. Using {len(vals)} convergent vectors.")
    elif solver == 'svd':
        _, vals, vecs_r = sci.sparse.linalg.svds(P, k=embed_dim + 1, which='LM')
        vecs = vecs_r.T
    elif solver == 'randomized':
        # we use SVD to calculate the EVD efficiently, this trick works only for PSD matrices
        svd = TruncatedSVD(n_components=embed_dim + 1, algorithm='randomized')
        s_vecs_l = svd.fit_transform(P)  # returns left_vectors * singular values
        s_vals = svd.singular_values_
        s_vecs_l = s_vecs_l / s_vals  # divide by singular values to get vecs
        s_vecs_r = svd.components_
        vecs = s_vecs_r  # eigenvectors are the singular vectors
        vals = s_vals ** 2  # eigenvalues are the singular values squared
    else:
        raise ValueError(f"Unsupported solver: {solver}")
    vals, vecs = sort_evd_components(vals, vecs)
    if return_vecs:
        return vals, vecs
    else:
        return np.real((vals[1:] ** t) * vecs[:, 1:])


def alternating_diffusion(s1=None, s2=None, embed_dim=2, t=1, K1=None, K2=None, stabilize=False, tol=0,
                          solver='arpack', delete_kernels=False, return_vecs=False):
    """"
        computes alternating diffusion map embedding
        Important note: the order of all measurements must be aligned (s1[i,:] and s2[i,:]
            must all be aligned measurements)
        s1 - (samples, features) sensor 1 samples
        s2 - (samples, features) sensor 2 samples
        embed_dim - embedding dimension
        t - diffusion time scale
        K1 (optional) - first sensor similarity kernel (if isn't provided the function will compute it
        K2 (optional) - second sensor similarity kernel (if isn't provided the function will compute it
        stabilize (optional) - whether to stabilize the matrix to find eigenvalues
        return: embedding (samples, embed_dim)
        and kernels K1,K2 for use in other functions
        """

    # calculate kernels - if provided use them, else calculate
    if K1 is None or K2 is None:
        if s1 is None or s2 is None:
            raise ValueError("s1 and s2 must be provided if Kernels are not provided")
        # the kernels here are normalized
        _, P1 = Create_Transition_Mat(s1)
        _, P2 = Create_Transition_Mat(s2)

    else:
        P1 = row_norm(K1)
        P2 = row_norm(K2)
    if delete_kernels:
        del K1
        del K2
        gc.collect()
    # alternating diffusion kernel
    # print_memory_usage('P1 and P2 computed')
    P1 = replace_nan_inf(P1, replacement_value=0)
    P2 = replace_nan_inf(P2, replacement_value=0)
    P1 = P1 @ P2
    # print_memory_usage('computed P, pre-deleted Kernels')
    if delete_kernels:
        del P2
        gc.collect()
    # print_memory_usage('computed P and deleted Kernels')
    # replace the NaN and inf vals with 0
    P1 = replace_nan_inf(P1, replacement_value=0)
    if sp.issparse(P1):
        P1 = sp.csr_matrix(P1)
    # compute embedding with eigen-decomposition
    return compute_embedding(P1, embed_dim, t, stabilize, tol=tol, solver=solver, return_vecs=return_vecs)


def apmc_embed(s1_ref, s1_full, s2_ref, embed_dim, scale=1, A1=None, K2_ref=None, return_vecs=False, solver='arpack'):
    """"
                Computes Anchor Partial Mutliview Clustring embedding (Guo 2019 AAAI)

                Inputs:
                s1_full - (samples_total, features) sensor 1 samples from the total set
                s1_ref - (samples_ref, features) sensor 1 samples from the reference set
                s2_ref - (samples_ref, features) sensor 2 samples from the reference set
                embed_dim - embedding dimension
                A1 (optional) - (samples_total, samples_ref) first sensor total to reference similarity kernel (if isn't provided the function will compute it)
                K2 (optional) - (samples_ref, samples_ref) second sensor similarity kernel (if isn't provided the function will compute it)
                ffbb - flag indicating if the used kernel is the FFBB kernel (P1_LP2Q2P1_R) or FBFB kernel (P1_LQ2P2Q1_R)

                return: embedding (samples, embed_dim)
                and kernels A1, K2
                """

    # calculate kernels - if provided use them, else calculate
    if A1 is None or K2_ref is None:
        A1, _, _ = Create_Asym_Tran_Kernel(s1_full, s1_ref, scale=scale)
        K2_ref, _ = Create_Transition_Mat(s2_ref, scale=scale)
    Nr = K2_ref.shape[0]
    Q1_p = row_norm(A1)
    Q2_ref = row_norm(K2_ref)
    # unify kernels
    Z = Q1_p
    Z[:Nr, :] = 0.5*(Q1_p[:Nr, :] + Q2_ref)

    # normalize Z
    col_sum = np.array(Z.sum(axis=0)).flatten()
    # Handle division by zero in col_sum
    col_sum[col_sum == 0] = 1
    # Compute the inverse of column sums
    inv_sum = col_sum ** (-0.5)
    # remove inf values in inv_sum
    inv_sum[np.isinf(inv_sum)] = 1
    # Create diagonal matrix with inverse column sums
    norm_mat = sp.diags(inv_sum)
    # Normalize columns
    Z_norm = Z @ norm_mat

    vals, vecs = SVD_trick(Z_norm, embed_dim, solver=solver)
    if return_vecs:
        return vals, vecs
    else:
        return vecs[:, 1:embed_dim + 1]


def ncca(s1_ref, s1_full, s2_ref, embed_dim, scale=1, A1=None, K2_ref=None, return_vecs=False):
    """"
                Computes NCCA (Michaeli 2016)

                Inputs:
                s1_full - (samples_total, features) sensor 1 samples from the total set
                s1_ref - (samples_ref, features) sensor 1 samples from the reference set
                s2_ref - (samples_ref, features) sensor 2 samples from the reference set
                embed_dim - embedding dimension
                A1 (optional) - (samples_total, samples_ref) first sensor total to reference similarity kernel (if isn't provided the function will compute it)
                K2 (optional) - (samples_ref, samples_ref) second sensor similarity kernel (if isn't provided the function will compute it)
                ffbb - flag indicating if the used kernel is the FFBB kernel (P1_LP2Q2P1_R) or FBFB kernel (P1_LQ2P2Q1_R)

                return: embedding (samples, embed_dim)
                and kernels A1, K2
                """

    # calculate kernels - if provided use them, else calculate
    if A1 is None or K2_ref is None:
        A1, _, _ = Create_Asym_Tran_Kernel(s1_full, s1_ref, scale=scale)
        K2_ref, _ = Create_Transition_Mat(s2_ref, scale=scale)
    Nr = K2_ref.shape[0]
    P1_ref = row_norm(A1[:Nr, :])
    Q2_ref = column_norm(K2_ref)
    # delete redundant kernels
    P_ref = replace_nan_inf(P1_ref@Q2_ref)
    # select svd kernel based on required method
    s_vecs_l, s_vals, s_vecs_r = sci.sparse.linalg.svds(P_ref, k=embed_dim + 1, which='LM')
    # sort SVD components
    s_vecs_l, s_vals, s_vecs_r = sort_svd_components(s_vecs_l, s_vals, s_vecs_r)
    # use Nystrom interpolation for extending the embedding outside the reference set
    P1_rest = row_norm(A1[Nr:, :])
    l_vecs_extend = P1_rest @ Q2_ref @ s_vecs_r.T * (1 / s_vals)
    vecs = np.concatenate((s_vecs_l, l_vecs_extend), axis=0)
    if return_vecs:
        return s_vals[1:embed_dim+1], vecs[:, 1:embed_dim+1]
    else:
        return vecs[:, 1:embed_dim+1]


def kcca(s1_ref, s1_full, s2_ref, embed_dim, scale=1, reg=1e-3, A1=None, K2_ref=None, return_vecs=False):
    """"
                Computes KCCA (Fukumizu 2007)

                Inputs:
                s1_full - (samples_total, features) sensor 1 samples from the total set
                s1_ref - (samples_ref, features) sensor 1 samples from the reference set
                s2_ref - (samples_ref, features) sensor 2 samples from the reference set
                embed_dim - embedding dimension
                t - diffusion time scale
                A1 (optional) - (samples_total, samples_ref) first sensor total to reference similarity kernel (if isn't provided the function will compute it)
                K2 (optional) - (samples_ref, samples_ref) second sensor similarity kernel (if isn't provided the function will compute it)
                ffbb - flag indicating if the used kernel is the FFBB kernel (P1_LP2Q2P1_R) or FBFB kernel (P1_LQ2P2Q1_R)

                return: embedding (samples, embed_dim)
                and kernels A1, K2
                """
    # calculate kernels - if provided use them, else calculate
    if A1 is None or K2_ref is None:
        A1, _, _ = Create_Asym_Tran_Kernel(s1_full, s1_ref, scale=scale)
        K2_ref, _ = Create_Transition_Mat(s2_ref, scale=scale)
    Nr = K2_ref.shape[0]
    K1_ref = A1[:Nr, :]
    one_N = np.ones((Nr, Nr)) / Nr
    K1_ref_centered = K1_ref - one_N @ K1_ref - K1_ref @ one_N + one_N @ K1_ref @ one_N
    K2_ref_centered = K2_ref - one_N @ K2_ref - K2_ref @ one_N + one_N @ K2_ref @ one_N
    # regularize the kernels
    K1_ref_centered += reg * np.eye(Nr)
    K2_ref_centered += reg * np.eye(Nr)
    # Solve the generalized eigenvalue problem
    eigvals, eigvecs = sp.linalg.eigsh(K1_ref_centered @ K2_ref_centered, M=K1_ref_centered @ K1_ref_centered,
                                       k=embed_dim+3, which='LM')
    # Sort eigenvectors by eigenvalue
    indices = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, indices]
    # use Nystrom interpolation for extending the embedding outside the reference set
    K1_oor = A1[Nr:, :]
    mean_K1_oor = np.mean(K1_oor, axis=0)
    K1_oor_centered = K1_oor - mean_K1_oor - np.mean(K1_oor, axis=1, keepdims=True) + np.mean(K1_ref)
    vecs_ref = K1_ref_centered @ eigvecs
    vecs_oor = K1_oor_centered @ eigvecs
    vecs = np.concatenate((vecs_ref[:, 1:embed_dim+1], vecs_oor[:, 1:embed_dim+1]), axis=0)

    if return_vecs:
        return eigvals[1:embed_dim+1], vecs
    else:
        return vecs


def kcca_impute(s1_full, s2_ref, embed_dim, scale=1, reg=1e-3, K1=None, K2_ref=None, return_vecs=False):
    """"
                    Computes KCCA with incomplete view (Trivedi 2010)

                    Inputs:
                    s1_full - (samples_total, features) sensor 1 samples from the total set
                    s1_ref - (samples_ref, features) sensor 1 samples from the reference set
                    s2_ref - (samples_ref, features) sensor 2 samples from the reference set
                    embed_dim - embedding dimension
                    t - diffusion time scale
                    A1 (optional) - (samples_total, samples_ref) first sensor total to reference similarity kernel (if isn't provided the function will compute it)
                    K2 (optional) - (samples_ref, samples_ref) second sensor similarity kernel (if isn't provided the function will compute it)
                    ffbb - flag indicating if the used kernel is the FFBB kernel (P1_LP2Q2P1_R) or FBFB kernel (P1_LQ2P2Q1_R)

                    return: embedding (samples, embed_dim)
                    and kernels A1, K2
                    """
    # calculate kernels - if provided use them, else calculate
    if K1 is None or K2_ref is None:
        K1, _ = Create_Transition_Mat(s1_full, scale=scale)
        K2_ref, _ = Create_Transition_Mat(s2_ref, scale=scale)

    # compute first view graph laplacian for smoothness imputation of the second view
    N = K1.shape[0]
    Nr = K2_ref.shape[0]
    D1 = np.diag(np.sum(K1, axis=0))
    L1 = D1 - K1
    K2 = np.zeros_like(K1)

    # impute second kernel - based on the closed form solution in the paper
    epsilon = 1e-5
    Lmm_inv = np.linalg.inv(L1[Nr:, Nr:] + epsilon * np.eye(L1[Nr:, Nr:].shape[0]))
    K2[:Nr, :Nr] = K2_ref
    K2[Nr:, :Nr] = -1 * Lmm_inv @ L1[Nr:, :Nr] @ K2_ref
    K2[:Nr, Nr:] = K2[Nr:, :Nr].T
    K2[Nr:, Nr:] = Lmm_inv @ L1[Nr:, :Nr] @ K2_ref @ L1[:Nr, Nr:] @ Lmm_inv

    return kcca_full(s1_full, s2_ref, embed_dim, scale, reg, K1=K1, K2=K2, return_vecs=return_vecs)


def kcca_full(s1_full, s2_full, embed_dim, scale=1, reg=1e-3, K1=None, K2=None, return_vecs=False):
    # similar implementation to the one use in the deeptime package
    """"
                Computes KCCA (Fukumizu 2007)

                Inputs:
                s1_full - (samples_total, features) sensor 1 samples from the total set
                s1_ref - (samples_ref, features) sensor 1 samples from the reference set
                s2_ref - (samples_ref, features) sensor 2 samples from the reference set
                embed_dim - embedding dimension
                t - diffusion time scale
                A1 (optional) - (samples_total, samples_ref) first sensor total to reference similarity kernel (if isn't provided the function will compute it)
                K2 (optional) - (samples_ref, samples_ref) second sensor similarity kernel (if isn't provided the function will compute it)
                ffbb - flag indicating if the used kernel is the FFBB kernel (P1_LP2Q2P1_R) or FBFB kernel (P1_LQ2P2Q1_R)

                return: embedding (samples, embed_dim)
                and kernels A1, K2
                """
    # calculate kernels - if provided use them, else calculate
    if K1 is None or K2 is None:
        K1, _ = Create_Transition_Mat(s1_full, scale=scale)
        K2, _ = Create_Transition_Mat(s2_full, scale=scale)
    N = K1.shape[0]
    I = np.eye(N)
    I_center = I - np.full((N, N), fill_value=1. / N)  # centering matrix
    G_0 = np.linalg.multi_dot([I_center, K1, I_center])
    G_1 = np.linalg.multi_dot([I_center, K2, I_center])

    K = sci.linalg.solve(G_0 + reg * I, G_0, assume_a='sym')
    Ak = sci.linalg.solve(G_1 + reg * I, G_1, assume_a='sym')
    A = K @ Ak

    vals, vecs = compute_embedding(A, embed_dim, return_vecs=True)

    if return_vecs:
        return vals[:embed_dim], np.real(vecs[:, :embed_dim])
    else:
        return np.real(vecs[:, :embed_dim])


def nystrom_ad(s1_ref, s1_full, s2_ref, embed_dim, t=1, A1=None, K2_ref=None, return_vecs=False):
    """"
                Computes Nystom ADM extension (Dov et al. 2016)

                Inputs:
                s1_full - (samples_total, features) sensor 1 samples from the total set
                s1_ref - (samples_ref, features) sensor 1 samples from the reference set
                s2_ref - (samples_ref, features) sensor 2 samples from the reference set
                embed_dim - embedding dimension
                t - diffusion time scale
                A1 (optional) - (samples_total, samples_ref) first sensor total to reference similarity kernel (if isn't provided the function will compute it)
                K2 (optional) - (samples_ref, samples_ref) second sensor similarity kernel (if isn't provided the function will compute it)
                ffbb - flag indicating if the used kernel is the FFBB kernel (P1_LP2Q2P1_R) or FBFB kernel (P1_LQ2P2Q1_R)

                return: embedding (samples, embed_dim)
                and kernels A1, K2
                """

    # calculate kernels - if provided use them, else calculate
    if A1 is None or K2_ref is None:
        A1, _, _ = Create_Asym_Tran_Kernel(s1_full, s1_ref)
        K2_ref, _ = Create_Transition_Mat(s2_ref)
    Nr = K2_ref.shape[0]
    P2_ref = row_norm(K2_ref)
    # select svd kernel based on required method
    vals, vecs = alternating_diffusion(embed_dim=embed_dim, t=t, K1=A1[:Nr, :], K2=K2_ref, return_vecs=True)
    # use Nystrom interpolation for extending the embedding outside the reference set
    P1_rest = row_norm(A1[Nr:, :])
    vecs_extend = P1_rest @ P2_ref @ vecs * (1 / vals)
    vecs = np.concatenate((vecs, vecs_extend), axis=0)
    if return_vecs:
        return vals[1:embed_dim + 1], vecs[:, 1:embed_dim + 1]
    else:
        return np.real((vals[1:embed_dim + 1] ** t) * vecs[:, 1:embed_dim + 1])


# implements trick to compute the EVD of a Gram matrix BB^T from the SVD of B
def SVD_trick(P, embed_dim, solver='arpack', side='left'):
    svd = TruncatedSVD(n_components=embed_dim + 1, algorithm=solver)
    s_vecs_l = svd.fit_transform(P)  # returns left_vectors * singular values
    s_vals = svd.singular_values_
    s_vecs_l = s_vecs_l / s_vals  # divide by singular values to get vecs
    s_vecs_r = svd.components_
    s_vecs_l, s_vals, s_vecs_r = sort_svd_components(s_vecs_l, s_vals, s_vecs_r)
    # eigenvectors are the singular vectors
    if side == 'left':
        vecs = s_vecs_l
    elif side == 'right':
        vecs = s_vecs_r
    else:
        raise ValueError(f'Unknown side: {side}')
    vals = s_vals ** 2  # eigenvalues are the singular values squared
    return vals, vecs


# trick for computing EVD of M1M2 through M2M1 used by LAD paper
def LAD_trick(M1, M2, dim, fast_comp=True, solver='arpack', delete_kernels=False, stabilize=False, tol=0):
    """

    :param M1: N X NR
    :param M2: NR X N
    :param dim: dimension of the embedding space
    :param fast_comp: flag indicating whether to compute fast or implicitly
    :return: embedding
    """
    # compute evd efficiently
    if fast_comp:
        if delete_kernels:
            M2 = M2 @ M1
            # print_memory_usage('M2 @ M1 computed')
            vals, vecs = compute_embedding(M2, embed_dim=dim + 1, return_vecs=True, solver=solver,
                                           tol=tol, stabilize=stabilize)
            del M2
            gc.collect()
        else:
            vals, vecs = compute_embedding(M2 @ M1, embed_dim=dim + 1, return_vecs=True, solver=solver,
                                           tol=tol, stabilize=stabilize)
        vecs_extended = M1 @ vecs  # extend to eigenvectors of M1M2 (N X N) with algebric trick
    else:
        vals, vecs_extended = compute_embedding(M1 @ M2, embed_dim=dim + 1, return_vecs=True, solver=solver,
                                       tol=tol, stabilize=stabilize)
    return vals, vecs_extended


def ad_forward_only(s1_full=None, s1_ref=None, s2_ref=None, embed_dim=2, t=1, A1=None, K2=None, fast_comp=True,
                    solver='arpack', delete_kernels=False, stabilize=False, tol=0, return_vecs=False):
    """"
            computes forward-only alternating diffusion map extension, extends the embedding
            with missing measurements from the second view.
            Important note: the order of all measurements must be aligned (s1_full[i,:], s1_ref[i,:] and s2_ref[i,:]
            must all be aligned measurements)

            Inputs:
            s1_full - (samples_total, features) sensor 1 samples from the total set
            s1_ref - (samples_ref, features) sensor 1 samples from the reference set
            s2_ref - (samples_ref, features) sensor 2 samples from the reference set
            embed_dim - embedding dimension
            t - diffusion time scale
            K1 (optional) - first sensor similarity kernel (if isn't provided the function will compute it
            K2 (optional) - second sensor similarity kernel (if isn't provided the function will compute it
            fast_comp - flag determining whether the EVD computation is explicit or uses LeAD trick

            return: embedding (samples, embed_dim)
            and kernels A1,K2
            """

    # calculate kernels - if provided use them, else calculate
    if A1 is None or K2 is None:
        A1, _, _ = Create_Asym_Tran_Kernel(s1_full, s1_ref)
        K2, _ = Create_Transition_Mat(s2_ref)
    # normalize matrices
    Q1_p = column_norm(A1)  # Q_1^+
    Q1_m = column_norm(A1.T)  # Q_1^-
    Q2_ref = column_norm(K2)

    if delete_kernels:
        del A1
        del K2
        gc.collect()
    # replace the NaN and inf vals with 0
    Q_p = replace_nan_inf(Q1_p@Q2_ref, replacement_value=0)
    if sp.issparse(Q_p):
        Q_p = sp.csr_matrix(Q_p)
    # print_memory_usage('P_L Computed')
    if delete_kernels:
        del Q1_p
        gc.collect()
    # replace the NaN and inf vals with 0
    Q_m = replace_nan_inf(Q2_ref@Q1_m, replacement_value=0)
    if sp.issparse(Q_m):
        Q_m = sp.csr_matrix(Q_m)
    # print_memory_usage('P_R Computed')
    if delete_kernels:
        del Q2_ref
        del Q1_m
        gc.collect()

    # embed based on the kernel Q_1^+@Q2_ref@Q2_ref@Q_1^- using LeAD trick
    vals, vecs = LAD_trick(Q_p, Q_m, dim=embed_dim, fast_comp=fast_comp, solver=solver, delete_kernels=delete_kernels,
                            tol=tol, stabilize=stabilize)
    if return_vecs:
        return vals, vecs
    else:
        return np.real((vals[1:embed_dim + 1]**t) * vecs[:, 1:embed_dim + 1])


def ad_backward_only(s1_full=None, s1_ref=None, s2_ref=None, embed_dim=2, t=1, A1=None, K2=None, fast_comp=True,
                    solver='arpack', delete_kernels=False, stabilize=False, tol=0, return_vecs=False):
    """"
            computes forward-only alternating diffusion map extension, extends the embedding
            with missing measurements from the second view.
            Important note: the order of all measurements must be aligned (s1_full[i,:], s1_ref[i,:] and s2_ref[i,:]
            must all be aligned measurements)

            Inputs:
            s1_full - (samples_total, features) sensor 1 samples from the total set
            s1_ref - (samples_ref, features) sensor 1 samples from the reference set
            s2_ref - (samples_ref, features) sensor 2 samples from the reference set
            embed_dim - embedding dimension
            t - diffusion time scale
            K1 (optional) - first sensor similarity kernel (if isn't provided the function will compute it
            K2 (optional) - second sensor similarity kernel (if isn't provided the function will compute it
            fast_comp - flag determining whether the EVD computation is explicit or uses LeAD trick

            return: embedding (samples, embed_dim)
            and kernels A1,K2
            """

    # calculate kernels - if provided use them, else calculate
    if A1 is None or K2 is None:
        A1, _, _ = Create_Asym_Tran_Kernel(s1_full, s1_ref)
        K2, _ = Create_Transition_Mat(s2_ref)

    # normalize kernels
    P1_p = row_norm(A1)
    P1_m = row_norm(A1.T)
    P2_ref = row_norm(K2)

    if delete_kernels:
        del A1
        del K2
        gc.collect()
    # replace the NaN and inf vals with 0
    P_p = replace_nan_inf(P1_p@P2_ref, replacement_value=0)
    if sp.issparse(P_p):
        P_p = sp.csr_matrix(P_p)
    # print_memory_usage('P_L Computed')
    if delete_kernels:
        del P1_p
        gc.collect()
    # replace the NaN and inf vals with 0
    P_m = replace_nan_inf(P2_ref@P1_m, replacement_value=0)
    if sp.issparse(P_m):
        P_m = sp.csr_matrix(P_m)
    # print_memory_usage('P_R Computed')
    if delete_kernels:
        del P2_ref
        del P1_m
        gc.collect()

    # embed based on the kernel P1L@P2_ref@P2_ref@P1_R using LeAD trick
    vals, vecs = LAD_trick(P_p, P_m, dim=embed_dim, fast_comp=fast_comp, solver=solver, delete_kernels=delete_kernels,
                            tol=tol, stabilize=stabilize)
    if return_vecs:
        return vals, vecs
    else:
        return np.real((vals[1:embed_dim + 1]**t) * vecs[:, 1:embed_dim + 1])


# a function that recieves the kernel matrices and computes the LeAD embedding
def LAD_embedding(s1_ref, s1_full, s2_ref, s2_full, dim=2, t=1, A1=None, A2=None, fast_comp=True,
                   solver='arpack', delete_kernels=False, stabilize=False, tol=0, return_vecs=False):
    # build kernels if not provided
    if A1 is None or A2 is None:
        A1 = Roseland_Asym_Kernel(s1_full, s1_ref)
        A2 = Roseland_Asym_Kernel(s2_full, s2_ref)
    # calculate second diffusion matrix
    temp_sum = np.array(A2.T @ np.sum(A2, axis=1)).flatten()
    D2_inv = sp.diags(1 / temp_sum)
    M2 = D2_inv @ A2.T  # N_R X N
    if delete_kernels:
        del A2
        gc.collect()
    # calculate first diffusion matrix
    temp_sum = np.array(A1 @ np.sum(M2, axis=1)).flatten()
    D1_inv = sp.diags(1 / temp_sum)
    M1 = D1_inv @ A1  # N X N_R
    if delete_kernels:
        del A1
        gc.collect()

    # replace the NaN and inf vals with 0
    M1 = replace_nan_inf(M1, replacement_value=0)
    if sp.issparse(M1):
        M1 = sp.csr_matrix(M1)
    # replace the NaN and inf vals with 0
    M2 = replace_nan_inf(M2, replacement_value=0)
    if sp.issparse(M2):
        M2 = sp.csr_matrix(M2)
    vals, vecs = LAD_trick(M1, M2, dim, fast_comp=fast_comp, solver=solver, delete_kernels=delete_kernels,
                            tol=tol, stabilize=stabilize)
    if return_vecs:
        return vals, vecs
    else:
        return np.real((vals[1:dim + 1]**t) * vecs[:, 1:dim + 1])


def alternating_roseland(s1_ref=None, s1_full=None, s2_ref=None, embed_dim=2, t=1, A1=None, K2=None,
                         solver='arpack', delete_kernels=False, return_vecs=False):
    """"
                computes alternating roseland map, extends the embedding
                with missing measurements from the second view.
                Important note: the order of all measurements must be aligned (s1_full[i,:], s1_ref[i,:] and s2_ref[i,:]
                must all be aligned measurements)

                Inputs:
                s1_full - (samples_total, features) sensor 1 samples from the total set
                s1_ref - (samples_ref, features) sensor 1 samples from the reference set
                s2_ref - (samples_ref, features) sensor 2 samples from the reference set
                embed_dim - embedding dimension
                t - diffusion time scale
                A1 (optional) - first sensor total to reference similarity kernel (if isn't provided the function will compute it)
                K2 (optional) - second sensor similarity kernel (if isn't provided the function will compute it)

                return: embedding (samples, embed_dim)
                and kernels A1,K2
                """

    # calculate kernels - if provided use them, else calculate
    if A1 is None or K2 is None:
        Nr = s2_ref.shape[0]
        A1, _, _ = Create_Asym_Tran_Kernel(s1_full, s1_ref)
        K2, _ = Create_Transition_Mat(s2_ref)

    # row_sum = A1 @ K2 @ K2 @ np.sum(A1.T, axis=1)
    # computing the row sum efficiently
    row_sum = np.sum(A1.T, axis=1)
    row_sum = K2 @ row_sum
    row_sum = K2 @ row_sum
    row_sum = A1 @ row_sum
    row_sum = np.array(row_sum).flatten()  # flatten row sum to eliminate singleton dimensions
    D_mh = sp.diags(row_sum ** -0.5)  # compute normalizing matrix
    # replace the NaN and inf vals with 0
    svd_kernel = replace_nan_inf(D_mh@A1@K2, replacement_value=0)
    if sp.issparse(svd_kernel):
        svd_kernel = sp.csr_matrix(svd_kernel)
    # delete redundant kernels
    if delete_kernels:
        del A1
        del K2
        gc.collect()
    # compute the EVD efficiently with SVD
    vals, vecs = SVD_trick(svd_kernel, embed_dim=embed_dim, solver=solver)
    vecs = D_mh@vecs  # convert to eigenvectors of D_inv@A1@K2@K2@A1.T
    # sort components
    if return_vecs:
        return vals, vecs
    else:
        return np.real((vals[1:embed_dim + 1] ** t) * vecs[:, 1:embed_dim + 1])


def ad_forward_backward(s1_ref=None, s1_full=None, s2_ref=None, embed_dim=2, t=1, A1=None, K2_ref=None, ffbb=False,
                        solver='arpack', delete_kernels=False, return_vecs=False):
    """"
                computes forward-backward alternating diffusion map extension, extends the embedding
                with missing measurements from the second view.
                Important note: the order of all measurements must be aligned (s1_full[i,:], s1_ref[i,:] and s2_ref[i,:]
                must all be aligned measurements)

                Inputs:
                s1_full - (samples_total, features) sensor 1 samples from the total set
                s1_ref - (samples_ref, features) sensor 1 samples from the reference set
                s2_ref - (samples_ref, features) sensor 2 samples from the reference set
                embed_dim - embedding dimension
                t - diffusion time scale
                A1 (optional) - first sensor total to reference similarity kernel (if isn't provided the function will compute it)
                K2 (optional) - second sensor similarity kernel (if isn't provided the function will compute it)
                ffbb - flag indicating if the used kernel is the FFBB kernel (P1_LP2Q2P1_R) or FBFB kernel (P1_LQ2P2Q1_R)

                return: embedding (samples, embed_dim)
                and kernels A1,K2
                """
    # print_memory_usage('Entered FBFB')
    # calculate kernels - if provided use them, else calculate
    if A1 is None or K2_ref is None:
        A1, _, _ = Create_Asym_Tran_Kernel(s1_full, s1_ref)
        K2_ref, _ = Create_Transition_Mat(s2_ref)

    if ffbb:
        Q1_p = column_norm(A1)  # Q1^+
        Q2_ref = column_norm(K2_ref)  # Q2 tilde
        Q = Q1_p @ Q2_ref
    else:
        Q1_p = column_norm(A1)
        P2_ref = row_norm(K2_ref)
        Q = Q1_p @ P2_ref
    # delete redundant kernels
    if delete_kernels:
        del A1
        del K2_ref
        del Q1_p
        gc.collect()
    if sp.issparse(Q):
        Q = sp.csr_matrix(Q)
    Q = replace_nan_inf(Q)
    vals, vecs = SVD_trick(Q, embed_dim, solver=solver)
    if return_vecs:
        return vals, vecs
    else:
        return np.real((vals[1:embed_dim + 1] ** t) * vecs[:, 1:embed_dim + 1])


def adm_plus(s1_ref=None, s1_full=None, s2_ref=None, embed_dim=2, t=1, A1=None, K2_ref=None, solver='arpack',
             delete_kernels=False, return_vecs=False):
    """"
                computes forward-backward alternating diffusion map extension, extends the embedding
                with missing measurements from the second view.
                Important note: the order of all measurements must be aligned (s1_full[i,:], s1_ref[i,:] and s2_ref[i,:]
                must all be aligned measurements)

                Inputs:
                s1_full - (samples_total, features) sensor 1 samples from the total set
                s1_ref - (samples_ref, features) sensor 1 samples from the reference set
                s2_ref - (samples_ref, features) sensor 2 samples from the reference set
                embed_dim - embedding dimension
                t - diffusion time scale
                A1 (optional) - first sensor total to reference similarity kernel (if isn't provided the function will compute it)
                K2 (optional) - second sensor similarity kernel (if isn't provided the function will compute it)
                ffbb - flag indicating if the used kernel is the FFBB kernel (P1_+P2Q2Q1_-) or BBFF kernel (Q1_+Q2P2P1_-)

                return: embedding (samples, embed_dim)
                and kernels A1,K2
                """
    # print_memory_usage('Entered FBFB')
    # calculate kernels - if provided use them, else calculate
    if A1 is None or K2_ref is None:
        A1, _, _ = Create_Asym_Tran_Kernel(s1_full, s1_ref)
        K2, _ = Create_Transition_Mat(s2_ref)

    Q1_p = column_norm(A1)  # Q1^- N_R X N
    Q2_ref = column_norm(K2_ref)  # Q2_ref N_R X N_R
    Q = Q1_p @ Q2_ref
    # delete redundant kernels
    if delete_kernels:
        del A1
        del K2_ref
        gc.collect()
    if sp.issparse(Q):
        Q = sp.csr_matrix(Q)
    Q = replace_nan_inf(Q)
    vals, vecs = SVD_trick(Q, embed_dim, solver=solver)
    if return_vecs:
        return vals, vecs
    else:
        return np.real((vals[1:embed_dim + 1] ** t) * vecs[:, 1:embed_dim + 1])


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


def embed_wrapper(s1_ref=None, s1_full=None, s2_ref=None, s2_full=None, method="forward_only",
                  embed_dim=2, t=1, K1=None, K2=None, solver='arpack', delete_kernels=False,
                  tol=0, stabilize=False, return_vecs=False):
    # this function embeds based to the specified method
    if method == "dm":
        return diffusion_map(s1_full, embed_dim=embed_dim, t=t, K=K1, solver=solver,
                             tol=tol, stabilize=stabilize, return_vecs=return_vecs)
    if method == "forward_only":
        return ad_forward_only(s1_full, s1_ref, s2_ref, embed_dim=embed_dim, t=t,
                               A1=K1, K2=K2, fast_comp=True, solver=solver, delete_kernels=delete_kernels,
                               tol=tol, stabilize=stabilize, return_vecs=return_vecs)
    if method == "backward_only":
        return ad_backward_only(s1_full, s1_ref, s2_ref, embed_dim=embed_dim, t=t,
                               A1=K1, K2=K2, fast_comp=True, solver=solver, delete_kernels=delete_kernels,
                               tol=tol, stabilize=stabilize, return_vecs=return_vecs)
    if method == "forward_only_slow":
        return ad_forward_only(s1_full, s1_ref, s2_ref, embed_dim=embed_dim, t=t,
                               A1=K1, K2=K2, fast_comp=False, solver=solver, delete_kernels=delete_kernels,
                               tol=tol, stabilize=stabilize, return_vecs=return_vecs)
    elif method == "ad":
        return alternating_diffusion(s1_full, s2_full, embed_dim=embed_dim, t=t,
                                     K1=K1, K2=K2, solver=solver, delete_kernels=delete_kernels, tol=tol,
                                     stabilize=stabilize, return_vecs=return_vecs)
    elif method == "ad_svd":
        return alternating_diffusion(s1_full, s2_full, embed_dim=embed_dim, t=t,
                                     K1=K1, K2=K2, solver='svd', delete_kernels=delete_kernels, tol=tol,
                                     stabilize=stabilize, return_vecs=return_vecs)
    elif method == "alternating_roseland":
        return alternating_roseland(s1_ref, s1_full, s2_ref, embed_dim,
                                    t=t, A1=K1, K2=K2, solver=solver, delete_kernels=delete_kernels)
    elif method == "lead":
        return LAD_embedding(s1_ref, s1_full, s2_ref, s2_full, dim=embed_dim,
                              t=t, A1=K1, A2=K2, solver=solver, delete_kernels=delete_kernels,
                              tol=tol, stabilize=stabilize, return_vecs=return_vecs)
    elif method == "ffbb":
        return ad_forward_backward(s1_ref, s1_full, s2_full,
                                   embed_dim=embed_dim, t=t, A1=K1, K2_ref=K2, ffbb=True, solver=solver,
                                   delete_kernels=delete_kernels, return_vecs=return_vecs)
    elif method == "fbfb":
        return ad_forward_backward(s1_ref, s1_full, s2_full,
                                   embed_dim=embed_dim, t=t, A1=K1, K2_ref=K2, ffbb=False, solver=solver,
                                   delete_kernels=delete_kernels, return_vecs=return_vecs)
    elif method == "adm_plus":
        return adm_plus(s1_ref, s1_full, s2_full,
                                   embed_dim=embed_dim, t=t, A1=K1, K2_ref=K2, solver=solver,
                                   delete_kernels=delete_kernels, return_vecs=return_vecs)
    elif method == "ncca":
        return ncca(s1_ref, s1_full, s2_full, embed_dim=embed_dim, A1=K1, K2_ref=K2, return_vecs=return_vecs)

    elif method == "nystrom":
        return nystrom_ad(s1_ref, s1_full, s2_ref, embed_dim, t=t, A1=K1, K2_ref=K2, return_vecs=return_vecs)

    elif method == "kcca":
        return kcca(s1_ref, s1_full, s2_ref, embed_dim=embed_dim, A1=K1, K2_ref=K2, return_vecs=return_vecs)

    elif method == "kcca_impute":
        return kcca_impute(s1_full, s2_ref, embed_dim=embed_dim, K1=K1, K2_ref=K2, return_vecs=return_vecs)

    elif method == "kcca_full":
        return kcca_full(s1_full, s2_full, embed_dim=embed_dim, K1=K1, K2=K2, return_vecs=return_vecs)
    elif method == 'apmc':
        return apmc_embed(s1_ref, s1_full, s2_ref, embed_dim=embed_dim, A1=K1, K2_ref=K2, return_vecs=return_vecs,
                          solver=solver)

