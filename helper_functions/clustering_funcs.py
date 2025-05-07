import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix
from scipy.linalg import sqrtm, det, inv


# align clusters in the same repetition, align based on label confusion
def align_clusters(labels_1, labels_2):
    """
    Aligns clusters between two sets of labels using the Hungarian algorithm.

    :param labels_1: np.array, clustering labels from method 1
    :param labels_2: np.array, clustering labels from method 2
    :return: np.array, aligned labels from method 2
    """
    # Create a confusion matrix between labels
    contingency_matrix = confusion_matrix(labels_1, labels_2)

    # Use the Hungarian algorithm to find the best matching clusters
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Create a mapping from labels_2 to labels_1
    label_mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # Apply the mapping to reassign labels_2
    aligned_labels_2 = np.array([label_mapping[label] for label in labels_2])

    return aligned_labels_2

# align clusters in different repetition, align based on distance between clusters, I chose the Jensen-Shannon distance
# under the gaussian distribution assumption, it only requires calculating.


def estimate_gaussian(samples):
    """
    Estimate the mean and covariance of a multivariate Gaussian from sample points.

    Parameters:
    - samples: (N, d) NumPy array, where N is the number of samples, and d is the dimension.

    Returns:
    - mean: (d,) NumPy array of estimated mean.
    - covariance: (d, d) NumPy array of estimated covariance matrix.
    """
    mean = np.mean(samples, axis=0)
    covariance = np.cov(samples, rowvar=False, bias=True)  # Bias=True for ML estimation
    return mean, covariance


def kl_gaussian(mu1, S1, mu2, S2):
    """
    Compute the KL divergence between two multivariate Gaussian distributions.

    Parameters:
    - mu1, S1: Mean and covariance of the first Gaussian.
    - mu2, S2: Mean and covariance of the second Gaussian.

    Returns:
    - KL divergence D_KL(N1 || N2)
    """
    d = mu1.shape[0]  # Dimensionality
    S2_inv = inv(S2)  # Inverse of covariance matrix S2

    trace_term = np.trace(S2_inv @ S1)
    mean_diff = (mu2 - mu1).reshape(-1, 1)  # Ensure column vector
    quadratic_term = mean_diff.T @ S2_inv @ mean_diff
    log_det_term = np.log(det(S2) / (det(S1) + 1e-10))  # Avoid log(0) by adding small constant

    return 0.5 * (trace_term + quadratic_term - d + log_det_term)


def jensen_shannon_gaussian_distance(samples1, samples2):
    """
    Compute the Jensen-Shannon Distance between two sets of sample points assuming Gaussian distributions.

    Parameters:
    - samples1, samples2: (N, d) NumPy arrays of sample points.

    Returns:
    - Jensen-Shannon Distance (JSD)
    """
    # Estimate Gaussian parameters from samples
    mu1, S1 = estimate_gaussian(samples1)
    mu2, S2 = estimate_gaussian(samples2)

    # Compute mixture covariance matrix
    S_mix = 0.5 * (S1 + S2)

    # Compute KL divergences
    kl_1 = kl_gaussian(mu1, S1, mu1, S_mix)
    kl_2 = kl_gaussian(mu2, S2, mu2, S_mix)

    # Compute the Jensen-Shannon Distance
    jsd = np.sqrt(0.5 * (kl_1 + kl_2))  # Taking square root for proper distance metric
    return jsd


# kde correlation distance calculation - correlation between the KDE estimates of each cluster
def get_unified_grid(samples_list, grid_points=80):
    """
    Compute a unified grid for KDE estimation with the same limits and dimensions as the samples.

    Parameters:
    - samples_list: List of all sample arrays (from both s1 and s2).
    - grid_points: Number of points per dimension in the unified grid.

    Returns:
    - unified_grid: (grid_points, d) NumPy array with the same dimension as the samples.
    """
    dim = samples_list[0].shape[0]  # Dimensionality of the data
    min_vals = np.min(np.vstack(samples_list), axis=0)
    max_vals = np.max(np.vstack(samples_list), axis=0)

    grid = [np.linspace(min_vals[i], max_vals[i], grid_points) for i in range(dim)]
    return np.array(grid).T  # Return the unified grid with the correct shape


def estimate_kde(samples, unified_grid):
    """
    Estimate the KDE for a set of sample points using a provided unified grid.

    Parameters:
    - samples: (N, d) NumPy array of data points.
    - unified_grid: 1D NumPy array (precomputed common grid).

    Returns:
    - kde_values: KDE estimated values at each grid point.
    """
    kde = gaussian_kde(samples.T)
    kde_values = kde(unified_grid.T)
    return kde_values / np.sum(kde_values)  # Normalize to sum to 1


def kde_correlation_distance(samples1, samples2):
    # Compute unified KDE grid across all clusters
    all_samples = list(samples1) + list(samples2)
    unified_grid = get_unified_grid(all_samples)

    # estimate KDEs
    kde1 = estimate_kde(samples1, unified_grid)  # KDE for cluster 1
    kde2 = estimate_kde(samples2, unified_grid)  # KDE for cluster 2

    return 1 - np.sum(kde1 * kde2)


def group_samples_by_label(samples, labels):
    """
    Groups samples based on unique labels.

    Parameters:
    - samples: (N, d) NumPy array of data points.
    - labels: (N,) NumPy array of cluster labels.

    Returns:
    - cluster_dict: Dictionary {label: corresponding_samples}.
    """
    unique_labels = np.unique(labels)
    return {label: samples[labels == label] for label in unique_labels}


def compute_distance_matrix(cluster_dict1, cluster_dict2, metric='kde_correlation'):
    """
    Compute the Jensen-Shannon Distance (JSD) matrix between two sets of clustered samples.

    Parameters:
    - cluster_dict1: Dictionary {label: samples} from dataset 1.
    - cluster_dict2: Dictionary {label: samples} from dataset 2.

    Returns:
    - jsd_matrix: (k, k) NumPy array of Jensen-Shannon Distances.
    - label_list1, label_list2: Ordered lists of labels corresponding to rows and columns.
    """
    label_list1 = sorted(cluster_dict1.keys())  # Sorted unique labels from set 1
    label_list2 = sorted(cluster_dict2.keys())  # Sorted unique labels from set 2

    k1, k2 = len(label_list1), len(label_list2)
    distance_matrix = np.zeros((k1, k2))

    for i, label1 in enumerate(label_list1):
        for j, label2 in enumerate(label_list2):
            if metric == 'gaussian':
                distance_matrix[i, j] = jensen_shannon_gaussian_distance(cluster_dict1[label1], cluster_dict2[label2])
            elif metric == 'kde_correlation':
                distance_matrix[i, j] = kde_correlation_distance(cluster_dict1[label1], cluster_dict2[label2])
            else:
                raise ValueError(f'Unsupported metric {metric}.')

    return distance_matrix, label_list1, label_list2


def align_clusters_different_rep(s1, label1, s2, label2, metric='jensen-shannon'):
    """
    Align clusters from two datasets using Jensen-Shannon Distance and the Hungarian Algorithm.

    Parameters:
    - s1, s2: (N, d) NumPy arrays of sample points.
    - label1, label2: (N,) NumPy arrays of cluster labels.

    Returns:
    - alignment: Dictionary {cluster_label_in_s1: best_matching_cluster_label_in_s2}.
    """
    # Group samples by their cluster labels
    clusters1 = group_samples_by_label(s1, label1)
    clusters2 = group_samples_by_label(s2, label2)

    # compute distance between clusters
    distance_matrix, labels1, labels2 = compute_distance_matrix(clusters1, clusters2, metric=metric)

    # Solve assignment using the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Create a mapping from labels_2 to labels_1
    label_mapping = {col: row for row, col in zip(row_indices, col_indices)}

    # Apply the mapping to reassign labels_2
    aligned_labels_2 = np.array([label_mapping[label] for label in label2])

    return aligned_labels_2

