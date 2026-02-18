import os
from time import time

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import json
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.io import loadmat
from scipy.linalg import svd
from helper_functions.embed_utils import prep_kernels
from helper_functions.embed_methods import embed_wrapper, fimvc_via_embed
from helper_functions.embed_methods import (OPT_METHODS, FULL_VIEW_DIFFUSION_METHODS, PARTIAL_VIEW_DIFFUSION_METHODS,
                                            FULL_VIEW_KERNEL_METHODS, PARTIAL_VIEW_KERNEL_METHODS, SINGLE_VIEW_METHODS)
from helper_functions.plotting_funcs import plot_embed, plot_embed_tsne
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


# ---------------- Data loading and pre-processing functions ----------------

def load_data(data_path, data_type='smooth'):
    if data_type == 'smooth':
        data = loadmat(f'{data_path}/all_mats_b0.1.mat')
        LR = data['all_mats_LR_smooth']
        RL = data['all_mats_RL_smooth']
    elif data_type == 'correlation':
        data = loadmat(f'{data_path}/all_mats_cc_NOGSR.mat')
        LR = data['all_mats_LR_cc']
        RL = data['all_mats_RL_cc']
    else:
        print(f'Data type {data_type} not supported')
    return LR, RL


def load_data_from_pkl(path, method):
    with open(f"{path}.pkl", 'rb') as fp:
        embed_dict = pickle.load(fp)
    results = embed_dict['df']
    ref_indicator = embed_dict['ref_indicator']
    task_labels_batch = embed_dict['labels_batch']
    if method in OPT_METHODS:
        embed_full = embed_dict['embed_full']
        embed_full_val = embed_dict['embed_full_val']
        return embed_full, embed_full_val, task_labels_batch, results
    else:
        vecs = embed_dict['vecs']
        vals = embed_dict['vals']
        vecs_val = embed_dict['vecs_val']
        vals_val = embed_dict['vals_val']
        return vecs, vals, vecs_val, vals_val, task_labels_batch, results


def prepare_data(data_LR, data_RL):
    '''
    :param data_LR: data n_edges x n_subjects x n_tasks - LR scan pattern FCN data
    :param data_LR: data n_edges x n_subjects x n_tasks - RL scan pattern FCN data
    :return stacked_data: n_subjects * n_tasks x n_edges - the stacked FCN data such
    task_label: n_subjects * n_tasks label vector indicating the task for each FCN
    subject_label: n_subjects * n_tasks label vector indicating the subject for each FCN
    '''
    n_edges = data_LR.shape[0]
    n_subjects = data_LR.shape[1]
    n_tasks = data_LR.shape[2]
    # stack data
    stacked_data_LR = data_LR.reshape(n_edges, n_tasks * n_subjects).T
    stacked_data_RL = data_RL.reshape(n_edges, n_tasks * n_subjects).T
    task_labels = np.tile(np.arange(n_tasks), n_subjects)
    subject_labels = np.repeat(np.arange(n_subjects), n_tasks)

    return stacked_data_LR, stacked_data_RL, task_labels, subject_labels


def calc_affine_invariant_dist(data1, data2, reg=False, diag_fill=2):
    '''
    Calculates the affine invariant metric between 2 edge sets representing SPD matrices
    :param data1: n_edges long vector representing first SPD matrix
    :param data2: n_edges long vector representing second SPD matrix
    :param reg: flag indicating whether to add lambda to the diagonal for regularization
    :return: affine-invariant distance between matrices
    '''
    # convert into matrices
    A = squareform(data1)
    B = squareform(data2)
    if reg:
        np.fill_diagonal(A, diag_fill)
        np.fill_diagonal(B, diag_fill)
    # Compute the eigenvalues and eigenvectors of A
    eigvals_A, eigvecs_A = np.linalg.eigh(A)

    # Compute the matrix B in the basis of the eigenvectors of A
    A_inv_sqrt = eigvecs_A @ np.diag(1.0 / np.sqrt(eigvals_A)) @ eigvecs_A.T
    C = A_inv_sqrt @ B @ A_inv_sqrt

    # Compute the eigenvalues of C
    eigvals_C = np.linalg.eigvalsh(C)

    # Compute the log of eigenvalues and their norm
    log_eigvals_C = np.log(eigvals_C)
    distance = np.linalg.norm(log_eigvals_C)

    return distance


def calc_affine_invariant_dist_svd(data1, data_comp, reg=False, diag_fill=2):
    '''
    Calculates the affine invariant metric between 2 edge sets representing SPD matrices
    :param data1: n_edges long vector representing first SPD matrix
    :param data2: n_edges long vector representing second SPD matrix
    :param reg: flag indicating whether to add lambda to the diagonal for regularization
    :return: affine-invariant distance between matrices
    '''
    # convert into matrices
    A = squareform(data1)
    if reg:
        np.fill_diagonal(A, diag_fill)
    # compute SVD
    u, s, _ = svd(A)

    # lift small eigenvalues
    s[s < 1e-3] = 1e-3
    S_inv_sqrt = np.diag(s ** -0.5)
    X_mod = u @ S_inv_sqrt @ u.T

    # compare to the rest of the matrices
    mat_num = data_comp.shape[0]
    dists = np.zeros(mat_num)
    for i in range(mat_num):
        B = squareform(data_comp[i, :])
        if reg:
            np.fill_diagonal(B, diag_fill)
        M = X_mod @ B @ X_mod
        _, s_m, _ = svd(M)
        dists[i] = np.sqrt(np.sum(np.log(np.diag(s_m))**2))

    return dists


def calculate_distances(data_LR, data_RL, metric='euclidean', reg=False, diag_fill=2):
    '''
    :param data_LR: data n_edges x n_subjects x n_tasks - LR scan pattern FCN data
    :param data_RL: data n_edges x n_subjects x n_tasks - RL scan pattern FCN data
    :param metric: metric to use for distance:
        euclidean - Frobenius distance between the matrices.
        affine_invariant - affine invariant SPD matrix distance between the matrices
    :return: dist_mat_LR: distance matrix between LR FCNs - n_tasks * n_subjects x  n_tasks * n_subjects
             dist_mat_RL: distance matrix between RL FCNs - n_tasks * n_subjects x  n_tasks * n_subjects
    '''
    # prepare the data, stack the values and return label indicators
    stacked_data_LR, stacked_data_RL, task_labels, subject_labels = prepare_data(data_LR, data_RL)
    n_measurements = stacked_data_LR.shape[0]

    # calculate the distances
    if metric == 'euclidean':
        dist_mat_LR = squareform(pdist(stacked_data_LR))
        dist_mat_RL = squareform(pdist(stacked_data_RL))
    elif metric == 'affine_invariant':
        dist_mat_LR = np.zeros((n_measurements, n_measurements))
        dist_mat_RL = np.zeros((n_measurements, n_measurements))
        print('Affine Invariant Distance Matrix Calculation Started. Row Progress Shown Below:')
        for i in tqdm(range(1, n_measurements)):
            dist_mat_LR[i, :i] = calc_affine_invariant_dist_svd(stacked_data_LR[i, :], stacked_data_LR[:i, :],
                                                           reg=reg, diag_fill=diag_fill)
            dist_mat_RL[i, :i] = calc_affine_invariant_dist_svd(stacked_data_RL[i, :], stacked_data_RL[:i, :],
                                                               reg=reg, diag_fill=diag_fill)
        # complete upper triangle
        dist_mat_LR = dist_mat_LR + dist_mat_LR.T
        dist_mat_RL = dist_mat_RL + dist_mat_RL.T
    else:
        raise ValueError('Invalid Metric - Returning Zero matrices')

    return dist_mat_LR, dist_mat_RL, task_labels, subject_labels


# ------------------- Kernel preparation functions -------------------

def shuffle_indices_while_preserving_labels(batch_idx, task_labels):
    # Ensure that batch_idx and task_labels have the same length
    assert len(batch_idx) == len(task_labels), "batch_idx and task_labels must have the same length"

    # Get unique labels
    unique_labels = np.unique(task_labels)

    # Initialize an empty list to store the shuffled indices
    shuffled_batch_idx = np.empty_like(batch_idx)

    for label in unique_labels:
        # Get the indices of all occurrences of the current label
        label_indices = np.where(task_labels == label)[0]

        # Shuffle these indices
        np.random.shuffle(label_indices)

        # Place these shuffled indices in the appropriate positions in shuffled_batch_idx
        shuffled_batch_idx[np.where(task_labels == label)[0]] = batch_idx[label_indices]

    return shuffled_batch_idx

def get_features_task_classification(data_LR, data_RL, task_labels, batch_idx, Nr, 
                                     N_val, embed_params, modalities='LR-LR'):
    """
    divide data to train, val and test for optimization methods
    """
    stacked_data_LR, _, _, _ = prepare_data(data_LR, data_RL)
    shuffled_idx = batch_idx
    # get task and subject labels for the current comparison
    task_labels_batch = task_labels[batch_idx]
    if embed_params['shuffle_subjects']:
        shuffled_idx = shuffle_indices_while_preserving_labels(batch_idx, task_labels_batch)
    if modalities == 'LR-LR':
        view1 = stacked_data_LR[batch_idx, :]
        view1_val = stacked_data_LR[batch_idx[:Nr + N_val], :]
        view2 = stacked_data_LR[shuffled_idx[:Nr], :]
        view2_val = stacked_data_LR[shuffled_idx[:Nr], :]
    else:
        raise ValueError(f'Unsupported modalities: {modalities}')
    return task_labels_batch, view1, view2, view1_val, view2_val


def get_kernels_task_classification(data_LR, task_labels, batch_idx, Nr, N_val, embed_params, method, 
                                    dist_mat_LR, dist_mat_RL=None, modalities='LR-LR', 
                                    plot_flag=False, k=None):
    '''
    :param task_labels: labels vector
    :param i_task: 1st task index - the one we use to identify
    :param j_task: 2nd task index - the one we need to identify
    :param embed_params: embedding parameters
    :param method: methods
    :param dist_mat_LR:
    :param dist_mat_RL:
    :param modalities: which 2 modalities are used for alternating diffusion
        LR-RL - LR and RL are the modalities
        tasks - 2 tasks are used as the modalities, the first modality is a common task (specified by common_task_idx)
         and the second modaility is a mixed task modality containing 2 different tasks (all possible permutations)
    :param common_task_idx: index of the common task used in the modality (only used when modalities='tasks')
    :return:
    '''
    shuffled_idx = batch_idx
    # get task and subject labels for the current comparison
    task_labels_batch = task_labels[batch_idx]
    if embed_params['shuffle_subjects']:
        shuffled_idx = shuffle_indices_while_preserving_labels(batch_idx, task_labels_batch)
    if modalities == 'LR-RL':
        if method in SINGLE_VIEW_METHODS or method in (FULL_VIEW_DIFFUSION_METHODS - {'lad'}):
            distance_mat1 = dist_mat_LR[batch_idx, :][:, batch_idx]
            distance_mat1_val = dist_mat_LR[batch_idx[:Nr + N_val], :][:, batch_idx[:Nr + N_val]]
            distance_mat2 = dist_mat_RL[shuffled_idx, :][:, shuffled_idx]
            distance_mat2_val = dist_mat_RL[shuffled_idx[:Nr + N_val], :][:, shuffled_idx[:Nr + N_val]]
        elif method in PARTIAL_VIEW_DIFFUSION_METHODS or method in (PARTIAL_VIEW_KERNEL_METHODS - {"kcca_impute"}):
            distance_mat1 = dist_mat_LR[batch_idx, :][:, batch_idx[:Nr]]
            distance_mat1_val = dist_mat_LR[batch_idx[:Nr + N_val], :][:, batch_idx[:Nr]]
            distance_mat2 = dist_mat_RL[shuffled_idx[:Nr], :][:, shuffled_idx[:Nr]]
            distance_mat2_val = dist_mat_RL[shuffled_idx[:Nr], :][:, shuffled_idx[:Nr]]
        elif method in {'kcca_impute'}:
            distance_mat1 = dist_mat_LR[batch_idx, :][:, batch_idx]
            distance_mat1_val = dist_mat_LR[batch_idx[:Nr + N_val], :][:, batch_idx[:Nr + N_val]]
            distance_mat2 = dist_mat_RL[shuffled_idx[:Nr], :][:, shuffled_idx[:Nr]]
            distance_mat2_val = dist_mat_RL[shuffled_idx[:Nr], :][:, shuffled_idx[:Nr]]
        elif method == 'lad':
            distance_mat1 = dist_mat_LR[batch_idx, :][:, batch_idx[:Nr]]
            distance_mat1_val = dist_mat_LR[batch_idx[:Nr + N_val], :][:, batch_idx[:Nr]]
            distance_mat2 = dist_mat_RL[shuffled_idx, :][:, shuffled_idx[:Nr]]
            distance_mat2_val = dist_mat_RL[shuffled_idx[:Nr + N_val], :][:, shuffled_idx[:Nr]]
        else:
            raise ValueError(f'Unsupported method: {method}')
    elif modalities == 'LR-LR':
        if method in SINGLE_VIEW_METHODS or method in (FULL_VIEW_DIFFUSION_METHODS - {'lad'}):
            distance_mat1 = dist_mat_LR[batch_idx, :][:, batch_idx]
            distance_mat1_val = dist_mat_LR[batch_idx[:Nr + N_val], :][:, batch_idx[:Nr + N_val]]
            distance_mat2 = dist_mat_LR[shuffled_idx, :][:, shuffled_idx]
            distance_mat2_val = dist_mat_LR[shuffled_idx[:Nr + N_val], :][:, shuffled_idx[:Nr + N_val]]
        elif method in PARTIAL_VIEW_DIFFUSION_METHODS or method in (PARTIAL_VIEW_KERNEL_METHODS - {"kcca_impute"}):
            distance_mat1 = dist_mat_LR[batch_idx, :][:, batch_idx[:Nr]]
            distance_mat1_val = dist_mat_LR[batch_idx[:Nr + N_val], :][:, batch_idx[:Nr]]
            distance_mat2 = dist_mat_LR[shuffled_idx[:Nr], :][:, shuffled_idx[:Nr]]
            distance_mat2_val = dist_mat_LR[shuffled_idx[:Nr], :][:, shuffled_idx[:Nr]]
        elif method in {'kcca_impute'}:
            distance_mat1 = dist_mat_LR[batch_idx, :][:, batch_idx]
            distance_mat1_val = dist_mat_LR[batch_idx[:Nr + N_val], :][:, batch_idx[:Nr + N_val]]
            distance_mat2 = dist_mat_LR[shuffled_idx[:Nr], :][:, shuffled_idx[:Nr]]
            distance_mat2_val = dist_mat_LR[shuffled_idx[:Nr], :][:, shuffled_idx[:Nr]]
        elif method == 'lad':
            distance_mat1 = dist_mat_LR[batch_idx, :][:, batch_idx[:Nr]]
            distance_mat1_val = dist_mat_LR[batch_idx[:Nr + N_val], :][:, batch_idx[:Nr]]
            distance_mat2 = dist_mat_LR[shuffled_idx, :][:, shuffled_idx[:Nr]]
            distance_mat2_val = dist_mat_LR[shuffled_idx[:Nr + N_val], :][:, shuffled_idx[:Nr]]
        else:
            raise ValueError(f'Unsupported method: {method}')
    elif modalities == 'labels':
        # the second view in this approach is the mean FCN between all subjects with the corresponding task
        # in the train data
        data_LR_stacked, _, _, _ = prepare_data(data_LR, data_LR)
        data_batch = data_LR_stacked[batch_idx[:Nr], :]
        n_labels = task_labels.max() + 1
        labels_train = task_labels_batch[:Nr]
        # take the mean for each label in the training set
        data_means = np.zeros((n_labels, data_LR_stacked.shape[1]))
        for task_i in range(n_labels):
            data_means[task_i, :] = data_batch[labels_train == task_i, :].mean()
        dist_means = squareform(pdist(data_means, metric='euclidean'))
        if method in SINGLE_VIEW_METHODS or method in (FULL_VIEW_DIFFUSION_METHODS - {'lad'}):
            distance_mat1 = dist_mat_LR[batch_idx, :][:, batch_idx]
            distance_mat2 = dist_means[task_labels_batch, :][:, task_labels_batch]
            # validation distance matrices
            distance_mat1_val = dist_mat_LR[batch_idx[:Nr + N_val], :][:, batch_idx[:Nr + N_val]]
            distance_mat2_val = dist_means[task_labels_batch[:Nr + N_val], :][:, task_labels_batch[:Nr + N_val]]
        elif method in PARTIAL_VIEW_DIFFUSION_METHODS or method in (PARTIAL_VIEW_KERNEL_METHODS - {"kcca_impute"}):
            distance_mat1 = dist_mat_LR[batch_idx, :][:, batch_idx[:Nr]]
            distance_mat2 = dist_means[task_labels_batch[:Nr], :][:, task_labels_batch[:Nr]]
            distance_mat1_val = dist_mat_LR[batch_idx[:Nr + N_val], :][:, batch_idx[:Nr]]
            distance_mat2_val = dist_means[task_labels_batch[:Nr], :][:, task_labels_batch[:Nr]]
        elif method in {'kcca_impute'}:
            distance_mat1 = dist_mat_LR[batch_idx, :][:, batch_idx]
            distance_mat1_val = dist_mat_LR[batch_idx[:Nr + N_val], :][:, batch_idx[:Nr + N_val]]
            distance_mat2 = dist_means[shuffled_idx[:Nr], :][:, shuffled_idx[:Nr]]
            distance_mat2_val = dist_means[shuffled_idx[:Nr], :][:, shuffled_idx[:Nr]]
        elif method == 'lad':
            distance_mat1 = dist_mat_LR[batch_idx, :][:, batch_idx[:Nr]]
            distance_mat2 = dist_means[task_labels_batch, :][:, task_labels_batch[:Nr]]
            distance_mat1_val = dist_mat_LR[batch_idx[:Nr + N_val], :][:, batch_idx[:Nr]]
            distance_mat2_val = dist_means[task_labels_batch[:Nr + N_val], :][:, task_labels_batch[:Nr]]
        else:
            raise ValueError(f'Unsupported method: {method}')
    else:
        raise ValueError(f'Unsupported modalities: {modalities}')
    if plot_flag:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(distance_mat1)
        ax.set_title('distance mat 1')
        fig, ax = plt.subplots(1, 1)
        ax.imshow(distance_mat2)
        ax.set_title('distance mat 2')
    K1, K2 = prep_kernels(dist_mat1=distance_mat1, dist_mat2=distance_mat2,
                          method=method, scale1=embed_params['param1'], scale2=embed_params['param2'],
                          zero_diag=embed_params['zero_diag'], k=k)
    K1_val, K2_val = prep_kernels(dist_mat1=distance_mat1_val, dist_mat2=distance_mat2_val,
                          method=method, scale1=embed_params['param1'], scale2=embed_params['param2'],
                          zero_diag=embed_params['zero_diag'], k=k)
    return task_labels_batch, K1, K2, K1_val, K2_val


def get_random_batches_indices(array, batch_size, seed=0):
    '''
    :param array: array to divide into batches
    :param batch_size: size of each batch
    :param seed: random seed
    :return: batches: indices of the batch division
    '''
    # Get the total number of entries
    num_entries = len(array)

    # Generate an array of indices and shuffle them
    indices = np.arange(num_entries)
    np.random.seed(seed)
    np.random.shuffle(indices)

    # Divide the indices into batches
    batches_idx = [indices[i:i + batch_size] for i in range(0, num_entries, batch_size)]

    return batches_idx

# ---------------- Classification and evaluation functions ----------------

# fit classification model and return predicted labels on test data
def fit_and_classify(train_data, test_data, train_labels, embed_params):
    # train model for task classification
    if embed_params['classifier'] == 'knn':
        k = 1
        model = KNeighborsClassifier(n_neighbors=k)
    elif embed_params['classifier'] == 'svm':
        model = SVC(kernel='linear')
    else:
        raise ValueError(f"Unsupported classifier: {embed_params['classifier']}")
    model.fit(train_data, train_labels)
    labels_pred = model.predict(test_data)
    return labels_pred


def calc_accuracy(embed, embed_val, Nr, N_val, task_labels_batch, embed_params):
    # split labels into train validation and test
    labels_train = task_labels_batch[:Nr]
    labels_val = task_labels_batch[Nr:Nr + N_val]
    labels_test = task_labels_batch[Nr + N_val:]
    # fit validation data
    labels_pred_val = fit_and_classify(train_data=embed_val[:Nr, :], test_data=embed_val[Nr:, :],
                                       train_labels=labels_train, embed_params=embed_params)
    acc_val = accuracy_score(labels_val, labels_pred_val)
    # fit test data
    labels_pred_test = fit_and_classify(train_data=embed[:Nr, :], test_data=embed[Nr + N_val:, :],
                                        train_labels=labels_train, embed_params=embed_params)
    acc_test = accuracy_score(labels_test, labels_pred_test)
    try:
        silhouette_pred = silhouette_score(embed[Nr + N_val:, :], labels_pred_test)
    except ValueError as e:
        # if all predicted labels are equal return -1, worst silhouette score
        print(f"Error calculating silhouette score: {e} returning -1")
        silhouette_pred = -1
    return acc_val, acc_test, silhouette_pred


def evaluate_sample(embed, embed_val, Nr, N_val, task_labels_batch, embed_params, 
                    method, dim, t, train_percent, sim_params, batch_index, seed,
                    run_time, results_dir):
    """
        Compute evaluation metrics for a single sample.
        plot visualization of the embedding if specified in sim_params.
    """
    acc_val, acc_test, silhouette_pred = calc_accuracy(embed=embed, embed_val=embed_val, Nr=Nr,
                                                       N_val=N_val, 
                                                       task_labels_batch=task_labels_batch, 
                                                       embed_params=embed_params)
    new_line = {
        'method': method,
        'param1': embed_params['param1'],
        'param2': embed_params['param2'],
        'dim': dim,
        't': t,
        'train_percent': train_percent,
        'batch_size': sim_params['batch_size'],
        'batch': batch_index,
        'seed': seed,
        'zero_diag': embed_params['zero_diag'],
        'classifier': embed_params['classifier'],
        'shuffle_subjects': embed_params['shuffle_subjects'],
        'valid_accuracy': acc_val,
        'test_accuracy': acc_test,
        'silhouette_gt': silhouette_score(embed, task_labels_batch),
        'silhouette_pred_test': silhouette_pred,
        'run_time': run_time
    }
    if sim_params['save_format'] == 'plot':
        ref_indicator = np.hstack((np.ones(Nr), np.zeros(N - Nr)))
        # plot_embed(embed[:, :2], title=f'batch_{i}_method_{method}_acc_{acc}',
        #            colors=task_labels_batch, ref_indicator=ref_indicator)
        title_str = f'{results_dir}/t_{t}_dim_{dim}'
        # plt.savefig(f'{title_str}.pdf', dpi=300, format='pdf')
        plot_embed_tsne(embed, title=f'batch_{batch_index}_method_{method}_acc_{acc_test}',
                        colors=task_labels_batch, ref_indicator=ref_indicator,
                        color_labels=sim_params['tasks_list'])
        plt.savefig(f'{title_str}_tsne_seed_{seed}.pdf', dpi=300, format='pdf')
    return new_line


def dist_mat_knn(distance_matrix_train_test, labels_train, k=5):
    num_test_samples = distance_matrix_train_test.shape[1]
    labels_pred = []

    for i in range(num_test_samples):
        # Get the distances from the current test point to all train points
        distances = distance_matrix_train_test[:, i]

        # Find the indices of the k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:k]

        # Get the labels of the k nearest neighbors
        k_nearest_labels = labels_train[k_nearest_indices]

        # Determine the most common label among the k nearest neighbors
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]

        # Append the predicted label to the list of predictions
        labels_pred.append(most_common_label)

    return np.array(labels_pred)


# ------------------------ Main Function that runs the experiment pipeline -------------------------

def task_classification(data_LR, data_RL=None, method='single', dist_mats=None, metric='euclidean',
                        embed_params=None, sim_params=None, train_percent=0.8):
    # get distance matrix
    if dist_mats is None:
        dist_mat_LR, dist_mat_RL, task_labels, subject_labels = calculate_distances(data_LR, data_RL, metric=metric)
        dist_mats = dict()
        dist_mats['LR'] = dist_mat_LR
        dist_mats['RL'] = dist_mat_RL
        dist_mats['task_labels'] = task_labels
        dist_mats['subject_labels'] = subject_labels
    else:
        dist_mat_LR = dist_mats['LR']
        dist_mat_RL = dist_mats['RL']
        task_labels = dist_mats['task_labels']
        subject_labels = dist_mats['subject_labels']

    # unify REST and REST2 labels
    if sim_params['unify_rest']:
        if 'REST2' in sim_params['tasks_list']:
            idx = sim_params['tasks_list'].index('REST2')
            task_labels[task_labels == idx] = 1  # label REST2 as REST
            task_labels[task_labels > idx] -= 1  # reset indices such that there are 8 classes
            sim_params['tasks_list'].remove('REST2')

    # cross all tasks and identify subjects
    n_subjects = data_LR.shape[1]
    n_tasks = data_LR.shape[2]
    # choose modalities for AD algorithm
    modalities = sim_params['modalities']
    common_task_idx = embed_params['common_task_idx']
    # divide data into batches
    results = []
    for seed in sim_params['seeds']:
        batches_idx = get_random_batches_indices(task_labels, sim_params['batch_size'], seed=seed)
        for i, batch_idx in enumerate(batches_idx):
            Nr = round(len(batch_idx) * train_percent)
            N_val = round(len(batch_idx) * sim_params['val_percent'])
            N = len(batch_idx)
            if method == 'single':
                # perform k-NN on the distance matrix between the FCNs
                task_labels_batch = task_labels[batch_idx]
                labels_train = task_labels_batch[:Nr]
                labels_test = task_labels_batch[Nr:]
                dist_mat = dist_mat_LR[batch_idx[:Nr], :][:, batch_idx[Nr:]]
                labels_pred = dist_mat_knn(dist_mat, labels_train)
                new_line = {
                    'method': method,
                    'param1': 0.0,
                    'param2': 0.0,
                    'dim': 0.0,
                    't': 0.0,
                    'train_percent': train_percent,
                    'batch_size': sim_params['batch_size'],
                    'batch': i,
                    'seed': seed,
                    'zero_diag': embed_params['zero_diag'],
                    'solver': embed_params['evd_solver'],
                    'classifier': embed_params['classifier'],
                    'shuffle_subjects': embed_params['shuffle_subjects'],
                    'accuracy': accuracy_score(labels_test, labels_pred)
                }
                results.append(new_line)
            elif method in OPT_METHODS:
                results_dir = (
                    f'{sim_params["figures_path"]}/batch_{i}_method_{method}_mu_{embed_params["param1"]}_'
                    f'train_percent_{train_percent}').replace('.', 'p')
                os.makedirs(results_dir, exist_ok=True)
                path = f"{results_dir}/data_seed_{seed}"
                if not sim_params['overwrite'] and os.path.isfile(f"{path}.pkl"):
                    embed_full, embed_full_val, task_labels_batch, loaded_results = load_data_from_pkl(path, method)
                    run_time = results[0]['run_time'] if len(loaded_results) > 0 else None
                else:
                    # optmization methods work with features and not kernels
                    (task_labels_batch, view1, view2, view1_val, 
                     view2_val) = get_features_task_classification(data_LR, data_RL, task_labels, 
                                                                   batch_idx, Nr, N_val, 
                                                                   embed_params, 
                                                                   modalities=modalities)
                    
                    # embed with desired method, embed to max dim to save computation time 
                    # and then calculate accuracy for different embedding dimensions by truncating the embedding
                    embed_dim = max(embed_params['embed_dims'])
                    start_time = time()
                    if method == 'fimvc_via':
                        embed_full = fimvc_via_embed(s1_full=view1, s2_ref=view2, embed_dim=embed_dim, 
                                                     mu=embed_params['param1'])
                        run_time = time() - start_time
                        embed_full_val = fimvc_via_embed(s1_full=view1_val, s2_ref=view2_val, 
                                                             embed_dim=embed_dim,
                                                             mu=embed_params['param1'])
                    else:
                        raise ValueError(f"Unsupported optimization method: {method}")
                    
                for dim in embed_params['embed_dims']:
                    embed = embed_full[:, :dim]
                    embed_val = embed_full_val[:, :dim]
                    new_line = evaluate_sample(embed, embed_val, Nr, N_val, task_labels_batch, 
                                                embed_params, method, dim, t=0.0, train_percent=train_percent,
                                                sim_params=sim_params, batch_index=i, seed=seed,
                                                results_dir=results_dir, run_time=run_time)
                    results.append(new_line)
                if sim_params['save_format'] == 'data':
                    embed_dict = dict()
                    embed_dict['embed_full'] = embed_full
                    embed_dict['embed_full_val'] = embed_full_val
                    embed_dict['df'] = results
                    embed_dict['ref_indicator'] = np.hstack((np.ones(Nr), np.zeros(N - Nr)))
                    embed_dict['labels_batch'] = task_labels_batch
                    embed_dict['tasks_list'] = sim_params['tasks_list']
                    with open(f"{results_dir}/data_seed_{seed}.pkl", 'wb') as fp:
                        pickle.dump(embed_dict, fp)
            
            # run kernel-based methods (ADM+, ADM, kCCA, NCCA, etc.)
            else:
                # create results directory
                t_list = [0] if method in PARTIAL_VIEW_KERNEL_METHODS.union(FULL_VIEW_KERNEL_METHODS) else embed_params['t_list']
                results_dir = (
                    f'{sim_params["figures_path"]}/batch_{i}_method_{method}_s1_{embed_params["param1"]}_'
                    f's2_{embed_params["param2"]}_train_percent_{train_percent}').replace('.', 'p')
                os.makedirs(results_dir, exist_ok=True)
                # load data if available
                path = f"{results_dir}/data_seed_{seed}"
                if not sim_params['overwrite'] and os.path.isfile(f"{path}.pkl"):
                    vecs, vals, vecs_val, vals_val, task_labels_batch, loaded_results = load_data_from_pkl(path, method)
                    run_time = results[0]['run_time'] if len(loaded_results) > 0 else None
                else:
                    # compute kernels
                    (task_labels_batch, K1, K2,
                     K1_val, K2_val) = get_kernels_task_classification(data_LR, task_labels, batch_idx, Nr, N_val, embed_params,
                                                                       method, dist_mat_LR, dist_mat_RL, modalities=modalities,
                                                                       plot_flag=sim_params['debug_plot'],
                                                                       k=embed_params['kernel_sparsity'])
                    # embed with desired method, embed to max dim to save computation time 
                    # and then calculate accuracy for different embedding dimensions by truncating the embedding
                    embed_dim = max(embed_params['embed_dims'])
                    start_time = time()
                    vals, vecs = embed_wrapper(method=method, embed_dim=embed_dim,
                                          K1=K1, K2=K2, solver=embed_params['evd_solver'],
                                          delete_kernels=embed_params['delete_kernels'], tol=embed_params['eig_tol'],
                                          stabilize=embed_params['stabilize'], return_vecs=True)
                    run_time = time() - start_time
                    # embed the validation set
                    vals_val, vecs_val = embed_wrapper(method=method, embed_dim=embed_dim,
                                               K1=K1_val, K2=K2_val, solver=embed_params['evd_solver'],
                                               delete_kernels=embed_params['delete_kernels'], tol=embed_params['eig_tol'],
                                               stabilize=embed_params['stabilize'], return_vecs=True)

                # calculate embedding for different embedding dimensions and t values efficiently
                for t in t_list:
                    for dim in embed_params['embed_dims']:
                        if method in PARTIAL_VIEW_KERNEL_METHODS:
                            embed = vecs[:, :dim]
                            embed_val = vecs_val[:, :dim]
                        else:
                            embed = np.real((vals[1:dim + 1]**t) * vecs[:, 1:dim + 1])
                            embed_val = np.real((vals_val[1:dim + 1] ** t) * vecs_val[:, 1:dim + 1])
                        new_line = evaluate_sample(embed, embed_val, Nr, N_val, task_labels_batch, 
                                                   embed_params, method, dim, t, train_percent, 
                                                   sim_params, batch_index=i, seed=seed, 
                                                   run_time=run_time, results_dir=results_dir)
                        results.append(new_line)
                if sim_params['save_format'] == 'data':
                    embed_dict = dict()
                    embed_dict['vecs'] = vecs
                    embed_dict['vals'] = vals
                    embed_dict['vecs_val'] = vecs_val
                    embed_dict['vals_val'] = vals_val
                    embed_dict['df'] = results
                    embed_dict['ref_indicator'] = np.hstack((np.ones(Nr), np.zeros(N - Nr)))
                    embed_dict['labels_batch'] = task_labels_batch
                    embed_dict['tasks_list'] = sim_params['tasks_list']
                    with open(f"{results_dir}/data_seed_{seed}.pkl", 'wb') as fp:
                        pickle.dump(embed_dict, fp)

    return pd.DataFrame(results)

