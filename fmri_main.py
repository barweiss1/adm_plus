import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
import pandas as pd
import json
import pickle
import os
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.io import loadmat
from helper_functions.fmri_funcs import (cross_task_fingerprint, load_data,
                                         calculate_distances, task_classification)


def my_save_csv(df, base_filename, sim_params):
    num = 1
    filename = f"{base_filename}.csv"

    # Check if file exists and increment number
    while os.path.exists(filename) and not sim_params['overwrite']:
        filename = f"{base_filename}({num}).csv"
        num += 1

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)


def load_simulation_params():
    sim_name = 'ncca_test'
    sim_params = {
        'sim_name': sim_name,
        'modalities': 'LR-LR',  # tasks, labels, LR-RL, LR-LR
        'debug_plot': False,
        'overwrite': True,
        'save_format': 'plot',  # 'plot', 'data'
        'batch_size': 4554,  # 506, 1518, 911, 2277
        'data_path': 'fmri_data',
        'seeds': [0, 3, 14, 35, 61, 78, 90, 102, 112, 123],  # use 10 different random seeds
        'figures_path': f'figures/fmri/task_classification/{sim_name}',
        'tasks_list': ['GAM', 'REST', 'REST2', 'LAN', 'MOT', 'REL', 'SOC', 'WM', 'EMO'],
        'unify_rest': True,
        'val_percent': 0.1  # validation set percentage
    }
    return sim_params


if __name__ == '__main__':
    sim_params = load_simulation_params()
    data_path = sim_params['data_path']
    figures_path = sim_params['figures_path']
    os.makedirs(figures_path, exist_ok=True)
    # load smooth matrices data
    LR_smooth, RL_smooth = load_data(data_path, data_type='smooth')
    # graph parameters
    n_edges = LR_smooth.shape[0]
    n_subjects = LR_smooth.shape[1]
    n_tasks = LR_smooth.shape[2]

    # calculate distance matrix
    if os.path.isfile(f"{data_path}/dist_mats_euclidean.pkl"):
        with open(f"{data_path}/dist_mats_euclidean.pkl", 'rb') as fp:
            dist_mats_euclidean = pickle.load(fp)
    else:
        dist_mat_LR, dist_mat_RL, task_labels, subject_labels = calculate_distances(LR_smooth, RL_smooth,
                                                                                    metric='euclidean')
        dist_mats_euclidean = dict()
        dist_mats_euclidean['LR'] = dist_mat_LR
        dist_mats_euclidean['RL'] = dist_mat_RL
        dist_mats_euclidean['task_labels'] = task_labels
        dist_mats_euclidean['subject_labels'] = subject_labels
        # save distance matrix
        with open(f"{data_path}/dist_mats_euclidean.pkl", 'wb') as fp:
            pickle.dump(dist_mats_euclidean, fp)
    # # run fingerprint analysis
    # acc_mat_LR, _ = cross_task_fingerprint(LR_smooth, method='single', dist_mats=dist_mats_euclidean)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(acc_mat_LR * 100, annot=True, fmt=".1f", cmap="coolwarm", xticklabels=tasks_list,
    #             yticklabels=tasks_list, vmin=0, vmax=100)
    # plt.savefig(f'{figures_path}/smooth_acc_mat_LR.pdf', format='pdf', dpi=300)
    # plt.show()
    # run fingerprint analysis - alternating diffusion
    # kernel_scales = [0.01, 0.02, 0.03, 0.2, 0.3, 0.4, 1, 2]
    # kernel_scales1 = [0.2, 0.3, 0.4, 0.5, 0.6]
    kernel_scales1 = [0.5]
    kernel_scales2 = [0.5]
    # kernel_scales2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # kernel_scales2 = [0.0005, 0.001, 0.003, 0.005, 0.01]
    # kernel_scales2 = [0.001, 0.002, 0.003, 0.01]
    # methods = ['single']
    # methods = ['nystrom', 'ncca', 'ad', 'dm', 'ffbb', 'fbfb', 'forward_only']
    methods = ['ncca']
    # train_percents = [0.1, 0.3, 0.5, 0.7]
    train_percents = [0.5]
    result_dfs = []
    for method in methods:
        method_result_dfs = []
        for train_percent in train_percents:
            for kernel_scale2 in tqdm(kernel_scales2):
                for kernel_scale1 in kernel_scales1:
                    embed_params = {
                        'embed_dims': [5, 10, 15, 20, 30, 40, 50, 60],
                        'common_task_idx': 1,
                        'kernel_sparsity': None,
                        'delete_kernels': True,
                        't_list': [0.1, 0.2, 0.3, 0.5, 1, 2],
                        'kernel_scale1': kernel_scale1,
                        'kernel_scale2': kernel_scale1,
                        'stabilize': False,
                        'shuffle_subjects': True,
                        'eig_tol': 0,
                        'classifier': 'knn',
                        'zero_diag': False,
                        'evd_solver': 'arpack',  # 'arpack' / 'randomized'
                        'subtract_min': False,
                    }
                    # fingerprinting
                    # acc_mat_ad, _ = cross_task_fingerprint(LR_smooth, RL_smooth, method=method,
                    #                                        dist_mats=dist_mats_euclidean, embed_params=embed_params)
                    # scale_str = str(kernel_scale).replace('.', 'p')
                    # plt.figure(figsize=(10, 8))
                    # tasks_idx = np.arange(len(tasks_list))
                    # tasks_idx = tasks_idx[tasks_idx != embed_params['common_task_idx']]
                    # tasks_list_wo_curr = np.array(tasks_list)[tasks_idx]
                    # sns.heatmap(acc_mat_ad * 100, annot=True, fmt=".1f", cmap="coolwarm",
                    #             xticklabels=tasks_list_wo_curr, yticklabels=tasks_list_wo_curr, vmin=0, vmax=100)
                    # plt.savefig(f"{figures_path}/smooth_acc_mat_{method}_scale_{scale_str}"
                    #             f"_dim_{embed_params['embed_dim']}_nzd.pdf", format='pdf', dpi=300)
                    # plt.show()
                    # task classification
                    results_df = task_classification(LR_smooth, RL_smooth, method=method, dist_mats=dist_mats_euclidean,
                                                     metric='euclidean', embed_params=embed_params,
                                                     sim_params=sim_params, train_percent=train_percent)
                    result_dfs.append(results_df)
                    method_result_dfs.append(results_df)
        method_df = pd.concat(method_result_dfs)
        method_df.to_csv(f'{figures_path}/{method}.csv')
    results_all_df = pd.concat(result_dfs)
    max_accuracy_df = results_all_df.loc[results_all_df.groupby(['method', 'train_percent', 'dim',
                                                                 'seed'])['accuracy'].idxmax()]
    my_save_csv(results_all_df, f'{figures_path}/results_all_{sim_params["modalities"]}_'
                                f'{sim_params["sim_name"]}_1nn', sim_params)
    my_save_csv(max_accuracy_df, f'{figures_path}/best_all_{sim_params["modalities"]}_'
                                 f'{sim_params["sim_name"]}_1nn', sim_params)


