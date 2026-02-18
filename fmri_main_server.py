import pandas as pd
import argparse
import json
import pickle
import os
from tqdm.auto import tqdm
from time import time
from helper_functions.fmri_funcs import (load_data, calculate_distances, task_classification)


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="fMRI Simulation Runner")
    parser.add_argument("task_id", type=int, nargs='?', default=0, help="Task ID for parallel execution")
    parser.add_argument("n_tasks", type=int, nargs='?', default=1, help="Number of tasks for parallel execution")
    parser.add_argument("--run_opt_methods", action='store_true', help="Run the optimization methods as well (FIMVC-VIA, DVSAI)")

    return parser.parse_args()


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
    sim_name = 'run_with_optimization_methods'
    sim_params = {
        'sim_name': sim_name,
        'modalities': 'LR-LR',  # tasks, labels, LR-RL, LR-LR
        'debug_plot': False,
        'overwrite': False,
        'summarize_results': False,
        'save_format': 'data',  # 'plot', 'data'
        'batch_size': 4554,  # 506, 1518, 911, 2277
        'data_path': 'fmri_data',
        'seeds': [0, 3, 14, 35, 61, 78, 90, 102, 112, 123],  # use 10 different random seeds
        'figures_path': f'figures/fmri/task_classification/{sim_name}',
        'tasks_list': ['GAM', 'REST', 'REST2', 'LAN', 'MOT', 'REL', 'SOC', 'WM', 'EMO'],
        'methods': ['adm_plus', 'backward_only', 'nystrom', 'ncca', 'kcca_impute', 'ad', 'dm',
                    'apmc', 'forward_only'],
        'opt_methods': ['fimvc_via'],  # optimization methods with different parameters
        'unify_rest': True,
        'embed_dims': [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        't_list': [0.1, 0.2, 0.3, 0.5, 1, 2],
        'kernel_scales1': [0.1, 0.5, 1, 2, 10],
        'kernel_scales2': [0.5],
        'same_scales': True,
        'mu_list': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # regularization parameters for FIMVC-VIA
        'train_percents': [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7],
        'val_percent': 0.1  # validation set percentage
    }
    return sim_params


def run_embed_method(LR_smooth, RL_smooth, dist_mats_euclidean, kernel_scale1, kernel_scale2,
                      method, train_percent, sim_params):
    # set same scale if required
    if sim_params['same_scales']:
        scale2 = kernel_scale1
    else:
        scale2 = kernel_scale2
    embed_params = {
        'embed_dims': sim_params['embed_dims'],
        'common_task_idx': 1,
        'kernel_sparsity': None,
        'delete_kernels': True,
        't_list': sim_params['t_list'],
        'param1': kernel_scale1,
        'param2': scale2,
        'stabilize': False,
        'shuffle_subjects': True,
        'eig_tol': 0,
        'classifier': 'knn',
        'zero_diag': False,
        'evd_solver': 'arpack',  # 'arpack' / 'randomized'
        'subtract_min': False,
    }
    # task classification
    results_df = task_classification(LR_smooth, RL_smooth, method=method, dist_mats=dist_mats_euclidean,
                                        metric='euclidean', embed_params=embed_params,
                                        sim_params=sim_params, train_percent=train_percent)
    return results_df

def run_opt_method(LR_smooth, RL_smooth, dist_mats_euclidean, mu, 
                   method, train_percent, sim_params):
    embed_params = {
        'embed_dims': sim_params['embed_dims'],
        'common_task_idx': 1,
        'kernel_sparsity': None,
        'delete_kernels': True,
        'param1': mu,
        'param2': None,
        'shuffle_subjects': True,
        'classifier': 'knn',
        'zero_diag': False,
    }
    # task classification
    results_df = task_classification(LR_smooth, RL_smooth, method=method, 
                                     dist_mats=dist_mats_euclidean,
                                     metric='euclidean', embed_params=embed_params,
                                     sim_params=sim_params, train_percent=train_percent)
    return results_df


def run_simulation(args, sim_params):
    start_time = time()
    data_path = sim_params['data_path']
    figures_path = sim_params['figures_path']
    os.makedirs(figures_path, exist_ok=True)

    # load smooth matrices data
    LR_smooth, RL_smooth = load_data(data_path, data_type='smooth')
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

    # run optimization methods if required
    train_percents = sim_params['train_percents']
    job_num = -1
    result_dfs = []
    
    if args.run_opt_methods:
        pbar = tqdm(total=len(sim_params['opt_methods']) *
                     len(train_percents) * 
                     len(sim_params['mu_list']), 
                     desc="Running Optimization Methods")
        for opt_method in sim_params['opt_methods']:
            for train_percent in train_percents:
                for mu in sim_params['mu_list']:
                    job_num += 1
                    if not job_num % args.n_tasks == args.task_id:
                        pbar.update(1)
                        continue
                    results_df = run_opt_method(LR_smooth, RL_smooth, dist_mats_euclidean,
                                                mu=mu, method=opt_method, train_percent=train_percent, 
                                                sim_params=sim_params)
                    result_dfs.append(results_df)
                    pbar.update(1)
    kernel_scales1 = sim_params['kernel_scales1']
    if sim_params['same_scales']:
        kernel_scales2 = [1] 
        param_len = len(sim_params['kernel_scales1'])
    else:
        param_len = len(sim_params['kernel_scales1']) * len(sim_params['kernel_scales2'])
        kernel_scales2 = sim_params['kernel_scales2']
    pbar = tqdm(total=len(sim_params['methods']) * 
                len(train_percents) * 
                param_len , desc="Running Embedding Methods")
    for method in sim_params['methods']:
        for train_percent in train_percents:
            for kernel_scale2 in kernel_scales2:
                for kernel_scale1 in kernel_scales1:
                    # if job isn't assigned to task_id continue
                    job_num += 1
                    if not job_num % args.n_tasks == args.task_id:
                        pbar.update(1)
                        continue
                    results_df = run_embed_method(LR_smooth, RL_smooth, dist_mats_euclidean, 
                                                   kernel_scale1, kernel_scale2, method, 
                                                   train_percent, sim_params)
                    result_dfs.append(results_df)
                    pbar.update(1)
        
    results_all_df = pd.concat(result_dfs)
    os.makedirs(f'{figures_path}/machine_output', exist_ok=True)
    my_save_csv(results_all_df, f'{figures_path}/machine_output/machine_{args.task_id}', sim_params)
    print(f"Total Simulation Time for task_id {args.task_id}: {time() - start_time:.2f} seconds")
    return results_all_df


def merge_csvs(sim_params):
    # Define the path to the directory containing the CSV files
    directory_path = f"{sim_params['figures_path']}/machine_output/"
    figures_path = sim_params['figures_path']

    # Initialize an empty list to store DataFrames
    dfs = []
    # Iterate over all CSV files in the directory
    for filename in os.listdir(directory_path):
        if filename.startswith("machine"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            dfs.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    max_accuracy_df = combined_df.loc[
        combined_df.groupby(['method', 'train_percent', 'dim', 'seed'])['valid_accuracy'].idxmax()]
    # Save the combined DataFrame to a new CSV file
    combined_output_path = f"{figures_path}/results_merged.csv"
    combined_df.to_csv(combined_output_path, index=False)
    max_accuracy_df.to_csv(f"{figures_path}/best_merged.csv", index=False)

    print(f"All CSV files have been merged into {combined_output_path}")


if __name__ == '__main__':
    args = setup_arg_parser()
    sim_params = load_simulation_params()
    if sim_params['summarize_results']:
        if args.task_id == 0:
            merge_csvs(sim_params)
    else:
        run_simulation(args, sim_params)
    if args.task_id == 0:
        with open(f"{sim_params['figures_path']}/sim_params.json", 'w') as fp:
            json.dump(sim_params, fp, indent=4)





