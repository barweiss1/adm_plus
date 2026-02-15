### Import relevant libraries
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import cv2
import tqdm
import os
from scipy import ndimage
from copy import deepcopy
from helper_functions.logo_funcs import (imresize_pad, lin2circ_angles, get_validation_indices,
                                         image_align_height, process_iteration)
import json
import pickle
import pandas as pd
import argparse
from time import time


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="fMRI Simulation Runner")
    parser.add_argument("task_id", type=int, nargs='?', default=0, help="Task ID for parallel execution")
    parser.add_argument("n_tasks", type=int, nargs='?', default=1, help="Number of tasks for parallel execution")

    return parser.parse_args()


def load_simulation_params():
    font_name = "DejaVu Sans"
    font_properties_title = font_manager.FontProperties(family=font_name, size=28)
    font_properties_ticks = font_manager.FontProperties(family=font_name, size=22)
    figsize = (8, 7)
    sim_params = {
        'delete_kernels': False,
        'generate_data': True,
        'data_path': 'logo_data/new_logos',
        'figures_path': 'figures/new_logo/dist_discrepency',
        'evd_solver': 'arpack',  # 'arpack' / 'randomized' / 'svd'
        'ad_methods': ['lead', 'forward_only', 'ncca', 'nystrom', 'adm_plus', 'backward_only'],
        'seeds': [0, 3, 14, 35, 61, 78, 90, 102, 112, 123],  # use 10 different random seeds
        'embed_dim': 2,
        't': 0,
        'scales': [2, 8, 10, 20],
        'angle_bias_factors': [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2],
        'angles_for_bias': 'common',  # specific1 / specific2 / common
        'im_resize_factor': 1.5,
        'Nr': 100,  # number of samples in the reference set,
        'N': 1000,  # number of total samples
        'valid_size': 0.2,
        'summarize_results': False,
        'figsize': figsize,
        'font_properties_title': font_properties_title,
        'font_properties_ticks': font_properties_ticks,
    }

    return sim_params


def merge_csvs(sim_params):
    # Define the path to the directory containing the CSV files
    N = sim_params['N']
    Nr = sim_params['Nr']
    ds_factor = sim_params['im_resize_factor']
    directory_path = (f'{sim_params["figures_path"]}/N_{N}_ds_factor_{ds_factor}_Nr_{Nr}_bias'
                      f'_{sim_params["angles_for_bias"]}/machine_output').replace('.', 'p')
    results_path = (f'{sim_params["figures_path"]}/N_{N}_ds_factor_{ds_factor}_Nr_{Nr}_bias'
                    f'_{sim_params["angles_for_bias"]}').replace('.', 'p')

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
    best_error_df = combined_df.loc[
        combined_df.groupby(['Method', 'bias_factor', 'seed'])['MAE_valid'].idxmin()]
    # Save the combined DataFrame to a new CSV file
    combined_output_path = f"{results_path}/results_merged.csv"
    combined_df.to_csv(combined_output_path, index=False)
    best_error_df.to_csv(f"{results_path}/best_merged.csv", index=False)

    print(f"All CSV files have been merged into {combined_output_path}")


def generate_data(data_path, sim_params):
    # load images
    image_path = sim_params['data_path']
    img = cv2.imread(f'{image_path}/elephant.jpg')
    img_view1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imread(f'{image_path}/snowman.jpg')
    img_common = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imread(f'{image_path}/sloth.jpg')
    img_view2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # align image heights to allow concatination
    height = 400  # height needs to be uniform for concatanation
    img_view1_rs = image_align_height(img_view1, height=height)
    img_common_rs = image_align_height(img_common, height=height)
    img_view2_rs = image_align_height(img_view2, height=height)

    # define rotation angular velocities
    w_view1 = 2.93 * 4  # [degrees/timestamp]
    w_view2 = 1.27 * 4  # [degrees/timestamp]
    w_common = 2.11 * 4  # [degrees/timestamp]

    # calculate angles
    N = sim_params['N']
    angles_view1_d = lin2circ_angles(w_view1 * np.linspace(0, N - 1, N))
    angles_common_d = lin2circ_angles(w_common * np.linspace(0, N - 1, N))  # generate angle vectors for dataset
    angles_view2_d = lin2circ_angles(w_view2 * np.linspace(0, N - 1, N))  # generate angle vectors for dataset

    # resize the common image to change common to specific ratio
    common_resized = imresize_pad(img_common_rs, sim_params['im_resize_factor'])

    # Create Dataset - Deterministic Rotation
    if not os.path.isfile(f"{data_path}/s1.npy"):
        sensor_1 = np.concatenate((img_view1_rs, img_common_rs), axis=1)
        sensor_2 = np.concatenate((img_common_rs, img_view2_rs), axis=1)
        d1 = sensor_1.size  # number of pixels/dimension of data points
        d2 = sensor_2.size  # number of pixels/dimension of data points
        s1_points_d = np.zeros((N, d1), dtype='uint8')  # initalize data points
        s2_points_d = np.zeros((N, d2), dtype='uint8')  # initalize data points

        for i in tqdm.tqdm(range(N)):
            # rotate images
            view1_rot = ndimage.rotate(img_view1_rs, angles_view1_d[i], reshape=False, mode='nearest')
            common_rot = ndimage.rotate(common_resized, angles_common_d[i], reshape=False, mode='nearest')
            view2_rot = ndimage.rotate(img_view2_rs, angles_view2_d[i], reshape=False, mode='nearest')
            # concatenate images
            sensor_1 = np.concatenate((view1_rot, common_rot), axis=1)
            sensor_2 = np.concatenate((common_rot, view2_rot), axis=1)
            # flatten images to create a vector
            s1_points_d[i, :] = sensor_1.reshape((1, d1))
            s2_points_d[i, :] = sensor_2.reshape((1, d2))

        print('Successfully Generated Dataset')

        data_dict = dict()
        data_dict['s1'] = s1_points_d
        data_dict['s2'] = s2_points_d
        data_dict['specific1_angles'] = angles_view1_d
        data_dict['common_angles'] = angles_common_d
        data_dict['specific2_angles'] = angles_view2_d
        os.makedirs(data_path, exist_ok=True)
        with open(f"{data_path}/data_dict.pkl", 'wb') as fp:
            pickle.dump(data_dict, fp)
            print('data dictionary saved successfully to file')
    else:
        print('Data Already Generated')


def run_simulation(task_id, n_tasks, sim_params):
    start_time = time()
    N = sim_params['N']
    Nr = sim_params['Nr']
    ds_factor = sim_params['im_resize_factor']
    data_path = f'{sim_params["data_path"]}/N_{N}_ds_factor_{ds_factor}'.replace('.', 'p')
    figures_path = (f'{sim_params["figures_path"]}/N_{N}_ds_factor_{ds_factor}_Nr_{Nr}_bias'
                    f'_{sim_params["angles_for_bias"]}').replace('.', 'p')

    # generate data if requested
    if sim_params['generate_data']:
        if task_id == 0:
            generate_data(data_path, sim_params)
        else:
            print(f"This is machine {task_id}, only machine 0 generates data, skipping...")
        return

    os.makedirs(figures_path, exist_ok=True)
    # load data
    with open(f"{data_path}/data_dict.pkl", 'rb') as fp:
        data_dict = pickle.load(fp)
        print('Data dictionary loaded successfully')

    results = []
    bias_factors = sim_params['angle_bias_factors']
    task_counter = 0  # task counter to assign jobs to the different machines
    for bias_factor in bias_factors:
        for seed in sim_params['seeds']:
            if task_counter % n_tasks == task_id:
                validation_idx = get_validation_indices(sim_params, seed=seed)
                new_results = process_iteration(data_dict, sim_params, validation_idx, figures_path,
                                                bias_factor=bias_factor, seed=seed)
                results.extend(new_results)
            task_counter += 1
    # save results to csv
    results_df = pd.DataFrame(results)
    os.makedirs(f'{figures_path}/machine_output', exist_ok=True)
    results_df.to_csv(f'{figures_path}/machine_output/machine_{task_id}.csv', index=False)
    print(f"Total Simulation Time for task_id {task_id}: {time() - start_time:.2f} seconds")
    # save sim_params
    if task_id == 0:
        sim_params_json = deepcopy(sim_params)
        sim_params_json['font_properties_title'] = 'Incompatible with JSON'
        sim_params_json['font_properties_ticks'] = 'Incompatible with JSON'
        with open(f"{figures_path}/sim_params.json", 'w') as fp:
            json.dump(sim_params_json, fp, indent=4)


if __name__ == '__main__':
    # run simulation
    args = setup_arg_parser()
    sim_params = load_simulation_params()
    if sim_params['summarize_results']:
        if args.task_id == 0:
            merge_csvs(sim_params)
    else:
        run_simulation(args.task_id, args.n_tasks, sim_params)




