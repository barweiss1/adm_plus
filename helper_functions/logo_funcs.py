import os

import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib as mpl
import numpy as np
import cv2
import scipy as sci
import tqdm
from scipy import ndimage
from helper_functions.embed_utils import Create_Transition_Mat, Create_Asym_Tran_Kernel
from helper_functions.embed_methods import embed_wrapper
from helper_functions.embed_methods import (OPT_METHODS, FULL_VIEW_DIFFUSION_METHODS, PARTIAL_VIEW_DIFFUSION_METHODS,
                                            FULL_VIEW_KERNEL_METHODS, PARTIAL_VIEW_KERNEL_METHODS, SINGLE_VIEW_METHODS)
import helper_functions.plotting_funcs as plot_funcs


# adds noise to an image in uint8 representation
def imnoise_gauss(im, sigma):
    # convert to double
    im_double = im/255
    im_d_noisy = im_double + np.random.normal(loc=0, scale=sigma, size=im.shape)
    return (im_d_noisy*255).astype(np.uint8)


def image_align_height(im, height=400):
    # pad image
    h, w = im.shape[:2]
    pad_h = int(h * 0.2) // 2  # 10% padding on each side
    pad_w = int(w * 0.2) // 2
    if w < 0.7 * h:
        pad_w = int(h * 0.25)
    if h < 0.7 * w:
        pad_h = int(w * 0.25)
    im_padded = cv2.copyMakeBorder(im, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # resize
    scale = im_padded.shape[0] / height
    width = int(im_padded.shape[1] / scale)
    im_rs = cv2.resize(im_padded, (width, height), interpolation=cv2.INTER_AREA)
    return im_rs


def imresize_pad(im, factor):
    if factor == 1:
        return im.astype(np.uint8)
    # resize image to be small
    shape = im.shape
    width = int(shape[0] / factor)
    height = int(shape[1] / factor)
    new_shape = (width, height)
    im_small = cv2.resize(im, new_shape, interpolation=cv2.INTER_AREA)

    # pad with zeros to just make object smaller
    x_start = int((shape[0] - new_shape[0]) / 2)
    y_start = int((shape[1] - new_shape[1]) / 2)
    im_pad = np.ones(shape) * 255
    im_pad[y_start:y_start + new_shape[1], x_start:x_start + new_shape[0]] = im_small
    return im_pad.astype(np.uint8)


# function that converts angle vector to [0,360] representation
def lin2circ_angles(angles, rep_type='0-360'):
    angles_circ = np.copy(angles)  # copy angles vectors
    # iterate until all values are under 360
    while (np.sum(angles_circ >= 360) != 0):
        angles_circ[angles_circ >= 360] = angles_circ[angles_circ >= 360] - 360
    # iterate until all values are over 0
    while (np.sum(angles_circ < 0) != 0):
        angles_circ[angles_circ < 0] = angles_circ[angles_circ < 0] + 360

    if rep_type == 'm180-180':
        angles_circ[angles_circ > 180] = angles_circ[angles_circ > 180] - 360

    return angles_circ


def circ2lin_angles(angles):
    lin_angles = np.copy(angles)
    # go over angles vector, when it drops by 360 degrees cancel add 360 to make it continous
    for i in range(len(angles) - 1):
        diff = lin_angles[i + 1] - lin_angles[i]
        if diff < -300:  # if there was a sudden drop
            lin_angles[i + 1:] = lin_angles[i + 1:] + 360

    return lin_angles


def sample_reference_set(data_dict, sim_params, bias_factor=0, seed=0):
    s1 = data_dict['s1']
    s2 = data_dict['s2']
    angles = lin2circ_angles(data_dict[f'{sim_params["angles_for_bias"]}_angles'])
    N = sim_params['N']
    Nr = sim_params['Nr']
    # define probability weights
    weights = (angles / np.max(angles)) ** bias_factor

    # normalize weights to get probabilities
    normalized_weights = weights / np.sum(weights)

    # randomly sample Nr angles with probability weight
    np.random.seed(seed)
    ref_idx = np.random.choice(len(angles), size=Nr, replace=False, p=normalized_weights)

    # select views
    total_idx = np.round(np.linspace(0, N - 1, N)).astype(int)
    single_idx = [i for i in total_idx if i not in ref_idx]  # the rest of the indices
    reorder_idx = np.concatenate((ref_idx, single_idx))
    reorder_idx = np.argsort(reorder_idx)

    # split reference set
    s1_ref = s1[ref_idx, :]
    s2_ref = s2[ref_idx, :]

    # create single sensor set - only with samples from sensor 1
    s1_single = s1[single_idx, :]
    s1_aligned = np.concatenate((s1_ref, s1_single), axis=0)
    s2_single = s2[single_idx, :]  # save sensor 2 images for reference to completed images
    s2_aligned = np.concatenate((s2_ref, s2_single), axis=0)

    return s1_ref, s2_ref, s1_aligned, s2_aligned, reorder_idx, ref_idx


def evaluate_embed(embed, sim_params, validation_idx, angles_common, method, scale, bias_factor, seed):
    Nr = sim_params['Nr']
    error_mae, error_std = embed_error(embed, angles_common, plot_flag=False, metric='MAE')
    error_mae_val, _ = embed_error(embed[validation_idx, :], angles_common[validation_idx], plot_flag=False,
                                   metric='MAE')
    error_mse, _ = embed_error(embed, angles_common, plot_flag=False, metric='RMSE')
    error_mae_center, error_std_center = embed_error(embed, angles_common, plot_flag=False, metric='MAE',
                                                     center_data=True)
    error_mse_center, _ = embed_error(embed, angles_common, plot_flag=False, metric='RMSE', center_data=True)
    new_line = {'Method': method,
                'scale': scale,
                'bias_factor': bias_factor,
                'seed': seed,
                'RMSE': error_mse,
                'MAE': error_mae,
                'MAE_valid': error_mae_val,
                'STD': error_std,
                'RMSE w centered data': error_mse_center,
                'MAE w centered data': error_mae_center,
                'STD center': error_std_center,
                'ref mean radius': np.mean(np.sqrt(embed[:Nr, 0] ** 2 + embed[:Nr, 1] ** 2)),
                'out of ref mean radius': np.mean(np.sqrt(embed[Nr:, 0] ** 2 + embed[Nr:, 1] ** 2))
                }
    return new_line


def process_iteration(data_dict, sim_params, validation_idx, figures_path, bias_factor=0, seed=0):
    (s1_ref, s2_ref, s1_aligned, s2_aligned,
     reorder_idx, ref_idx) = sample_reference_set(data_dict, sim_params, bias_factor, seed)
    iteration_path = f'{figures_path}/bias_factor_{bias_factor}_seed_{seed}'.replace('.', 'p')
    os.makedirs(iteration_path, exist_ok=True)
    # load paramerters for plot
    Nr = sim_params['Nr']
    figsize = sim_params['figsize']
    font_properties_ticks = sim_params['font_properties_ticks']
    angles_common = lin2circ_angles(data_dict[f'common_angles'])
    results = []
    for scale in tqdm.tqdm(sim_params['scales']):
        A1, _, _ = Create_Asym_Tran_Kernel(s1_aligned, s1_ref, mode='median', scale=scale)
        A2, _, _ = Create_Asym_Tran_Kernel(s2_aligned, s2_ref, mode='median', scale=scale)
        if {'kcca_impute', 'ad', 'ad_svd'}.intersection(sim_params['ad_methods']):
            K1, _ = Create_Transition_Mat(s1_aligned, scale=scale)
            K2, _ = Create_Transition_Mat(s2_aligned, scale=scale)
        K1_ref, _ = Create_Transition_Mat(s1_ref, scale=scale)
        K2_ref, _ = Create_Transition_Mat(s2_ref, scale=scale)

        for method in sim_params['ad_methods']:
            method_key = f'{method}_scale_{scale}'
            if method in PARTIAL_VIEW_DIFFUSION_METHODS or method in (PARTIAL_VIEW_KERNEL_METHODS - {"kcca_impute"}):
                embed = embed_wrapper(s1_ref, s1_aligned, s2_ref, s2_aligned, method=method,
                                      embed_dim=sim_params['embed_dim'], t=sim_params['t'],
                                      K1=A1, K2=K2_ref, solver=sim_params['evd_solver'],
                                      delete_kernels=sim_params['delete_kernels'])
                embed = embed[reorder_idx, :]
            elif method in SINGLE_VIEW_METHODS or method in (FULL_VIEW_DIFFUSION_METHODS - {'lad'}):
                embed = embed_wrapper(s1_ref, s1_aligned, s2_ref, s2_aligned, method=method,
                                      embed_dim=sim_params['embed_dim'], t=sim_params['t'],
                                      K1=K1, K2=K2, solver=sim_params['evd_solver'],
                                      delete_kernels=sim_params['delete_kernels'])
                embed = embed[reorder_idx, :]
            elif method in {"kcca_impute"}:
                embed = embed_wrapper(s1_ref, s1_aligned, s2_ref, s2_aligned, method=method,
                                      embed_dim=sim_params['embed_dim'], t=sim_params['t'],
                                      K1=K1, K2=K2_ref, solver=sim_params['evd_solver'],
                                      delete_kernels=sim_params['delete_kernels'])
                embed = embed[reorder_idx, :]
            elif method == "lad":
                embed = embed_wrapper(s1_ref, s1_aligned, s2_ref, s2_aligned, method=method,
                                      embed_dim=sim_params['embed_dim'], t=sim_params['t'],
                                      K1=A1, K2=A2, solver=sim_params['evd_solver'],
                                      delete_kernels=sim_params['delete_kernels'])
                embed = embed[reorder_idx, :]

            plot_method_embedding(embed, iteration_path, angles_common, Nr, method_key, ref_idx,
                                  plot_flag=False, pointsize=20, pointsize_ref=30,
                                  fontproperties=font_properties_ticks, figsize=figsize)
            new_line = evaluate_embed(embed, sim_params, validation_idx, angles_common, method, scale,
                                      bias_factor, seed)
            results.append(new_line)
    return results


def embed_error(embed, real_angles, plot_flag=False, metric='MAE', center_data=False):
    # calculate angles from embedding
    embed_r = np.real(embed)  # make sure embed is real
    if center_data:
        embed_r = embed_r - np.mean(embed_r, axis=0)
    embed_angles = np.arctan2(embed_r[:, 1], embed_r[:, 0])
    embed_angles = embed_angles * 180 / np.pi  # convert from radians to degrees
    embed_angles = lin2circ_angles(embed_angles, rep_type='m180-180')
    neg_embed_angles = -embed_angles  # if angles change counter-clockwise need to use the negative of the angles
    real_angles_360 = lin2circ_angles(real_angles, rep_type='m180-180')
    error_sig = 180 - abs(abs(embed_angles - real_angles_360) - 180)
    error_sig = error_sig - np.mean(error_sig)
    neg_error_sig = 180 - abs(abs(neg_embed_angles - real_angles_360) - 180)
    neg_error_sig = neg_error_sig - np.mean(neg_error_sig)
    # remove angle offset since the error metric needs to be invariant to rotations
    # choose lowest error at each entry
    best_error_sig = np.where(np.abs(error_sig) < np.abs(neg_error_sig), error_sig, neg_error_sig)
    if plot_flag:
        fig, ax = plot_funcs.subplots_plot(1, 1)
        ax.plot(error_sig, label=f'error signal std={np.std(error_sig)}')
        ax.plot(neg_error_sig, label=f'negative error signal std={np.std(neg_error_sig)}')
        ax.plot(best_error_sig, label='min error signal')
        ax.legend()
    if metric == 'MAE':
        error = np.mean(np.abs(error_sig))
        neg_error = np.mean(np.abs(neg_error_sig))
        best_error = np.mean(np.abs(best_error_sig))
    else:
        error = np.sqrt(np.mean(error_sig ** 2))
        neg_error = np.sqrt(np.mean(neg_error_sig ** 2))
        best_error = np.sqrt(np.mean(best_error_sig ** 2))
    if error < neg_error:
        return error, np.std(error_sig)
    else:
        return neg_error, np.std(neg_error_sig)


def embed_error_old(embed, real_angles, plot_flag=False, metric='MAE', center_data=False):
    # calculate angles from embedding
    embed_r = np.real(embed)  # make sure embed is real
    if center_data:
        embed_r = embed_r - np.mean(embed_r, axis=0)
    embed_angles = np.arctan2(embed_r[:, 1], embed_r[:, 0])
    embed_angles = embed_angles * 180 / np.pi  # convert from radians to degrees
    neg_embed_angles = -embed_angles  # if angles change counter-clockwise need to use the negative of the angles
    error_sig = embed_angles - real_angles
    error_sig = error_sig - np.mean(error_sig)
    error_sig = lin2circ_angles(error_sig, rep_type='m180-180')  # angle differences should be in [0,360] range
    error_sig = error_sig - np.mean(error_sig)
    neg_error_sig = neg_embed_angles - real_angles
    neg_error_sig = neg_error_sig - np.mean(neg_error_sig)
    neg_error_sig = lin2circ_angles(neg_error_sig, rep_type='m180-180')  # angle differences should be in [0,360] range
    neg_error_sig = neg_error_sig - np.mean(neg_error_sig)
    # remove angle offset since the error metric needs to be invariant to rotations
    # choose lowest error at each entry
    best_error_sig = np.where(np.abs(error_sig) < np.abs(neg_error_sig), error_sig, neg_error_sig)
    if plot_flag:
        fig, ax = plot_funcs.subplots_plot(1, 1)
        ax.plot(error_sig, label=f'error signal std={np.std(error_sig)}')
        ax.plot(neg_error_sig, label=f'negative error signal std={np.std(neg_error_sig)}')
        ax.plot(best_error_sig, label='min error signal')
        ax.legend()
    if metric == 'MAE':
        error = np.mean(np.abs(error_sig))
        neg_error = np.mean(np.abs(neg_error_sig))
        best_error = np.mean(np.abs(best_error_sig))
    else:
        error = np.sqrt(np.mean(error_sig**2))
        neg_error = np.sqrt(np.mean(neg_error_sig**2))
        best_error = np.sqrt(np.mean(best_error_sig ** 2))
    if error < neg_error:
        return error, np.std(error_sig)
    else:
        return neg_error, np.std(neg_error_sig)


def get_validation_indices(sim_params, seed=0):
    '''
    :param array: array to divide into batches
    :param batch_size: size of each batch
    :param seed: random seed
    :return: batches: indices of the batch division
    '''

    # Generate an array of indices and shuffle them
    indices = np.arange(sim_params['N'])
    np.random.seed(seed)
    np.random.shuffle(indices)

    N_valid = round(sim_params['valid_size'] * sim_params['N'])

    return indices[:N_valid]


def plot_method_embedding(embed, figures_path, angles, Nr, method, ref_idx, plot_flag=True, pointsize=20,
                          pointsize_ref=30, fontproperties=None, tick_fontproperties=None, figsize=(8, 7)):
    fig, ax = plot_funcs.subplots_plot(1, 1, figsize=figsize)
    colors = lin2circ_angles(angles)
    N_d = embed.shape[0]
    total_idx = np.round(np.linspace(0, N_d - 1, N_d)).astype(int)
    single_idx = [i for i in total_idx if i not in ref_idx]
    # define font
    if fontproperties is None:
        my_fontproperties = font_manager.FontProperties(family='Times New Roman', size=18)
    else:
        my_fontproperties = fontproperties
    if tick_fontproperties is None:
        my_tick_fontproperties = font_manager.FontProperties(family='Times New Roman', size=18)
    else:
        my_tick_fontproperties = tick_fontproperties
    ax.scatter(embed[ref_idx, 0], embed[ref_idx, 1], marker='x', c=colors[ref_idx],
               label='Full View', s=pointsize_ref, cmap='viridis')
    ax.scatter(embed[single_idx, 0], embed[single_idx, 1], marker='.', c=colors[single_idx],
               label='Partial View', s=pointsize, cmap='viridis')

    # plot a circle to show how well it fits into a circle
    # Calculate radii from the origin (assuming points are centered at (0, 0))
    radii = np.linalg.norm(embed, axis=1)
    median_radius = np.median(radii)
    quantile_15 = np.quantile(radii, 0.15)
    quantile_85 = np.quantile(radii, 0.85)

    # Plot the median radius circle
    circle = plt.Circle((0, 0), median_radius, color='black', linestyle='--', fill=False, label='Median Radius')
    ax.add_artist(circle)
    # circle = plt.Circle((0, 0), quantile_25, color='gray', linestyle='--', alpha=0.5, fill=False,
    #                     label='0.25 Quant Radius')
    # ax.add_artist(circle)
    # circle = plt.Circle((0, 0), quantile_75, color='gray', linestyle='--', alpha=0.5, fill=False,
    #                     label='0.75 Quant Radius')
    # ax.add_artist(circle)

    # Plot the shaded region for the 0.25 and 0.75 quantile radii
    theta = np.linspace(0, 2 * np.pi, 100)
    x_quantile_15 = quantile_15 * np.cos(theta)
    y_quantile_15 = quantile_15 * np.sin(theta)
    x_quantile_85 = quantile_85 * np.cos(theta)
    y_quantile_85 = quantile_85 * np.sin(theta)
    ax.fill(np.concatenate([x_quantile_85, x_quantile_15[::-1]]),
            np.concatenate([y_quantile_85, y_quantile_15[::-1]]),
            color='gray', alpha=0.3, label='0.15-0.85 Quantile Radius')

    # Set axis aspect ratio to be equal
    ax.set_aspect('equal')
    max_extent = np.max(np.abs(embed)) * 1.1  # Add a small margin
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)

    # ax.set_title("M&M Angle Colors - Spectral Completed Data, $N_R = {}$".format(Nr))
    ax.set_xlabel("First Diffusion Coordinate", font_properties=my_fontproperties)
    ax.set_ylabel("Second Diffusion Coordinate", font_properties=my_fontproperties)
    # ax.legend(loc='upper right')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(my_tick_fontproperties)
    if plot_flag:
        plt.show()
    plt.savefig(f"{figures_path}/{method}_embedding.pdf", dpi=300, format='pdf', bbox_inches='tight')

    fig, ax = plot_funcs.subplots_plot(1, 1, figsize=figsize)
    ax.scatter(embed[single_idx, 0], embed[single_idx, 1], marker='.', c='r',
               label='Partial View', s=pointsize)
    ax.scatter(embed[ref_idx, 0], embed[ref_idx, 1], marker='.', c='b', label='Full View', s=pointsize)
    # ax.set_title("Spectral Completed - Reference Vs. Completed")
    ax.set_xlabel("First Diffusion Coordinate", font_properties=my_fontproperties)
    ax.set_ylabel("Second Diffusion Coordinate", font_properties=my_fontproperties)

    ax.legend(loc='upper right')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(my_fontproperties)
    if plot_flag:
        plt.show()
    plt.savefig(f"{figures_path}/{method}_embedding_comp_vs_ref.pdf", dpi=300, format='pdf', bbox_inches='tight')

