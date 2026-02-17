from skimage.measure import regionprops, find_contours
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
from sklearn.neighbors import NearestNeighbors
from deeptime.data import bickley_jet
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
from helper_functions.embed_methods import embed_wrapper
from deeptime.clustering import KMeans
from sklearn.metrics import silhouette_score
from helper_functions.clustering_funcs import align_clusters


def create_grid(points, labels, grid_size_x, grid_size_y, bounds=None):
    """
    Assign points to a discrete grid and perform majority voting for each pixel.
    """
    # Determine the bounds of the grid (if not provided)
    if bounds is None:
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
    else:
        min_x, min_y, max_x, max_y = bounds

    # Create the grid
    x_bins = np.linspace(min_x, max_x, grid_size_x)
    y_bins = np.linspace(min_y, max_y, grid_size_y)

    # Digitize the points into grid bins
    x_indices = np.digitize(points[:, 0], x_bins) - 1
    y_indices = np.digitize(points[:, 1], y_bins) - 1

    # Create an empty grid to hold cluster assignments
    grid = np.zeros((grid_size_x, grid_size_y), dtype=int)

    # Assign points to the grid based on majority vote
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            # Get points in the current grid cell
            mask = (x_indices == i) & (y_indices == j)
            if np.any(mask):
                # Perform majority vote to assign the cluster
                majority_label = np.argmax(np.bincount(labels[mask]))
                grid[i, j] = majority_label  # Labels start at 1

    return grid


def estimate_volume(grid):
    """
    Estimate the volume of each cluster by counting the number of pixels in the grid.
    """
    unique_labels, counts = np.unique(grid, return_counts=True)
    volumes = {label: count for label, count in zip(unique_labels, counts)}
    return volumes


def estimate_circumference(grid):
    """
    Estimate the circumference of each cluster using boundary detection on the grid.
    """
    circumferences = {}
    unique_labels = np.unique(grid)

    for label in unique_labels:
        # Find the boundaries of the cluster on the grid
        contours = find_contours(grid == label, level=0.5)
        total_perimeter = sum(len(contour) for contour in contours)

        # Approximate the circumference as the sum of boundary lengths
        circumferences[label] = total_perimeter

    return circumferences


def dynamic_isoperimetric_score(points1, points2, labels, grid_size_x, grid_size_y, bounds=None):
    grid1 = create_grid(points1, labels, grid_size_x, grid_size_y, bounds)
    volumes1 = estimate_volume(grid1)
    circumferences1 = estimate_circumference(grid1)
    grid2 = create_grid(points2, labels, grid_size_x, grid_size_y, bounds)
    volumes2 = estimate_volume(grid2)
    circumferences2 = estimate_circumference(grid2)
    isoperims = []
    circumference_ratios = []
    volume_ratios = []
    unique_labels = np.unique(labels)
    # calculate dynamic isoperimertic ratio for each cluster
    for label in unique_labels:
        circumference_avg = (circumferences1[label] + circumferences2[label]) / 2
        volume_avg = (volumes1[label] + volumes2[label]) / 2
        volume_min = min(volumes1[label], volumes2[label])
        circumference_min = min(circumferences1[label], circumferences2[label])
        isoperims.append(circumference_avg / volume_min)
        circumference_ratios.append(circumference_avg / circumference_min)
        volume_ratios.append(volume_avg / volume_min)

    return (grid1, grid2, np.array(isoperims).mean(), np.array(circumference_ratios).mean(),
            np.array(volume_ratios).mean())


def calc_graph_volume_perimeter(A, cluster_labels):
    '''

    :param A: adjacency matrix
    :param cluster_labels: cluster assignment
    :return: graph volume - sum over all edge weights connecting the cluster
    '''

    unique_labels = np.unique(cluster_labels)
    volumes = {}
    perimeters = {}
    # calculate node degrees
    node_degrees = np.sum(A, axis=1)
    for cluster in unique_labels:
        # extract all nodes in the cluster
        cluster_nodes = np.where(cluster_labels == cluster)[0]
        volumes[cluster] = node_degrees[cluster_nodes].sum()
        # calculate perimeter
        perimeter = 0
        for node in cluster_nodes:
            non_cluster_nodes = np.where(cluster_labels != cluster)[0]
            # get all perimeter edge weights
            neighbors = A[node, non_cluster_nodes]
            perimeter += np.sum(neighbors)
        perimeters[cluster] = perimeter

    return volumes, perimeters


def calc_graph_isoperim(A, labels):
    volumes, perimeters = calc_graph_volume_perimeter(A, labels)
    volume_tot = np.sum(A)
    unique_labels = np.unique(labels)
    isoperims = []
    for label in unique_labels:
        volume_min = min(volumes[label], volume_tot - volumes[label])
        isoperims.append(perimeters[label] / volume_min)

    return np.array(isoperims).mean()


def calc_graph_dynamic_isoperim(A1, A2, labels):
    '''

    :param A1: first view adjacency matrix
    :param A2: second view adjacency matrix
    :param labels: cluster assignment
    :return: the dynamic isoperimetric ratio (Froyland 2015) of the cluster assignment.
    '''
    isoperim1 = calc_graph_isoperim(A1, labels)
    isoperim2 = calc_graph_isoperim(A2, labels)

    return (isoperim1 + isoperim2) / 2


def calc_graph_dynamic_isoperim_4(A1, A2, A3, A4, labels):
    isoperim1 = calc_graph_isoperim(A1, labels)
    isoperim2 = calc_graph_isoperim(A2, labels)
    isoperim3 = calc_graph_isoperim(A3, labels)
    isoperim4 = calc_graph_isoperim(A4, labels)

    return np.array([isoperim1, isoperim2, isoperim3, isoperim4]).mean()


def calc_embed_isoperim(embed, labels, k=10, symmetric=True):
    '''

    :param embed: embedding of the points
    :param labels: cluster assignment
    :param k: number of neighbors for the graph construction
    :param symmetric: if k-nn are considered symmetric
    :return: the isoperimetric ratio of the clustering (sort of a min-cut score)
    '''
    # calculate a k-NN adjacency matrix in the embedding and calculate the isoperimetric ratio
    # Fit k-NN model on the embedding
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(embed)

    # Find the k-nearest neighbors for each point
    distances, indices = nbrs.kneighbors(embed)

    # Initialize the adjacency matrix
    n_points = embed.shape[0]
    adjacency_matrix = np.zeros((n_points, n_points))

    # Populate the adjacency matrix
    for i in range(n_points):
        for j in range(1, k):  # Start from 1 to avoid self-loops
            adjacency_matrix[i, indices[i, j]] = 1
            if symmetric:
                adjacency_matrix[indices[i, j], i] = 1  # Make symmetric for regular k-NN graph

    # calculate isoperimetric ratio
    volumes, perimeters = calc_graph_volume_perimeter(adjacency_matrix, labels)
    unique_labels = np.unique(labels)
    isoperims = []
    for label in unique_labels:
        isoperims.append(perimeters[label] / volumes[label])

    return np.array(isoperims).mean()


def create_dataset(sim_params, figures_path, seed=None):
    n_particles = sim_params['N']
    dataset = bickley_jet(n_particles, n_jobs=8, seed=seed)
    c = np.copy(dataset[0, :, 0])
    c /= c.max()
    if sim_params['animate']:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ani = dataset.make_animation(c=c, agg_backend=False, interval=75, fig=fig, ax=ax, max_frame=100, s=50)
        ani.save(f'{figures_path}/bickley_jet_animation.mp4', writer='ffmpeg', fps=30, dpi=300)
    return dataset, c


def plot_clustering(dataset, c, figures_path, method, font_properties, sim_params, cmap='viridis'):
    # Create the custom colormap
    dark2 = get_cmap('Dark2', 8)  # Get 9 colors from the Dark2 colormap
    colors = dark2(np.linspace(0, 1, 8))  # Extract the 9 colors
    colors = np.vstack([colors, [0.2, 0.4, 1, 1]])  # Append white color ([R,G,B,A] = [1,1,1,1])

    # Create a custom ListedColormap
    custom_cmap = mcolors.ListedColormap(colors)

    Nr = sim_params['Nr']
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    data = dataset.endpoints_dataset().data
    if cmap == 'custom':
        cmap = custom_cmap
    scatter = plt.scatter(data[Nr:, 0], data[Nr:, 1], c=c[Nr:], s=10, marker='o', cmap=cmap)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('coherent set', fontproperties=font_properties)
    # Apply font properties to colorbar tick labels
    # color_labels = np.linspace(0, sim_params['clusters'] - 1, sim_params['clusters']).astype(int)
    color_labels = np.arange(sim_params['clusters'])
    cbar.set_ticks(np.linspace(0, sim_params['clusters'] - 1.9, sim_params['clusters']) + 0.5)
    cbar.set_ticklabels(color_labels, fontproperties=font_properties)
    # for label in cbar.ax.get_yticklabels():
    #     label.set_fontproperties(font_properties)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_properties)
    plt.xlabel('x', fontproperties=font_properties)
    plt.ylabel('y', fontproperties=font_properties)
    # Set background to white
    ax.set_facecolor('white')  # Axes background
    fig.patch.set_facecolor('white')  # Figure background
    # Remove grid
    ax.grid(False)
    plt.savefig(f'{figures_path}/{method}_clustering_partial.pdf', dpi=300, format='pdf', bbox_inches='tight')
    scatter = plt.scatter(data[:Nr, 0], data[:Nr, 1], c=c[:Nr], s=10, marker='^', cmap=cmap)
    plt.savefig(f'{figures_path}/{method}_clustering.pdf', dpi=300, format='pdf', bbox_inches='tight')


def plot_hit_or_miss(data, c, c_ref, figures_path, method, font_properties, sim_params, cmap='viridis'):
    '''
    Plots locations with hit or miss based on correct clustering of c relatively to c_ref
    :param data: location of particles to plot
    :param c: clustering assignment
    :param c_ref: reference (ground truth) clustering assignment
    :param figures_path: path to save plot
    :param method: method used for the name of the saved file
    :param font_properties:
    :param sim_params: simulation parameters
    :param cmap: color map used for the clusters
    :return:
    '''
    # Create the custom colormap
    dark2 = get_cmap('Dark2', 8)  # Get 9 colors from the Dark2 colormap
    colors = dark2(np.linspace(0, 1, 8))  # Extract the 9 colors
    colors = np.vstack([colors, [0.2, 0.4, 1, 1]])  # Append white color ([R,G,B,A] = [1,1,1,1])

    # Create a custom ListedColormap
    custom_cmap = mcolors.ListedColormap(colors)

    Nr = sim_params['Nr']
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    if cmap == 'custom':
        cmap = custom_cmap
    scatter = plt.scatter(data[c == c_ref, 0], data[c == c_ref, 1], c=c[c == c_ref], s=10, marker='.', cmap=cmap)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('coherent set', fontproperties=font_properties)
    # Apply font properties to colorbar tick labels
    # color_labels = np.linspace(0, sim_params['clusters'] - 1, sim_params['clusters']).astype(int)
    color_labels = np.arange(sim_params['clusters'])
    cbar.set_ticks(np.linspace(0, sim_params['clusters'] - 1.9, sim_params['clusters']) + 0.5)
    cbar.set_ticklabels(color_labels, fontproperties=font_properties)
    # for label in cbar.ax.get_yticklabels():
    #     label.set_fontproperties(font_properties)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_properties)
    plt.xlabel('x', fontproperties=font_properties)
    plt.ylabel('y', fontproperties=font_properties)
    # Set background to white
    ax.set_facecolor('white')  # Axes background
    fig.patch.set_facecolor('white')  # Figure background
    # Remove grid
    ax.grid(False)
    scatter = plt.scatter(data[c != c_ref, 0], data[c != c_ref, 1], c=c[c != c_ref], s=15, marker='x', cmap=cmap)
    plt.xlim([0, 20])
    plt.ylim([-3, 3])
    plt.savefig(f'{figures_path}/{method}_hit_or_miss.pdf', dpi=300, format='pdf', bbox_inches='tight')


def plot_wrapper(data, c, figures_path, method, font_properties, sim_params, cmap='viridis',
                 c_ref=None, plot_type='scatter'):
    # Create the custom colormap
    dark2 = get_cmap('Dark2', 8)  # Get 9 colors from the Dark2 colormap
    colors = dark2(np.linspace(0, 1, 8))  # Extract the 9 colors
    colors = np.vstack([colors, [0.2, 0.4, 1, 1]])  # Append white color ([R,G,B,A] = [1,1,1,1])

    # Create a custom ListedColormap
    custom_cmap = mcolors.ListedColormap(colors)

    Nr = sim_params['Nr']
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    if cmap == 'custom':
        cmap = custom_cmap

    norm = mcolors.Normalize(vmin=0, vmax=sim_params['clusters'] - 1)

    # Choose the required plot type
    if plot_type == 'scatter':
        scatter = plt.scatter(data[Nr:, 0], data[Nr:, 1], c=c[Nr:], s=10, marker='o', cmap=cmap)
        scatter = plt.scatter(data[:Nr, 0], data[:Nr, 1], c=c[:Nr], s=10, marker='^', cmap=cmap)
    elif plot_type == 'hit-or-miss':
        if c_ref is None:
            c_ref = c
            raise Warning('No Reference Clustering Provided for Hit or Miss Plot')
        # if clustering is the same don't plot misses
        if not np.all(c == c_ref):
            # scatter = plt.scatter(data[c != c_ref, 0], data[c != c_ref, 1], c=c[c != c_ref],
            # s=15, marker='x', cmap=cmap)
            # scatter = plt.scatter(data[c != c_ref, 0], data[c != c_ref, 1], c=c[c != c_ref], s=15, marker='.',
            #                       cmap=cmap, edgecolors='black', norm=norm)
            scatter = plt.scatter(data[c != c_ref, 0], data[c != c_ref, 1], s=15, marker='o',
                                  edgecolors=cmap(norm(c[c != c_ref])), facecolors='none')
        scatter = plt.scatter(data[c == c_ref, 0], data[c == c_ref, 1], c=c[c == c_ref], s=15, marker='.', cmap=cmap,
                              norm=norm)



    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('coherent set', fontproperties=font_properties)
    # Apply font properties to colorbar tick labels
    # color_labels = np.linspace(0, sim_params['clusters'] - 1, sim_params['clusters']).astype(int)
    color_labels = np.arange(sim_params['clusters'])
    cbar.set_ticks(np.linspace(0, sim_params['clusters'] - 1.9, sim_params['clusters']) + 0.5)
    cbar.set_ticklabels(color_labels, fontproperties=font_properties)
    # for label in cbar.ax.get_yticklabels():
    #     label.set_fontproperties(font_properties)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_properties)
    plt.xlabel('x', fontproperties=font_properties)
    plt.ylabel('y', fontproperties=font_properties)
    # Set background to white
    ax.set_facecolor('white')  # Axes background
    fig.patch.set_facecolor('white')  # Figure background
    # Remove grid
    ax.grid(False)
    # plt.xlim([0, 20])
    # plt.ylim([-3, 3])
    plt.savefig(f'{figures_path}/{method}_{plot_type}.pdf', dpi=300, format='pdf', bbox_inches='tight')


def create_views(dataset, sim_params):
    # ref split
    Nr = sim_params['Nr']
    if sim_params['views'] == 'traj':
        # use trajectories as views
        traj_len = sim_params['traj_len']
        s1_full = flatten_timeseries(dataset.data[:traj_len])
        s2_full = flatten_timeseries(dataset.data[-traj_len:])
        return s1_full, s2_full, None, None
    elif sim_params['views'] == 'frame':
        # use a frame at different times
        data_end = dataset.endpoints_dataset()
        s1_full = data_end.data
        s2_full = data_end.data_lagged
        return s1_full, s2_full, None, None
    elif sim_params['views'] == 'short_traj':
        # use a short trajectory at the end of the dynamics
        lag = sim_params['lag']
        traj_len = sim_params['traj_len']
        s1_full = flatten_timeseries(dataset.data[-traj_len:])
        s2_full = flatten_timeseries(dataset.data[-traj_len - lag: -lag])
        return s1_full, s2_full, None, None
    elif sim_params['views'] == 'multi_frame':
        times = sim_params['times']
        s1_full = dataset.data[times[0]]
        s2_full = dataset.data[times[1]]
        s3_full = dataset.data[times[2]]
        s4_full = dataset.data[times[3]]
        return s1_full, s2_full, s3_full, s4_full
    else:
        raise ValueError(f'invalid views type {sim_params["views"]}')


def method_analysis(s1_ref, s1_full, s2_ref, s2_full, method, sim_params, figures_path, K1, K2, dataset,
                    font_properties, c_align=None):
    embed = embed_wrapper(s1_ref, s1_full, s2_ref, s2_full, method=method, embed_dim=sim_params['embed_dim'],
                          t=sim_params['t'], K1=K1, K2=K2)
    kmeans = KMeans(n_clusters=sim_params['clusters'], n_jobs=8).fit(embed).fetch_model()
    c = kmeans.transform(embed)
    if c_align is not None:
        c = align_clusters(c_align, c)
    if sim_params['animate']:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ani = dataset.make_animation(c=c/sim_params['clusters'], agg_backend=False, interval=75, fig=fig, ax=ax,
                                     s=50, cmap='Set1')
        ani.save(f'{figures_path}/{method}_clustering_animation.mp4', writer='ffmpeg', fps=10, dpi=300)
    plot_clustering(dataset, c, figures_path, method, cmap=sim_params['cmap'], font_properties=font_properties,
                    sim_params=sim_params)
    return embed, c


def evaluate_metrics(s1_full, s2_full, c, embed, sim_params, K1, K2, K3=None, K4=None, method='not specified',
                     rep='no reps'):
    score1 = silhouette_score(s1_full, c)
    score2 = silhouette_score(s2_full, c)
    score_embed = silhouette_score(embed, c)
    # grid1, grid2, isoperim_score, circumferernce_ratio, volume_ratio = dynamic_isoperimetric_score(s1_full, s2_full, c,
    #                                                                                                grid_size_x=50,
    #                                                                                                grid_size_y=15,
    #                                                                                                bounds=(
    #                                                                                                0, -3, 20, 3))
    if sim_params['views'] == 'multi_frame':
        isoperim4 = calc_graph_dynamic_isoperim_4(K1, K2, K3, K4, c)
    else:
        isoperim4 = 'N/a'
    new_line = {
        'method': method,
        'rep': rep,
        'silhouette_score_s1': score1,
        'silhouette_score_s2': score2,
        'silhouette_score_avg': (score1 + score2) / 2,
        'silhouette_score_embed': score_embed,
        # 'isoperim_score': isoperim_score,
        # 'circumferernce_ratio': circumferernce_ratio,
        # 'volume_ratio': volume_ratio,
        'dynamic_graph_isoperim_score': calc_graph_dynamic_isoperim(K1, K2, c),
        'dynamic_graph_isoperim_score_4': isoperim4,
        'embed_graph_isoperim_score_k_10': calc_embed_isoperim(embed, c, k=10, symmetric=True),
        'embed_graph_isoperim_score_k_50': calc_embed_isoperim(embed, c, k=50, symmetric=True)}
    return new_line


def flatten_timeseries(data):
    '''

    :param data: timeseries data (timestamps x particles x position)
    :return: fetatures: flattened timeseries data (particles x timstamps * position)

    '''
    # rearange dimensions to get (particles x timestamps x position)
    data_transposed = np.transpose(data, (1, 0, 2))
    # flatten data
    return data_transposed.reshape((data_transposed.shape[0], -1))

