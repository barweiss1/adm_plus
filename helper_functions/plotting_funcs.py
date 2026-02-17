#  import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
from sklearn.manifold import TSNE
import numpy as np

# set global parameters

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

# create fig
# fig = plt.figure(figsize=(10,6)) # units are inch
# ax = plt.axes((0.1,0.1,0.5,0.8))
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['top'].set_visible(False)


def remove_spines(ax, flag_r=False, flag_l=False, flag_b=False, flag_t=False):
    ax.spines['right'].set_visible(flag_r)
    ax.spines['left'].set_visible(flag_l)
    ax.spines['bottom'].set_visible(flag_b)
    ax.spines['top'].set_visible(flag_t)

    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    return ax


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


# create subplots for imshow - remove spines and ticks and create fig and ax objects
def subplots_imshow(nrows, ncols, figsize=(8, 4), spines=True):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if (nrows == 1) and (ncols == 1):
        ax = remove_ticks(ax)
        ax = remove_spines(ax, spines, spines, spines, spines)
    else:
        ax = ax.ravel()
        for i in range(nrows*ncols):
            ax[i] = remove_spines(ax[i], spines, spines, spines, spines)
            ax[i] = remove_ticks(ax[i])

    return fig, ax


# create subplots for plots - remove spines and ticks and create fig and ax objects
def subplots_plot(nrows, ncols, figsize=(8, 4)):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if (nrows == 1) and (ncols == 1):
        ax = remove_spines(ax,flag_l=True,flag_b=True)
    else:
        ax = ax.ravel()
        for i in range(nrows*ncols):
            ax[i] = remove_spines(ax[i],flag_l=True,flag_b=True)

    return fig, ax


# plot 3D embedding
def plot_3d_embed(embed, figsize=(8, 8), title=None, colors='b'):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2], marker='.', c=colors)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("First Diffusion Coordinate")
    ax.set_ylabel("Second Diffusion Coordinate")
    ax.set_zlabel("Third Diffusion Coordinate")
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    return fig, ax


# plot 2D embedding
def plot_2d_embed(embed, figsize=(8, 8), title='', colors='b', ref_indicator=None):
    fig, ax = subplots_plot(1, 1, figsize=figsize)
    if ref_indicator is None:
        ax.scatter(embed[:, 0], embed[:, 1], marker='o', s=15, c=colors)
    else:
        ax.scatter(embed[ref_indicator == 0, 0], embed[ref_indicator == 0, 1],
                   marker='^', s=15, c=colors[ref_indicator == 0])
        ax.scatter(embed[ref_indicator == 1, 0], embed[ref_indicator == 1, 1],
                   marker='o', s=15, c=colors[ref_indicator == 1])
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("First Diffusion Coordinate")
    ax.set_ylabel("Second Diffusion Coordinate")
    ax.set_aspect('equal')
    return fig, ax


# wrapper function that calls 2d or 3d plot based on embedding dimension
def plot_embed(embed, figsize=(8, 8), title='', colors='b', ref_indicator=None):
    embed_dim = embed.shape[1]
    if embed_dim == 2:
        return plot_2d_embed(embed, figsize=figsize, title=title, colors=colors, ref_indicator=ref_indicator)
    elif embed_dim == 3:
        return plot_3d_embed(embed, figsize=figsize, title=title, colors=colors)
    else:
        print("Invalid Embedding Dimension for Plot")
        return None, None


def plot_embed_tsne(embedding, perplexity=30, n_iter=1000, random_state=42, title=None, colors='b', ref_indicator=None,
                    color_labels=None, cmap='Dark2', figsize=(8, 7), show_legend=True, point_size=15, fontsize=24,
                    font_properties=None):
    """
    Visualizes high-dimensional data in 2D using t-SNE.

    Parameters:
    - embedding: numpy array or similar, shape (n_samples, n_features)
        High-dimensional data to be visualized.
    - perplexity: float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
    - n_iter: int, optional (default: 1000)
        Number of iterations for optimization.
    - random_state: int, optional (default: 42)
        Random state for reproducibility.
    - color_labels: list of strings, optional
        Labels for each color in the colorbar.

    Returns:
    - None: The function displays the t-SNE plot.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(embedding)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    if ref_indicator is None:
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], s=point_size, c=colors, cmap=cmap, marker='.')
    else:
        scatter = ax.scatter(tsne_results[ref_indicator == 0, 0], tsne_results[ref_indicator == 0, 1], s=point_size,
                             c=colors[ref_indicator == 0], cmap=cmap, marker='.', label='out of reference')
        ax.scatter(tsne_results[ref_indicator == 1, 0], tsne_results[ref_indicator == 1, 1], s=2*point_size,
                   c=colors[ref_indicator == 1], cmap=cmap, marker='x', label='reference')

    # Set titles and labels
    if title is not None:
        ax.set_title(f't-SNE {title}')
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    if font_properties is not None:
        plt.xticks(fontsize=fontsize, fontproperties=font_properties)  # Increase x-tick fontsize
        plt.yticks(fontsize=fontsize, fontproperties=font_properties)  # Increase y-tick fontsize
        plt.xlabel('t-SNE Dimension 1', fontsize=fontsize, fontproperties=font_properties)
        plt.ylabel('t-SNE Dimension 2', fontsize=fontsize, fontproperties=font_properties)
    else:
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
    # Adding colorbar with labels
    if color_labels is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_ticks(np.linspace(0, len(color_labels) - 1.9, len(color_labels)) + 0.5)
        cbar.set_ticklabels(color_labels, fontproperties=font_properties)

    # Customizing legend
    if show_legend:
        legend = ax.legend()
        for handle in legend.legendHandles:
            handle.set_facecolor('black')  # Set the facecolor to black
            handle.set_edgecolor('black')  # Set the edgecolor to black

    return fig, ax