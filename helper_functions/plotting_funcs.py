# %% import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator

# %% set global parameters

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

# %% create fig
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

