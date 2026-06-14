# ADM+

This repository contains the code used for the paper **"Extracting Common Components from Partially Observed Views Using Diffusion Geometry"**, published in **Transactions on Machine Learning Research (TMLR)**.

The project studies multi-view data in which only a subset of samples is observed in all views. The main goal is to recover the latent component that is common across the views, even when many samples are available in only one view. The code implements ADM+ and several diffusion-geometry and kernel-based baselines, and reproduces the experiments from the paper.

## What Is Included

- ADM+ and related partial-view embedding methods.
- Full-view baselines such as diffusion maps and alternating diffusion maps.
- Partial-view baselines including Nystrom ADM, NCCA, APMC, KCCA with imputation, forward-only ADM, and backward-only ADM.
- Experiments for rotating image pairs, Bickley jet coherent-set recovery, and fMRI task classification.
- Plotting and post-processing notebooks for the main paper and appendix figures.

## Installation

Create a Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


The code is organized as plain Python modules and notebooks. Run scripts and notebooks from the repository root so imports such as `helper_functions.embed_methods` resolve correctly.

## Repository Structure

```text
.
|-- helper_functions/
|   |-- embed_methods.py       # ADM+, diffusion maps, ADM variants, NCCA, KCCA, APMC
|   |-- embed_both_views.py    # class-based interface for two partially observed views
|   |-- embed_utils.py         # kernel construction, Markov normalization, SVD/EVD helpers
|   |-- logo_funcs.py          # rotating-logo data generation, evaluation, and plotting
|   |-- bickley_funcs.py       # Bickley jet data, clustering, and coherent-set metrics
|   |-- fmri_funcs.py          # fMRI loading, distances, embeddings, classification
|   |-- clustering_funcs.py    # clustering alignment and evaluation helpers
|   |-- plotting_funcs.py      # shared plotting utilities
|   |-- opt_methods.py         # optimization-based incomplete multi-view baselines
|   `-- utils.py               # sparse-matrix I/O, NaN/inf handling, memory logging
|-- logo_data/                 # input images for the rotating-logo experiment
|-- fmri_data/                 # local fMRI data used by the fMRI scripts
|-- animations/                # paper animations and visual illustrations
|-- *_main*.ipynb              # experiment notebooks
|-- *_server.py                # parallel experiment runners
`-- requirements.txt
```

## Core ADM+ Interfaces

The repository contains two ADM+ interfaces, matching the two partially observed settings described in the paper:

- `helper_functions/embed_both_views.py` provides a class interface for the setting where both views may have missing samples. This is the two-missing-views variant described in Appendix A of the paper, and it uses a different algorithmic construction from the single-missing-view case.
- `helper_functions/embed_methods.py` provides the function interface used by the original notebooks and scripts for the single-missing-view setting, where one view is fully available and the other is available only on the reference set.

### Two Missing Views

Use the class-based implementation when samples may be missing from either view.

Missing samples are represented by rows whose entries are all `NaN`. Samples observed in both views are used as the reference/anchor set, while samples observed in only one view are embedded through the two-missing-views ADM+ extension and fused into a common embedding.

```python
import numpy as np
from helper_functions.embed_both_views import ADM_PLUS, ADMPlusConfig

cfg = ADMPlusConfig(
    embed_dim=30,
    kernel_scale1=1.0,
    kernel_scale2=1.0,
    t=0.1,
    fusion_method="apmc",
)

model = ADM_PLUS(cfg)
model.fit(X1, X2)       # X1 and X2 have shape (n_samples, n_features)
embedding = model.get_embedding()
```

### Single Missing View

Use the function-based implementation when the first view is available for all samples and the second view is available only for the reference samples. This is the interface used by most of the original experiment notebooks.

The relevant functions are `adm_plus` and `embed_wrapper` in `helper_functions/embed_methods.py`.

## Experiments

### Rotating Images

The rotating-image experiment creates two synchronized views. Each view contains a view-specific rotating object and a shared rotating object. The task is to recover the common rotation angle.

- `logo_main_scales.ipynb` runs the main rotating-image experiment across common-image sizes.
- `logo_main_discrepency.ipynb` runs the rotating-image experiment with a distribution discrepancy between the fully observed reference set and the partially observed set.
- `logo_main_server.py` is the parallel runner for larger rotating-image sweeps.
- `logo_plots.ipynb` post-processes rotating-image results and creates plots.
- `logo_two_missing.ipynb` runs the appendix experiment with two missing views.

The raw image assets are included in `logo_data/`. Generated arrays and results are written under `logo_data/` and `figures/logo/` depending on the simulation parameters.

Example parallel run:

```bash
python logo_main_server.py <task_id> <n_tasks>
```

On SLURM:

```bash
python logo_main_server.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
```

For multiple machines or CPU workers, pass a different `task_id` in `[0, n_tasks - 1]` and the same `n_tasks` to each process.

### Bickley Jet

The Bickley jet experiment uses the `deeptime` Bickley jet simulator to study coherent-set recovery from partially observed dynamical views.

- `bickley_main.ipynb` runs the Bickley jet experiment.
- `helper_functions/bickley_funcs.py` contains the dataset creation, clustering, dynamic isoperimetric scores, graph metrics, and plotting utilities.
- `animations/` contains visual illustrations used for the Bickley jet experiment.

Outputs are written to a `figures/bickley_*` directory.

### fMRI Task Classification

The fMRI experiment evaluates common-component embeddings on task classification from two fMRI views.

- `fmri_main.py` is a smaller single-run script.
- `fmri_main_server.py` is the parallel runner used for larger hyperparameter sweeps.
- `fMRI_plots.ipynb` creates runtime, hyperparameter-sensitivity, and embedding visualizations.
- `helper_functions/fmri_funcs.py` contains the fMRI data loading, distance computation, embedding, and classification code.

The scripts expect LR fMRI FCNs data under `fmri_data/` (we computed it using the code from https://github.com/carricky/graph_learning_FC). Distance matrices are cached as `fmri_data/dist_mats_euclidean.pkl` when generated.

Example run:

```bash
python fmri_main_server.py <task_id> <n_tasks>
```

To include the optimization-based baselines, which are substantially slower:

```bash
python fmri_main_server.py <task_id> <n_tasks> --run_opt_methods
```

After all parallel jobs finish, merge machine outputs with:

```bash
python fmri_main_server--summarize_results
```

### Appendix and Illustrations

- `diffusion_simulation.ipynb` creates diffusion-geometry illustrations.
- `logo_two_missing.ipynb` runs the two-missing-views rotating-image appendix experiment.
- `animations/method_comparison.gif` and `animations/physics_animation.gif` provide visual summaries and Bickley jet animations.

## Implemented Methods

The main embedding methods are implemented in `helper_functions/embed_methods.py` and wrapped by `embed_wrapper`.

Common method keys used by the notebooks and scripts include:

- `adm_plus`: ADM+ for partially observed views.
- `ad`: alternating diffusion maps on fully observed paired views.
- `ad_svd`: SVD-based alternating diffusion variant.
- `dm`: single-view diffusion maps.
- `forward_only`: forward partial-view ADM extension.
- `backward_only`: backward partial-view ADM extension.
- `nystrom`: Nystrom extension for alternating diffusion maps.
- `ncca`: nonparametric canonical correlation analysis baseline.
- `apmc`: anchor partial multi-view clustering baseline.
- `kcca_full`: full-view kernel CCA.
- `kcca_impute`: KCCA with missing-view imputation.
- `fimvc_via`: optimization-based incomplete multi-view baseline.

## Data and Outputs

Some experiments create large intermediate files and result directories. The main output locations are:

- `figures/logo/` for rotating-image experiment results.
- `figures/fmri/` for fMRI task-classification results.
- `figures/bickley_*` for Bickley jet experiment results.
- `logo_data/` for generated rotating-image arrays.
- `fmri_data/` for fMRI matrices and cached distances.

The repository currently includes local data and generated outputs used during development. If running from a fresh clone or on another machine, verify that large data files are present before launching the corresponding experiment.

## Citation

If you find this work interesting or useful in your research, please cite our paper:

```bibtex
@article{admplus2026extracting,
  title   = {Extracting Common Components from Partially Observed Views Using Diffusion Geometry},
  journal = {Transactions on Machine Learning Research},
  author = {Weiss, Bar and Wu, Hau-Tieng and Talmon, Ronen},
  year    = {2026},
  note    = {TMLR}
}
```

## Notes

- Run notebooks and scripts from the repository root.
- Large parameter sweeps can take a long time and may require multiple CPU workers.
- Several scripts save CSV files, cached distance matrices, generated datasets, and figures in place.
- The fMRI optimization baselines are disabled by default in `fmri_main_server.py` because of their runtime.
