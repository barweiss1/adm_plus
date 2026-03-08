# ADM+

Here we provide code for the paper "Extracting Common Components from Partially Observed Views Using Diffusion Geometry".

## Experiments 

We provide all the helper function required to run all the experiemnts in the **helper_functions** directory and the notebooks can be used to execute the experiments.

- `bickley_main.ipynb` - runs the bickley entire bickley experiment.
- `logo_main_scales.ipynb` - runs the rotating characters experiment with different common image sizes.
- `logo_main_discrepeny.ipynb` - runs the rotating characters experiment with distributional discrepency between the fully viewed set and the partially viewed set. This is a long run time so it is best to run it in parallel on some server, sending splits with different paramerters to different CPUs `logo_main_server.py` implements this.
- `fmri_main_server.py` - implements the fMRI experiment to run in parallel on multiple CPUs.
- `fMRI_plots.ipynb` - post processing fMRI results.
- `logo_plots.ipynb` - post processing rotating images results.

## Appendix Figures
- `diffusion_simulation.ipynb` - Notebook creating diffusion illustrations.
- `logo_two_missing.ipynb` - Runs rotating character experiment with two missing views.

## Animations 
The `animations` directory contains several illustrations for the Bickley jet experiment.
