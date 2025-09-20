# ADM+

Here we provide code for the paper "Extracting Common Components from Partially Observed Views Using Diffusion Geometry, Bar Weiss, Hau-Tieng Wu and Ronen Talmon".

## Experiments 

We provide all the helper function required to run all the experiemnts in the **helper_functions** directory and the notebooks can be used to execute the experiments.

- **bickley_main.ipynb** - runs the bickley entire bickley experiment.
- **logo_main_scales.ipynb** - runs the rotating characters experiment with different common image sizes.
- **logo_main_discrepeny.ipynb** - runs the rotating characters experiment with distributional discrepency between the fully viewed set and the partially viewed set. This is a long run time so it is best to run it in parallel on some server, sending splits with different paramerters to different CPUs.

Code for the fMRI experiment is provided in **helper_functions/fmri_funcs.py** except for the code that generates the FCNs, for that refer to the paper that presents the methods we use to generate the FCN. We don't provide looping over samples as it should be run on parallel as well.
