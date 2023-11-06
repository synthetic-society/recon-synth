# A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data

This repository contains the source code for the paper _A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data_ by M.S.M.S. Annamalai, A. Gadotti, L. Rocher, presented August 2024 at USENIX Security 2024 [link](https://arxiv.org/abs/2301.10053).

## Install
Dependencies are managed by `mamba/conda`. The required dependencies can be installed using the command `[conda/mamba] env create -f env.yml` and then run `[conda/mamba] activate recon_synth`.

A license to Gurobi is also necessary to run the code, which for academics can be gotten for free [here](https://www.gurobi.com/features/academic-named-user-license/). The license can be subsequently installed using the command `grbgetkey <KEY>`.

## Usage
The entire experiment pipeline to generate the results we used for the paper can be run using the following series of commands. Please note that you can change the `REPS`, `N_PROCS`, and `DATA_DIR` variables in these scripts to change the number of repetitions of the privacy game, number of processors to use during parallelization, and the output directory of the results respectively.

```bash
$ cd priv_game
$ scripts/prep_exps.sh
$ scripts/run_exps.sh
$ scripts/prep_extra_exps.sh
$ scripts/run_extra_exps.sh
$ scripts/time_attack.sh
$ scripts/log_memory.sh
```

After running all the experiments, results can be visualized using the `plot_results.ipynb` notebook.

## Available SDG Models
Here is a table of available SDG models and their descriptions.  
| SDG Model | Description |
| --- | --- |
| NonPrivate | Sample directly from target dataset with replacement |
| CTGAN | Generative adversarial network from MIT SDV library |
| IndHist | Independently sample attributes from 1D histogram |
| BayNet_3parents | Bayesian network (with hyperparameter: 3 parents) |
| PrivBayes_3parents_1eps | PrivBayes model (with hyperparameters: 3 parents, $\varepsilon = 1$) |
| RAP_2Kiters | Relaxed Adaptive Projection algorithm (with hyperparameter: 2000 iterations)
| RAP_2Kiters_1eps | RAP algorithm (with hyperparameters: 2000 iterations, $\varepsilon = 1$)

## Available Datasets
We provide 2 datasets in this repo `acs` and `fire`.

## Acknowledgements

Code in the `generative_models/rap_src` folder is cloned and slightly modified (to suit our import system) from [dp-query-release](https://github.com/terranceliu/dp-query-release).
