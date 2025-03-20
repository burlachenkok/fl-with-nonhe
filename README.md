This is a repository with  experiments for the paper **Federated Learning is Better with Non-Homomorphic Encryption**

This repository includes source code and guidelines for reproducing experiments for this paper.

## Prerequisites

The experiments have been constructed via modifying FL_PyTorch:

* [https://arxiv.org/abs/2202.03099](https://arxiv.org/abs/2202.03099)
* [https://github.com/burlachenkok/flpytorch](https://github.com/burlachenkok/flpytorch)

This simulator is constructed based on the PyTorch computation framework. The first step is preparing the environment. 

If you have installed [conda](https://docs.conda.io/en/latest/) environment and package manager then you should perform only the following steps for preparing the environment.

```
conda create -n fl python=3.9.1 -y
conda activate fl
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

For experiments in supplementary part we have [https://graphviz.org/download/](graphviz) with verion: `dot - graphviz version 8.0.5 (20230430.1635)`

Our modification of the simulator is located in `./fl_pytorch`. Use this version that we're providing instead of the Open Source version.

Also, your OS should have installed a BASH interpreter or its equivalent.

## Place with Execution Command Lines for Main Paper Part

Change the working directory to `"./fl_pytorch"` and execute `conda activate fl`. Next, the sequence of scripts to execute:


* [experiment_for_sec_5_1__case_1_case_2.sh](fl_pytorch/experiment_for_sec_5_1__case_1_case_2.sh) contains experiments for section 5.1 Synthetic Experiments. Case 1 / Case 2.

* [experiment_for_sec_5_1__case_3.sh](fl_pytorch/experiment_for_sec_5_1__case_3.sh) contains experiments for section 5.1 Synthetic Experiments. Case 3.

* [experiment_for_sec_5_1__exp_4.sh](fl_pytorch/experiment_for_sec_5_1__exp_4.sh) contains experiments for section 5.1 Synthetic Experiments (comparison of DCGD/PermK/AES and GD/CKKS)

* [experiment_for_sec_5_1__exp_5.sh](fl_pytorch/experiment_for_sec_5_1__exp_5.sh) contains experiments for section H.1 Synthetic Experiments (Exploring Problem Dimension)

* [experiment_for_sec_5_2.sh](fl_pytorch/experiment_for_sec_5_2.sh) contains experiments for section 5.2 Image Classification Application.


## Tracking results online

If you want to use [WandB](https://wandb.ai/settings) online tool to track the progress of the numerical experiments please specify:
* `--wandb-key "xxxxxxxxxxx" ` with a key from your wandb profile: [https://wandb.ai/settings](https://wandb.ai/settings
* `--wandb-project-name "vvvvvvvvvv"` with a project name that you're planning to use.
You should replace `--wandb-project-name "vvvvvvvvvv"` with a project name that you're planning to use or leave the default name. Both of these keys should be replaced manually if you're interested in WandB support.

## Visualization of the Results

The result in binary files can be loaded into the simulator `fl_pytorch\fl_pytorch\gui\start.py`. After this plots can be visualized in the *Analysis* tab. 

Recommendations on how to achieve this are available in TUTORIAL for this simulator.

For the purpose of publication plots in the paper, we have used line size 4, and font size 37.

## Source Code for Adaptable Compute and Communication for DCGD/PermK

The source code for this experiment is located here [digraph.py](fl_pytorch/simulation_of_dependence_chain/digraph.py) and contains experiments for the section "Compute and Communication Overlap for DCGD/PermK" (H.2).

Change the working directory to `"./fl_pytorch/simulation_of_dependence_chain"` and execute from a command line `conda activate fl`. The sequence of steps to execute:

1. `python digraph.py`

2. Close or save all plots that you will see.

3. Launch `generate_plans.sh` to convert from dot text description to `PDF` that contains detailed plans of execution.
