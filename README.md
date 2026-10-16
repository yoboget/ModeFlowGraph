# Simple and Critical Iterative Denoising
### Discrete diffusion: Better, Faster, Simpler

Official code repository for the paper: [Simple and Critical Iterative Denoising](https://arxiv.org/html/2503.21592)

<p align="center">
<img src="misc/combined_graphs.gif" width="400"> <img src="misc/combined_image.gif" width="400">
</p>

<p align="center">
<img src="misc/validity.png" width="250"> <img src="misc/nspdk.png" width="250"> <img src="misc/fcd.png" width="250"> 
</p>

## Installation

We recommend to install the dependencies in the following order.

1. rdkit [rdkit.org/docs/Install.html](https://www.rdkit.org/docs/Install.html) 

2. pytorch [pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

3. pytorch-geometric [pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

4. dig [https://diveintographs.readthedocs.io/en/latest/intro/installation.html](https://diveintographs.readthedocs.io/en/latest/intro/installation.html)

5. Anything else in the [requirements.txt](doc/requirements.txt)


For the evaluation of generic graph generation tasks, run the following command to compile the ORCA program (see http://www.biolab.si/supp/orca/orca.html):

```sh
cd eval/orca
g++ -O2 -std=c++11 -o orca orca.cpp
```

## Training

Our model takes 4 main arguments:

```--dataset``` (str.) The dataset to train. Available: 'qm9', 'qm9H', 'zinc', 'planar', 'sbm', 'enzymes'.

```--work_type``` (str.) Select 'train' or 'sample'.

```--wandb``` Weight and bias 'init' argument ('on', 'off', 'no' for online, offline, disabled)

```--denoiser_dir``` (str.) Path to the directory of the trained denoiser (used with 'sample').


## Datasets

The datasets and splits are automatically downloaded during preprocessing in a folder called 'data'.

## Configuration

The configuration for all experiments are in the 'config' folder. 
The configuration files correspond to the configurations used in our experiments. 

## License

 This work is licensed under CC BY-NC-SA 4.0 
 https://creativecommons.org/licenses/by-nc-sa/4.0/
