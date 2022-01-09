
# OpenAI Gym

This is the modified version of [Decision Transformer repository](https://github.com/kzl/decision-transformer). For installation please check installation sub-topic below. 

The difference of this repository compared to the original is there is an additional dataset created from [Drone environment](https://github.com/utiasDSL/gym-pybullet-drones). The training using drone dataset can be initialized with the following:
```
    python experiment.py --env drone --dataset expert
```
Before running please install the Drone Environment in order to be able to run the simulation. Additionally, [install dataset](https://drive.google.com/file/d/1UjY-G1cZcpEP6ibJ-282XjJzvC9NO5N2/view?usp=sharing) from Google Drive and move the pickle file to ```./decision_transformer/data/```.

For rendered evaluation of the drone environment run the following command:

```
python evaluation.py --env drone
```

<hr />

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

## Example usage

Experiments can be reproduced with the following:

```
python experiment.py --env hopper --dataset medium --model_type dt
```

Adding `-w True` will log results to Weights and Biases.
