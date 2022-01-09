import gym
import numpy as np
from numpy.core.fromnumeric import var
import torch
import wandb
import time
import argparse
import pickle
import random
import sys
import os
import pybullet as pb

from decision_transformer.models.decision_transformer import DecisionTransformer

from decision_transformer.envs.env import Environment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="drone")
    parser.add_argument(
        "--dataset", type=str, default="expert"
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument(
        "--mode", type=str, default="normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--model_type", type=str, default="dt"
    )  # dt for decision transformer, bc for behavior cloning
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5 * 1e-5)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_eval_episodes", type=int, default=50)
    parser.add_argument("--max_iters", type=int, default=350)
    parser.add_argument("--num_steps_per_iter", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--log_to_wandb", "-w", type=bool, default=False)

    variant = vars(parser.parse_args())

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if variant["env"] == "drone":
        env = Environment(gui=True, record=False, obstacles=False, visualize=True)
    elif variant["env"] == "halfcheetah":
        env = gym.make("HalfCheetah-v3")

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    K = variant["K"]
    batch_size = variant["batch_size"]
    num_eval_episodes = variant["num_eval_episodes"]

    if variant["env"] == "drone":
        max_ep_len = 302
    elif variant["env"] == "halfcheetah":
        max_ep_len = 1000

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant["embed_dim"],
        n_layer=variant["n_layer"],
        n_head=variant["n_head"],
        n_inner=4 * variant["embed_dim"],
        activation_function=variant["activation_function"],
        n_positions=1024,
        resid_pdrop=variant["dropout"],
        attn_pdrop=variant["dropout"],
    )
    if variant["env"] == "drone":
        model.load_state_dict(
            torch.load(
                "./decision_transformer/evaluation/dt-drone-checkpoint.pth",
                map_location=torch.device(device),
            )
        )
    elif variant["env"] == "halfcheetah":
        model.load_state_dict(
            torch.load(
                "./decision_transformer/evaluation/dt-halfcheetah-checkpoint.pth",
                map_location=torch.device(device),
            )
        )
    model = model.to(device)
    model.eval()

    if variant["env"] == "drone":
        state_mean = torch.from_numpy(
            np.array(
                [
                    -0.0888972551,
                    0.159591115,
                    1.70317850,
                    0.0292241870,
                    0.0335000330,
                    0.0541813925,
                    -0.00252566758,
                    0.000500925758,
                    0.0507742928,
                    0.986369715,
                    -0.0195447855,
                    0.0328154033,
                    -0.0590810013,
                    -0.0261396909,
                    0.00398025944,
                    0.0163620272,
                ]
            )
        ).to(device=device)
        state_std = torch.from_numpy(
            np.array(
                [
                    1.68057057,
                    1.70206415,
                    0.8951205,
                    0.53535807,
                    0.52892231,
                    0.28193956,
                    0.08667433,
                    0.08430454,
                    0.09542234,
                    0.0276694,
                    0.69341357,
                    0.72668841,
                    0.3711086,
                    1.19910975,
                    1.16446809,
                    0.61872726,
                ]
            )
        ).to(device=device)
    elif variant["env"] == "halfcheetah":
        state_mean = torch.from_numpy(
            np.array(
                [
                    -0.04489148,
                    0.03232588,
                    0.06034835,
                    -0.17081226,
                    -0.19480659,
                    -0.05751596,
                    0.09701628,
                    0.03239211,
                    11.047426,
                    -0.07997331,
                    -0.32363534,
                    0.36297753,
                    0.42322603,
                    0.40836546,
                    1.1085187,
                    -0.4874403,
                    -0.0737481,
                ]
            )
        ).to(device=device)
        state_std = torch.from_numpy(
            np.array(
                [
                    0.04002118,
                    0.4107858,
                    0.54217845,
                    0.41522816,
                    0.23796624,
                    0.62036866,
                    0.30100912,
                    0.21737163,
                    2.2105937,
                    0.572586,
                    1.7255033,
                    11.844218,
                    12.06324,
                    7.0495934,
                    13.499867,
                    7.195647,
                    5.0264325,
                ]
            )
        ).to(device=device)

    done = False
    target_return = 0.0

    while True:
        done = False

        if variant["env"] == "drone":
            env.createTarget()
        state = env.reset()

        states = (
            torch.from_numpy(state)
            .reshape(1, state_dim)
            .to(device=device, dtype=torch.float32)
        )
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        ep_return = target_return
        target_return = torch.tensor(
            ep_return, device=device, dtype=torch.float32
        ).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        while not done:

            # add padding
            actions = torch.cat(
                [actions, torch.zeros((1, act_dim), device=device)], dim=0
            )
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )

            actions[-1] = action
            action = action.detach().cpu().numpy()

            if variant["env"] == "drone":
                action = {"0": np.array(action * env.MAX_RPM)}

            state, reward, done, _ = env.step(action)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            if variant["env"] == "drone":
                pb.addUserDebugText(
                    text="Distance: "
                    + str(np.linalg.norm(env.target_point - state[0:3]))[:7],
                    textPosition=[state[0], state[1], state[2] + 0.1],
                    textColorRGB=[0, 0, 1],
                    textSize=1.5,
                    lifeTime=0.1,
                )

            env.render()
            time.sleep(0.1)
