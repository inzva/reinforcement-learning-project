import argparse
import datetime
import gym
import numpy as np
import copy
import itertools
import torch
from sac import SAC
from env import Environment
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from her import sample_achieved_goals, calculate_reward

parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Args")
parser.add_argument(
    "--env-name",
    default="Drone Env",
    help="Mujoco Gym environment (default: HalfCheetah-v2)",
)
parser.add_argument(
    "--policy",
    default="Gaussian",
    help="Policy Type: Gaussian | Deterministic (default: Gaussian)",
)
parser.add_argument(
    "--eval",
    type=bool,
    default=True,
    help="Evaluates a policy a policy every 10 episode (default: True)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor for reward (default: 0.99)",
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.005,
    metavar="G",
    help="target smoothing coefficient(τ) (default: 0.005)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    metavar="G",
    help="learning rate (default: 0.0003)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.2,
    metavar="G",
    help="Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)",
)
parser.add_argument(
    "--automatic_entropy_tuning",
    type=bool,
    default=True,
    metavar="G",
    help="Automaically adjust α (default: False)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=31316969,
    metavar="N",
    help="random seed (default: 123456)",
)
parser.add_argument(
    "--batch_size", type=int, default=512, metavar="N", help="batch size (default: 512)"
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=100000001,
    metavar="N",
    help="maximum number of steps (default: 1000000)",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    metavar="N",
    help="hidden size (default: 256)",
)
parser.add_argument(
    "--updates_per_step",
    type=int,
    default=5,
    metavar="N",
    help="model updates per simulator step (default: 10)",
)
parser.add_argument(
    "--start_steps",
    type=int,
    default=1000,
    metavar="N",
    help="Steps sampling random actions (default: 10000)",
)
parser.add_argument(
    "--target_update_interval",
    type=int,
    default=1,
    metavar="N",
    help="Value target update per no. of updates per step (default: 1)",
)
parser.add_argument(
    "--replay_size",
    type=int,
    default=10000000,
    metavar="N",
    help="size of replay buffer (default: 10000000)",
)
parser.add_argument("--cuda", action="store_true", help="run on CUDA (default: False)")
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = Environment(gui=False, record=False, obstacles=False)
agent = SAC(env._observationSpace()["state"].shape[0], env._actionSpace(), args)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# Agent

# Tensorboard
writer = SummaryWriter(
    "runs/{}_SAC_{}_{}_{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        args.env_name,
        args.policy,
        "autotune" if args.automatic_entropy_tuning else "",
    )
)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# HER
episode_transitions = []

# Training Loop
total_numsteps = 0
updates = 0
i = 0
best_reward = -np.inf
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False

    env.createTarget()
    state = env.reset()["state"]

    action = {"0": np.array([0.0, 0.0, 0.0, 0.0])}
    while not done:

        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                (
                    critic_1_loss,
                    critic_2_loss,
                    policy_loss,
                    ent_loss,
                    alpha,
                ) = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar("loss/critic_1", critic_1_loss, updates)
                writer.add_scalar("loss/critic_2", critic_2_loss, updates)
                writer.add_scalar("loss/policy", policy_loss, updates)
                writer.add_scalar("loss/entropy_loss", ent_loss, updates)
                writer.add_scalar("entropy_temprature/alpha", alpha, updates)
                updates += 1

        action2 = {"0": np.array((action+1)/2 * env.MAX_RPM)}
        next_state, reward, done, _ = env.step(action2)  # Step
        next_state = next_state["state"]
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.maxEpisodeSteps else float(not done)

        memory.push(
            state, action, reward, next_state, mask
        )  # Append transition to memory

        episode_transitions.append((state, action, reward, next_state, mask))

        state = next_state

    ## create hindsight experience replay
    for trans_idx, transition in enumerate(episode_transitions):

        if trans_idx == len(episode_transitions) - 1:
            break

        sampled_goals = sample_achieved_goals(trans_idx, episode_transitions, 4)

        for goal in sampled_goals:

            state, action, reward, next_state, done = copy.deepcopy(transition)
            state[3:6] = [state[0] - goal[0], state[1] - goal[1], state[2] - goal[2]]
            next_state[3:6] = [next_state[0] - goal[0], next_state[1] - goal[1], next_state[2] - goal[2]]
            new_reward = calculate_reward(pos=state[0:3], diff=state[3:6])

            memory.push(state, action, new_reward, next_state, done)
    episode_transitions = []

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar("reward/train", episode_reward, i_episode)
    print(
        "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
            i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
        )
    )

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.0
        episodes = 30
        for _ in range(episodes):
            env.createTarget()
            state = env.reset()["state"]
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                action2 = {"0": np.array((action+1)/2 * env.MAX_RPM)}

                next_state, reward, done, _ = env.step(action2)
                next_state = next_state["state"]
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        if best_reward < avg_reward:
            agent.save_model("Drone", suffix=str(i_episode))
            best_reward = max(best_reward, avg_reward)

        writer.add_scalar("avg_reward/test", avg_reward, i_episode)

        print("----------------------------------------")
        print(
            "Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 3))
        )
        print("----------------------------------------")

env.close()
