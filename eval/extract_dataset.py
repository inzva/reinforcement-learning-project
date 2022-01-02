from env import Environment, RepeatAction
import torch
import torch.nn as nn
import pybullet as pb
import numpy as np
import time
from model import GaussianPolicy
import json
import pickle
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

env = Environment(gui=False, record=False, obstacles=False, visualize=False) 

env.obstacle_type = 1
env.wall_pos = 1
obs_shape = env._observationSpace()["state"].shape[0]
action_shape = env._actionSpace().shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = GaussianPolicy(
    num_inputs=obs_shape,
    num_actions=action_shape,
    hidden_dim=256,
    action_space=env._actionSpace(),
)
agent.load_state_dict(torch.load("sac_actor_Drone_9520", map_location=device))
agent.to(device)

done = False
action = None
samples = []
reward_prev = 0
count_ = 0
cum_reward = 0.0
max_episodes = 3000

while count_ < max_episodes:
    state = env.reset()["state"]
    env.createTarget()
    experience = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'terminals': []}
    done = False
    while not done:
        with torch.no_grad():
            action = agent.select_action(state, True)
            action_ = action
        action = {"0": np.array(action * env.MAX_RPM)}
        
        next_state, reward, done, _ = env.step(action)

        experience['observations'].append(state) 
        experience['actions'].append(action_)
        experience['rewards'].append(reward)
        experience['next_observations'].append(next_state)
        experience['terminals'].append(done) 

        state = next_state["state"]
    
    experience['observations'] = np.array(experience['observations'])
    experience['actions'] = np.array(experience['actions'])
    experience['rewards'] = np.array(experience['rewards'])
    experience['next_observations'] = np.array(experience['next_observations'])
    experience['terminals'] = np.array(experience['terminals'])
    
    samples.append(experience)
    count_ += 1
    print(count_)

        
""" dumped = json.dumps(samples, cls=NumpyEncoder)

with open('sample2.json', 'w') as f:
    json.dump(dumped, f) """


with open('drone-expert-v2.pickle', 'wb') as f:
    pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

