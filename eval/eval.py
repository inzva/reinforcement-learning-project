from env import Environment, RepeatAction
import torch
import torch.nn as nn
import pybullet as pb
import numpy as np
import time
from model import GaussianPolicy
import time


env = Environment(gui=True, record=False, obstacles=False, visualize=False)

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


targets = []
pid_count = 0
control_count = 0

done = False
pid = False
counter = 0
action = None
idx = 0

# time.sleep(10)

while True:
    state = env.reset()["state"]
    env.createTarget()
    while not done:
        with torch.no_grad():
            action = agent.select_action(state, True)
        action = {"0": np.array(action * env.MAX_RPM)}

        next_state, reward, done, _ = env.step(action)
        state = next_state["state"]
        pb.addUserDebugText(
            text="Distance: " + str(np.linalg.norm(env.target_point - state[0:3])),
            textPosition=[state[0], state[1], state[2] + 0.1],
            textColorRGB=[0, 0, 1],
            textSize=1.5,
            lifeTime=0.1,
        )
        if np.linalg.norm(env.target_point - state[0:3]) < 0.25:
            env.removeTargetCircle()
            env.createTarget()
        env.render()
        time.sleep(0.05)
