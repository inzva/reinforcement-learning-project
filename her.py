import numpy as np


def sample_achieved_goal(trans_idx, episode_transitions):
    selected_idx = np.random.choice(np.arange(trans_idx + 1, len(episode_transitions)))
    selected_goal = episode_transitions[selected_idx][0][0:3]

    return selected_goal


def sample_achieved_goals(trans_idx, episode_transitions, k):
    return [sample_achieved_goal(trans_idx, episode_transitions) for _ in range(k)]


def calculate_reward(pos, diff):
    reward = 0
    dist = np.linalg.norm(diff) ** 2
    reward = -np.log(2.5*dist+1)

    if pos[2] < 0.1:
            reward -= 1

    return reward
