import pybullet as pb
import gym
import numpy as np
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import (
    DroneModel,
    Physics,
    ImageType,
    BaseAviary,
)
import time

class Environment(BaseAviary):
    def __init__(
        self,
        drone_model=DroneModel.CF2X,
        num_drones: int = 1,
        initial_xyzs=np.array([[0, 0, 1]]),
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        freq: int = 60,
        aggregate_phy_steps: int = 1,
        gui=False,
        record=False,
        obstacles=False,
        user_debug_gui=False,
        visualize=False,
    ):
        self.max_XYZ = [5.0, 5.0, 5.0]
        self.min_XYZ = [-5.0, -5.0, 0.1]
        self.obstacle_type = 0
        self.EPISODE_LEN_SEC = 5
        self.maxEpisodeSteps = self.EPISODE_LEN_SEC * freq
        self.target_point = np.array([0, 0, 3])
        self.columnIDs = []
        self.margin = 0.3
        self.VISUALIZE = visualize
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obstacles=obstacles,
            user_debug_gui=user_debug_gui,
            vision_attributes=True,
        )

    def _actionSpace(self):

        action_lower = np.array([-1.0, -1.0, -1.0, -1.0])
        action_upper = np.array([1.0, 1.0, 1.0, 1.0])

        return spaces.Box(low=action_lower, high=action_upper, dtype=np.float32)

    def _observationSpace(self):

        #### Observation vector ### X      Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array(
            [
                ## position
                -np.inf,
                -np.inf,
                -np.inf,
                ## diff
                -np.inf,
                -np.inf,
                -np.inf,
                ## Quaternions
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                ## Linear Vel
                -np.inf,
                -np.inf,
                -np.inf,
                ## Angular Vel
                -np.inf,
                -np.inf,
                -np.inf,
            ]
        )
        obs_upper_bound = np.array(
            [
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                1.0,
                1.0,
                1.0,
                1.0,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            ]
        )
        return spaces.Dict(
            {
                "state": spaces.Box(
                    low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32
                )
            }
        )

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        target = self.target_point

        pos = state[0:3]
        diff = [pos[0] - target[0], pos[1] - target[1], pos[2] - target[2]]
        quaternions = state[3:7]  # 4
        lin_vel = state[10:13]  # 3
        ang_vel = state[13:16]  # 3

        state_combined = [
            pos,
            diff,
            quaternions,
            lin_vel,
            ang_vel,
        ]

        state_new = []
        for i in state_combined:
            for j in i:
                state_new.append(j)

        obs = {
            "state": state_new,
        }

        return obs

    def _preprocessAction(self, action):

        clipped_action = np.zeros((1, 4))

        for k, v in action.items():
            clipped_action[int(k), :] = np.clip(np.array(v), 0, self.MAX_RPM)
        return clipped_action

    def _computeReward(self):

        state = self._getDroneStateVector(0)
        target = self.target_point

        pos = state[0:3]
        diff = [pos[0] - target[0], pos[1] - target[1], pos[2] - target[2]]

        reward = 0
        dist = np.linalg.norm(diff) ** 2
        reward = -np.log(2.5*dist+1)

        if pos[2] < 0.1:
            reward -= 1
            
        return reward

    def _computeDone(self):
        z_pos = self._getDroneStateVector(0)[2]
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _computeInfo(self):
        return {}

    def _addObstacles(self):
        self.createTarget()
        print("Target: ", self.target_point)

    def createTarget(self):
        drone_pos = self._getDroneStateVector(0)[0:3]

        new_x = np.clip(drone_pos[0] + np.random.uniform(-2, 2), -5, 5)
        new_y = np.clip(drone_pos[1] + np.random.uniform(-2, 2), -5, 5)
        new_z = np.clip(drone_pos[2] + np.random.uniform(-0.5, 3), 0.5, 5)
        self.target_point = np.array([new_x, new_y, new_z])
