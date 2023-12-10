import time

import malmoenv
import gym
import numpy as np
import math

from malmoenv.core import ActionSpace

from .info_parser import InfoParser
from .constants import *


class CustomObservationSpace(gym.spaces.Box):
    """Space for our custom observations
    """

    def __init__(self):
        # deltax, deltay, deltaz, agentyaw, agentpitch
        gym.spaces.Box.__init__(self,
                                low=np.array([-100, -100, -100, -180, -90]), high=np.array([100, 100, 100, 180, 90]),
                                shape=(5,), dtype=np.float64)


class CustomEnv(malmoenv.core.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info_parser = InfoParser()

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.observation_space = CustomObservationSpace()
        self.action_space = ActionSpace(ACTIONS)

    def _execute_action(self, action: int):

        command = self.action_space[action]
        if " " in command:
            command, val = command.split()
        obs, reward, done, info = AGENT_INIT, 0, False, {}

        if command == "turn":
            self.agent_yaw += YAW_DELTA * int(val)
            obs, reward, done, info = super().step(f"setYaw {self.agent_yaw}")
        elif command == "pitch":
            self.agent_pitch += PITCH_DELTA * int(val)
            obs, reward, done, info = super().step(f"setPitch {self.agent_pitch}")
        elif command == "use":
            obs, reward, done, info = super().step("use 1")
            if not done:
                time.sleep(BOW_SLEEP_TIME)  # Charging the bow for this time duration
                obs, reward, done, info = super().step("use 0")  # Release the bow
            if not done:
                # wait for arrow entity to appear
                check_count = 0
                while not done and check_count < 70 and not self.info_parser.has_new_arrow(info):
                    if LOGGING and ARROW_LOGGING:
                        print(f'arrow appearing... (check {check_count})')
                    obs, reward, done, info = super().step("turn 0")
                    check_count += 1

                if check_count < 50:
                    # wait for arrow entity to land
                    while not done and not self.info_parser.arrow_is_landed(info):
                        if LOGGING and ARROW_LOGGING:
                            print('landing arrow... ')
                        obs, reward, done, info = super().step("turn 0")
                        while not done and not info:
                            obs, reward, done, info = super().step("turn 0")

                reward -= 10  # penalty for firing arrow
        elif command == "wait":
            obs, reward, done, info = super().step("turn 0")

        return obs, reward, done, info

    def step(self, action):
        obs, reward, done, info = self._execute_action(action)

        info_dict = self.info_parser.evalInfo(info)
        new_obs, reward_delta = self.info_parser.parseInfo(info_dict)
        if reward is None and LOGGING:
            print("NONE", action, obs, reward, done, info)
        reward = reward or 0
        if LOGGING and abs(reward + reward_delta) > 5:
            print(new_obs, reward + reward_delta)

        return new_obs, reward + reward_delta, False, done, info_dict

    def reset(self, seed=None, options=None):
        super().reset()
        return np.array(AGENT_INIT+(self.agent_yaw, self.agent_pitch), np.float64), {}
