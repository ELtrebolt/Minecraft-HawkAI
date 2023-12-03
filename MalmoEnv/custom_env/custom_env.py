import time

import malmoenv
import gym
import numpy as np
import math

from malmoenv.core import ActionSpace

AGENT_INIT = (15, 57, 0, 90, 0)
CREEPER_INIT = (15, 57, -6, 0)


def evalInfo(info):
    if info:
        return eval(info.replace('false', 'False').replace('true', 'True'))
    else:
        return {}

def parseInfo(info):
    agentx, agenty, agentz, agentyaw, agentpitch = AGENT_INIT
    creeperx, creepery, creeperz, creeperyaw = CREEPER_INIT
    reward = 0

    if info:
        if 'LineOfSight' in info and info['LineOfSight']['type'] == 'Creeper':
            reward += 10

        creeperAlive = False
        for i in info['entities']:
            if i['name'] == 'Creeper':
                creeperAlive = True
                creeperx = i['x']
                creepery = i['y']
                creeperz = i['z']
                creeperyaw = i['yaw']
            elif i['name'] == 'MalmoTutorialBot':
                agentx = i['x']
                agenty = i['y']
                agentz = i['z']
                agentyaw = i['yaw']
                agentpitch = i['pitch']
            elif i == info['entities'][-1] and i['name'] == 'Arrow':
                if abs(i['motionX']) < 0.3 and abs(i['motionY']) < 0.3 and abs(i['motionZ']) < 0.3:
                    last_arrow_dis = math.sqrt( ( math.pow((creeperx - i['x']),2) + math.pow((creepery - i['y']),2) ) )
                    print("DISTANCE ", last_arrow_dis)
                    reward -= last_arrow_dis
        # if 'IsAlive' in info:
        #     isAlive = info['IsAlive']
        # if 'DamageDealt' in info:
        #     damageDealt = info['DamageDealt']
        if 'DamageTaken' in info:
            damageTaken = info['DamageTaken']
            if damageTaken > 0:
                reward -= 100

    state = [agentx - creeperx, agenty - creepery, agentz - creeperz, agentyaw, agentpitch]

    return state, reward


class CustomObservationSpace(gym.spaces.Box):
    """Space for our custom observations
    """

    def __init__(self):
        # deltax, deltay, deltaz, agentyaw, agentpitch
        gym.spaces.Box.__init__(self,
                                -1000.0, high=1000.0,
                                shape=(5,), dtype=np.float64)


class CustomEnv(malmoenv.core.Env):
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.observation_space = CustomObservationSpace()
        actions = ["turn 1", "turn -1", "pitch 1", "pitch -1", "use 1", "wait"]
        self.action_space = ActionSpace(actions)

    def _execute_action(self, action: int):
        SLEEP_TIME = 0.15
        command = self.action_space[action]
        if " " in command:
            command, val = command.split()
        obs, reward, done, info = AGENT_INIT, 0, False, {}

        if command == "turn":
            obs, reward, done, info = super().step(action)
            if not done:
                time.sleep(SLEEP_TIME)
                obs, reward, done, info = super().step("turn 0")
        elif command == "pitch":
            obs, reward, done, info = super().step(action)
            if not done:
                time.sleep(SLEEP_TIME)
                obs, reward, done, info = super().step("pitch 0")
        elif command == "use":
            obs, reward, done, info = super().step("use 1")
            if not done:
                time.sleep(0.35)
                obs, reward, done, info = super().step("use 0")
        elif command == "wait":
            obs, reward, done, info = super().step("turn 0")

        return obs, reward, done, info

    def step(self, action):
        obs, reward, done, info = self._execute_action(action)

        info_dict = evalInfo(info)
        new_obs, reward_delta = parseInfo(info_dict)
        if reward is None:
            print("NONE", action, obs, reward, done, info)
        reward = reward or 0
        if abs(reward + reward_delta) > 5:
            print(new_obs, reward + reward_delta)

        return new_obs, reward + reward_delta, False, done, info_dict

    def reset(self, seed=None, options=None):
        super().reset()

        return AGENT_INIT, {}
