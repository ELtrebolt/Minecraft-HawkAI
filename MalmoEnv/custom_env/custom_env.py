import time

import malmoenv
import gym
import numpy as np
import math

AGENT_INIT = (15, 57, 0, 90, 0)
CREEPER_INIT = (8, 57, 1, 270)
SLEEP_TIME = 0.075


def evalInfo(info):
    if info:
        return eval(info.replace('false', 'False').replace('true', 'True'))
    else:
        return {}

def parseInfo(info, prev_damageDealt=0, prev_damageTaken=0, prev_arrow=0):
    agentx, agenty, agentz, agentyaw, agentpitch = AGENT_INIT
    last_arrow_dis = float('inf')
    creeperx, creepery, creeperz, creeperyaw = CREEPER_INIT
    reward = 0

    if info:
        if 'LineOfSight' in info and info['LineOfSight']['type'] == 'Creeper':
            print('+100 IN LINE OF SIGHT')
            reward += 100

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
                if i['id'] != prev_arrow and abs(i['motionX']) < 0.3 and abs(i['motionY']) < 0.3 and abs(i['motionZ']) < 0.3:
                    last_arrow_dis = math.sqrt( ( math.pow((creeperx - i['x']),2) + math.pow((creeperz - i['z']),2) ) )
                    print("NEW ARROW DISTANCE ", last_arrow_dis, i['id'], prev_arrow)
                    reward -= last_arrow_dis * .1
                    prev_arrow = i['id']
                    
        if 'DamageDealt' in info:
            damageDealt = info['DamageDealt']
            if damageDealt != prev_damageDealt:
                reward += 1000
                print('+1000 DAMAGE DEALT')
                prev_damageDealt = damageDealt

        if not creeperAlive:
            reward += 10000

        if 'DamageTaken' in info:
            damageTaken = info['DamageTaken']
            if damageTaken != prev_damageTaken:
                print('-10 DAMAGE TAKEN')
                reward -= 100
                prev_damageTaken = damageTaken

    state = [agentx-creeperx, agenty-creepery, agentz-creeperz, agentyaw, agentpitch]

    return np.array(state, dtype=np.float64), reward, [prev_damageDealt, prev_damageTaken, prev_arrow]


class CustomObservationSpace(gym.spaces.Box):
    """Space for our custom observations
    """

    def __init__(self):
        # deltax, deltay, deltaz, agentyaw, agentpitch, last_arrow_dis
        LOWS = np.array([-100, -100, -100, -180, -90], dtype=np.float64)
        HIGHS = np.array([100, 100, 100, 180, 90], dtype=np.float64)
        gym.spaces.Box.__init__(self,
                                low=LOWS, high=HIGHS, 
                                # shape=(1,7)
                                dtype=np.float64)
                                # shape will be filled automatically if LOWS and HIGHS is np array


class CustomEnv(malmoenv.core.Env):
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        # action_space already defined in env
        self.observation_space = CustomObservationSpace()
        self.prev = [0, 0, 0]

    def step(self, action):
        obs, reward, done, info = super().step(action)
        action_str, val = self.action_space[action].split()
        if not done:
            # Only Turn a set amount
            if self.action_space[action].startswith("turn"):
                time.sleep(SLEEP_TIME)
                obs, reward, done, info = super().step(self.action_space.actions.index("turn 0"))
            # Only Turn a set amount
            if self.action_space[action].startswith("pitch"):
                time.sleep(SLEEP_TIME)
                obs, reward, done, info = super().step(self.action_space.actions.index("pitch 0"))
            
            if self.action_space[action].startswith("use 1"):
                print('-5 ARROW FIRED')
                reward -= 5
            
            # if self.action_space[action] == "use 0":
            #     print("sleeping")
            #     time.sleep(1)
        info_dict = evalInfo(info)
        
        new_obs, reward_delta, new_previous = parseInfo(info_dict, self.prev[0], self.prev[1], self.prev[2])
        self.prev = new_previous
        if reward is None:
            print("NONE", action, obs, reward, done, info)
        reward = reward or 0

        if abs(reward + reward_delta) > 5:
            print(new_obs, reward + reward_delta)

        return new_obs, reward+reward_delta, False, done, info_dict

    def reset(self, seed=None, options=None):
        super().reset()

        return np.array(AGENT_INIT, dtype=np.float64), {}