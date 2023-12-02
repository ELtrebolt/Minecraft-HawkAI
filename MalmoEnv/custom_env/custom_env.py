import malmoenv
import gym
import numpy as np
import math

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
            # elif i == info['entities'][-1] and i['name'] == 'Arrow':
            #     if abs(i['motionX']) < 0.3 and abs(i['motionY']) < 0.3 and abs(i['motionZ']) < 0.3:
            #         last_arrow_dis = math.sqrt( ( math.pow((creeperx - i['x']),2) + math.pow((creepery - i['y']),2) ) )

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
                                low=np.float64('-inf'), high=np.float64('inf'),
                                shape=(1, 5), dtype=np.float64)


class CustomEnv(malmoenv.core.Env):
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)

        self.observation_space = CustomObservationSpace()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info_dict = evalInfo(info)
        new_obs, reward_delta = parseInfo(info_dict)
        if abs(reward + reward_delta) > 5:
            print(new_obs, reward + reward_delta)

        return new_obs, reward + reward_delta, False, done, info_dict

    def reset(self, seed=None, options=None):
        super().reset()

        return np.array(AGENT_INIT, dtype=np.float64), {}
