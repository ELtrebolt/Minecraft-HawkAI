import math
import numpy as np
from .constants import AGENT_INIT, CREEPER_INIT, LOGGING


class InfoParser:
    def __init__(self):
        self.prev_arrow_id = 0
        self.prev_damage_dealt = 0
        self.prev_damage_taken = 0

    @staticmethod
    def evalInfo(info):
        if info:
            return eval(info.replace('false', 'False').replace('true', 'True'))
        else:
            return {}

    def parseInfo(self, info):
        agentx, agenty, agentz, agentyaw, agentpitch = AGENT_INIT
        creeperx, creepery, creeperz, creeperyaw = CREEPER_INIT
        reward = 0

        if info:
            if 'LineOfSight' in info and info['LineOfSight']['type'] == 'Creeper':
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
                    if i['id'] != self.prev_arrow_id and i["y"] < 57.3:
                        last_arrow_dis = math.sqrt((math.pow((creeperx - i['x']), 2) + math.pow((creeperz - i['z']), 2)))
                        if LOGGING:
                            print("DISTANCE ", last_arrow_dis)
                        reward -= (last_arrow_dis ** 2) / 3 - 3
                        self.prev_arrow_id = i['id']
            # if 'IsAlive' in info:
            #     isAlive = info['IsAlive']
            # if 'DamageDealt' in info:
            #     damageDealt = info['DamageDealt']
            if 'DamageTaken' in info:
                damageTaken = info['DamageTaken']
                if damageTaken > 0:
                    reward -= 100

        state = np.array([agentx - creeperx, agenty - creepery, agentz - creeperz, agentyaw, agentpitch], np.float64)

        return state, reward
