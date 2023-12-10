import math
import numpy as np
from .constants import AGENT_INIT, CREEPER_INIT, LOGGING


class InfoParser:
    def __init__(self):
        self.prev_arrow_id = 0
        self.prev_creeper_life = 20.0
        self.prev_damage_taken = 0

    @staticmethod
    def evalInfo(info):
        if info:
            return eval(info.replace('false', 'False').replace('true', 'True'))
        else:
            return {}

    def parseInfo(self, info):
        agentx, agentz, agentyaw, agentpitch = AGENT_INIT + (90, 0)
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
                    if self._creeper_took_damage(i):
                        reward += 500
                        print('+500 DAMAGE DEALT')
                        self.prev_creeper_life = i['life']
                    else:
                        self.prev_creeper_life = i['life']

                elif i['name'] == 'MalmoTutorialBot':
                    agentx = i['x']
                    agentz = i['z']
                    agentyaw = i['yaw']
                    agentpitch = i['pitch']
                elif i == info['entities'][-1] and i['name'] == 'Arrow':
                    if i['id'] != self.prev_arrow_id and i["y"] < 57.3:
                        last_arrow_dis = math.sqrt(
                            (math.pow((creeperx - i['x']), 2) + math.pow((creeperz - i['z']), 2)))
                        if LOGGING:
                            print("DISTANCE ", last_arrow_dis)
                        reward -= (last_arrow_dis ** 1.8) / 3.5 - 16
                        self.prev_arrow_id = i['id']
            # if 'IsAlive' in info:
            #     isAlive = info['IsAlive']

            if 'DamageTaken' in info:
                damageTaken = info['DamageTaken']
                if damageTaken > 0:
                    reward -= 100

        state = np.array([agentx - creeperx, agentz - creeperz, agentyaw, agentpitch], np.float64)

        return state, reward

    def damage_is_dealt(self, info):
        if 'entities' in info:
            for i in info['entities']:
                if i['name'] == 'Creeper':
                    return self._creeper_took_damage(i)
        return False

    def _creeper_took_damage(self, creeper):
        return creeper['life'] < self.prev_creeper_life and not math.isclose(creeper['life'], self.prev_creeper_life)

    def has_new_arrow(self, info_str):
        info = self.evalInfo(info_str)
        creeper_hit = self.damage_is_dealt(info)
        arrow_appeared = 'entities' in info and (i := info['entities'][-1])['name'] == 'Arrow' and i[
            'id'] != self.prev_arrow_id
        return creeper_hit or arrow_appeared

    def arrow_is_landed(self, info_str):
        info = self.evalInfo(info_str)
        creeper_hit = self.damage_is_dealt(info)
        arrow_landed = 'entities' in info and (i := info["entities"][-1])["name"] == "Arrow" and i["y"] < 57.3

        return creeper_hit or arrow_landed
