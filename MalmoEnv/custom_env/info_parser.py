import math
import numpy as np
from .constants import AGENT_XYZ_INIT, CREEPER_INIT, LOGGING


def calculate_creeper_yaw_pitch(creep_x, creep_y, creep_z):
    agent_x, agent_y, agent_z = AGENT_XYZ_INIT
    d_x = creep_x - agent_x
    d_y = creep_y - agent_y
    d_z = creep_z - agent_z

    yaw = -math.degrees(math.atan2(d_x, d_z)) % 360
    pitch = math.degrees(math.atan2(d_y, math.sqrt(d_z ** 2 + d_x ** 2)))

    return yaw, pitch


class InfoParser:
    def __init__(self):
        self.prev_arrow_id = 0
        self.prev_creeper_life = 20.0
        self.prev_damage_taken = 0
        self.creeperx, self.creepery, self.creeperz, _ = CREEPER_INIT

    def init_creeper(self, creep_x_init, creep_y_init, creep_z_init):
        self.creeperx = creep_x_init
        self.creepery = creep_y_init
        self.creeperz = creep_z_init

    @staticmethod
    def evalInfo(info):
        if info:
            return eval(info.replace('false', 'False').replace('true', 'True'))
        else:
            return {}

    def parseInfo(self, info, agentyaw, agentpitch):
        agentx, agenty, agentz = AGENT_XYZ_INIT
        reward = 0

        if info:
            if 'LineOfSight' in info and info['LineOfSight']['type'] == 'Creeper':
                reward += 100

            creeperAlive = False
            for i in info['entities']:
                if i['name'] == 'Creeper':
                    creeperAlive = True
                    self.creeperx = i['x']
                    self.creepery = i['y']
                    self.creeperz = i['z']
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
                            (math.pow((self.creeperx - i['x']), 2) + math.pow((self.creeperz - i['z']), 2)))
                        if LOGGING:
                            print("DISTANCE ", last_arrow_dis)
                        reward -= (last_arrow_dis ** 2) / 3 - 16
                        self.prev_arrow_id = i['id']
            # if 'IsAlive' in info:
            #     isAlive = info['IsAlive']

            if 'DamageTaken' in info:
                damageTaken = info['DamageTaken']
                if damageTaken > 0:
                    reward -= 100

        creeper_yaw, creeper_pitch = calculate_creeper_yaw_pitch(self.creeperx, self.creepery, self.creeperz)
        agentyaw %= 360
        agentpitch = max(min(agentpitch, 90), -90)

        yaw_diff = creeper_yaw - agentyaw
        pitch_diff = creeper_pitch - agentpitch
        state = np.array([yaw_diff, pitch_diff], np.float64)

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

