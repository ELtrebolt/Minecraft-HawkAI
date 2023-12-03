# ------------------------------------------------------------------------------------------------
# Copyright (c) 2018 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

import malmoenv
import argparse
from pathlib import Path
import time
from lxml import etree
from threading import Thread
import threading
import numpy as np
from collections import defaultdict
import math

# Modeling
from sample import QNetwork, DQN

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='malmovnv test')
    # 'missions/mobchase_single_agent.xml'
    parser.add_argument('--mission', type=str, default='../sample_missions/eating_1.xml', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000, help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='the mission server DNS or IP address')
    parser.add_argument('--server2', type=str, default=None, help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--port2', type=int, default=9000, help="(Multi-agent) role N's mission port")
    parser.add_argument('--episodes', type=int, default=10, help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=0, help='the start episode - default is 0')
    parser.add_argument('--resync', type=int, default=0, help='exit and re-sync on every N - default 0 meaning never')
    parser.add_argument('--experimentUniqueId', type=str, default='test1', help="the experiment's unique id.")
    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server

    xml = Path(args.mission).read_text()

    mission = etree.fromstring(xml)
    number_of_agents = len(mission.findall('{http://ProjectMalmo.microsoft.com}AgentSection'))
    print("number of agents: " + str(number_of_agents))

    def run(role):
        ACTIONS = ['turn', 'use']
        env = malmoenv.make()
        env.init(xml,
                 args.port, server=args.server,
                 server2=args.server2, port2=(args.port + role),
                 role=role,
                 exp_uid=args.experimentUniqueId,
                 episode=args.episode, resync=args.resync, action_filter=ACTIONS)
        state_size = 9
        action_size = 4

        def log(message):
            print('[' + str(role) + '] ' + message)

        agent = DQN(state_size, action_size)

        episodesMetrics, episodesRewards, positives = [], [], []
        for r in range(args.episodes):
            log("reset " + str(r))
            env.reset()

            done = False 
            LOS, creeperAlive, totalReward = False, True, 0
            agentx, agenty, agentz, agentyaw = 15,57,0,90
            creeperx, creepery, creeperz, creeperyaw = 15,57,-10,0
            last_arrow_dis = float('inf')
            state = [agentx, agenty, agentz, agentyaw,
                    creeperx, creepery, creeperz, creeperyaw, last_arrow_dis]

            while not done:
                # Random action
                # action = 1

                # 0 aim down / 1 aim up
                # 2 turn right / 3 turn left
                # 4 begin charging bow
                # 5 release bow
                action = agent.act(state)

                # Turn until pig is aligned with cursor
                # Hold Shoot based on distance to pig
                # action = 0

                log("action: " + str(env.action_space[action]))
                obs, reward, done, info = env.step(action)

                log("reward: " + str(reward))
                # log("done: " + str(done))
                log("info: " + str(info))
                log(" obs: " + str(obs))

                if info:
                    info = eval(info.replace('false', 'False').replace('true', 'True'))
                    if 'LineOfSight' in info and info['LineOfSight']['type'] == 'Creeper':
                        LOS = True
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
                        elif i == info['entities'][-1] and i['name'] == 'Arrow':
                            if abs(i['motionX']) < 0.3 and abs(i['motionY']) < 0.3 and abs(i['motionZ']) < 0.3:
                                last_arrow_dis = math.sqrt( ( math.pow((creeperx - i['x']),2) + math.pow((creepery - i['y']),2) ) )

                    if 'IsAlive' in info:
                        isAlive = info['IsAlive']
                    if 'DamageDealt' in info:
                        damageDealt = info['DamageDealt']
                    if 'DamageTaken' in info:
                        damageTaken = info['DamageTaken']
                        if damageTaken > 0:
                            reward -= 100
                totalReward += reward

                next_state = [agentx, agenty, agentz, agentyaw,
                        creeperx, creepery, creeperz, creeperyaw, last_arrow_dis]
                agent.remember(state, action, reward, next_state, done)
                agent.train()
                state = next_state
                log("last_arrow_dis: " + str(last_arrow_dis))

                time.sleep(.05)

            TARGET_UPDATE_FREQ = 10
            if r % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            for i in range(len(env.action_space)):
                log("action: " + str(env.action_space[i]))
            
            metrics = [isAlive, creeperAlive, damageDealt, damageTaken]

            episodesMetrics.append(metrics)
            log("metrics: " + str(metrics))

            episodesRewards.append(totalReward)
            log("totalReward: " + str(totalReward))

        log("AllRewards: " + str(episodesRewards))
        log("Positives: " + str(positives))
        env.close()

    threads = [Thread(target=run, args=(i,)) for i in range(number_of_agents)]

    [t.start() for t in threads]
    [t.join() for t in threads]
