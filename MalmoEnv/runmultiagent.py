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
import random

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
from stable_baselines3 import DQN

from custom_env.custom_env import CustomEnv
from stable_baselines3.common.logger import configure

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='malmovnv test')
    # 'missions/mobchase_single_agent.xml'
    parser.add_argument('--mission', type=str, default='../sample_missions/eating_1.xml', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000, help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='the mission server DNS or IP address')
    parser.add_argument('--server2', type=str, default=None, help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--port2', type=int, default=9000, help="(Multi-agent) role N's mission port")
    parser.add_argument('--episodes', type=int, default=1, help='the number of resets to perform - default is 1')
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
        ACTIONS = ['turn', 'pitch', 'use']
        env = CustomEnv()
        env.init(xml,
                 args.port, server=args.server,
                 server2=args.server2, port2=(args.port + role),
                 role=role,
                 exp_uid=args.experimentUniqueId,
                 episode=args.episode, resync=args.resync, action_filter=ACTIONS)

        def log(message):
            print('[' + str(role) + '] ' + message)

        tmp_path = "./logs/"

        try:
            model = DQN.load('model2', env=env)
        except:
            new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

            model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=tmp_path, learning_rate=0.01)
            model.set_logger(new_logger)

            for i in range(15):
                model.learn(total_timesteps=10000, reset_num_timesteps=False)
                model.save('model2')

        print('--- DONE TRAINING ---')
        for i in range(10):
            print('Testing in ' + str(10-i))
            time.sleep(1)

        obs, info = env.reset()
        num_episodes = 5
        cur = 0
        while True:
            action, _states = model.predict(obs)
            print(action, _states)
            obs, reward, terminated, truncated, info = env.step(action)
            print(obs, reward)
            if reward > 50:
                model.save_replay_buffer(tmp_path + 'test.pkl')
            if terminated or truncated:
                cur += 1
                if cur > num_episodes:
                    break
                obs, info = env.reset()

        # for r in range(args.episodes):
        #     log("reset " + str(r))
        #     env.reset()
        #
        #     done = False
        #     LOS = False
        #     while not done and not LOS:
        #         # Random action
        #         action = env.action_space.sample()
        #
        #         # 0 aim down / 1 aim up
        #         # 2 turn right / 3 turn left
        #         # 4 right click shoot bow
        #
        #         # Turn until pig is aligned with cursor
        #         # Hold Shoot based on distance to pig
        #         # action = 0
        #
        #         log("action: " + str(env.action_space[action]))
        #
        #         if env.action_space[action] == "use 1":
        #             env.step(action)
        #             sleep_time = random.random() * 2
        #             log(f"Sleep Time: {sleep_time}")
        #             time.sleep(sleep_time)
        #             log("action: use 0")
        #             obs, reward, done, info = env.step(env.action_space.actions.index('use 0'))
        #         else:
        #             obs, reward, done, info = env.step(action)
        #
        #         log("reward: " + str(reward))
        #         # log("done: " + str(done))
        #         log("info: " + str(info))
        #         log(" obs: " + str(obs))
        #
        #         if info:
        #             info = eval(info.replace('false', 'False').replace('true', 'True'))
        #             if 'LineOfSight' in info and info['LineOfSight']['type'] == 'Pig':
        #                 LOS = True
        #             # player_coords, pig_coords = get_player_and_pig_coords(info)
        #             # log(str(player_coords) + ' ||| ' + str(pig_coords))
        #             # yaw, pitch = calculate_yaw_and_pitch(player_coords, pig_coords)
        #             # log(str(yaw) + ' ||| ' + str(pitch))
        #             # dyaw, dpitch = discretize_yaw_and_pitch(yaw, pitch)
        #             # log(str(dyaw) + ' ||| ' + str(dpitch))
        #             # turn, move = decide_movement_actions(dyaw, dpitch)
        #             # log(str(turn) + ' ||| ' + str(move))
        #
        #
        #         time.sleep(.05)
        #
        #     for i in range(len(env.action_space)):
        #         log("action: " + str(env.action_space[i]))

        env.close()

    def get_player_and_pig_coords(info):
        pig_coords = {}
        player_coords = {}
        for i in info['entities']:
            if i['name'] == 'Pig':
                pig_coords['x'] = i['x']
                pig_coords['y'] = i['y']
                pig_coords['z'] = i['z']
            elif i['name'] == 'MalmoTutorialBot':
                player_coords['x'] = i['x']
                player_coords['y'] = i['y']
                player_coords['z'] = i['z']
        return [player_coords, pig_coords]

    def calculate_yaw_and_pitch(player_coords, pig_coords):
        # Calculate relative position
        rel_x = pig_coords['x'] - player_coords['x']
        rel_y = pig_coords['y'] - player_coords['y']
        rel_z = pig_coords['z'] - player_coords['z']

        # Calculate yaw (horizontal angle)
        yaw = math.atan2(rel_z, rel_x) * 180 / math.pi
        pitch = math.atan2(rel_y, math.sqrt(rel_x**2 + rel_z**2)) * 180 / math.pi

        return yaw, pitch

    # Constants for discretization
    NUM_YAW_BINS = 4  # Number of bins for yaw (adjust as needed)
    NUM_PITCH_BINS = 3  # Number of bins for pitch (adjust as needed)

    def discretize_yaw_and_pitch(yaw, pitch):
        # Discretize yaw and pitch into bins
        discrete_yaw = round((yaw % 360) / 360 * (NUM_YAW_BINS - 1))
        discrete_pitch = round((pitch + 90) / 180 * (NUM_PITCH_BINS - 1))

        return discrete_yaw, discrete_pitch

    def decide_movement_actions(discrete_yaw, discrete_pitch):
        # Based on the discretized yaw and pitch, decide on movement actions
        # This is a simplified example; you may need to adjust based on your specific movement constraints

        # Example: Turn towards the desired yaw
        turn_action = 1 if discrete_yaw > 0 else -1

        # Example: Move forward or backward based on pitch
        move_action = 1 if discrete_pitch > 0 else -1

        return turn_action, move_action

    threads = [Thread(target=run, args=(i,)) for i in range(number_of_agents)]

    [t.start() for t in threads]
    [t.join() for t in threads]
