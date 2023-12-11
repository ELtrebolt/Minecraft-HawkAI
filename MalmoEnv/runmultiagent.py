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
import argparse
from pathlib import Path
import time
from lxml import etree
from threading import Thread
import threading
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from custom_env.custom_env import CustomEnv

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

    def eval_model(model, env):
        obs, info = env.reset()
        num_episodes = 5
        cur = 0
        while True:
            action, _states = model.predict(obs)
            print(action, _states)
            obs, reward, terminated, truncated, info = env.step(action)
            print(obs, reward)
            if terminated or truncated:
                cur += 1
                if cur > num_episodes:
                    break
                obs, info = env.reset()

    def load_saved_model(path, env):
        return DQN.load(path, env=env)

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

        # == used to load and evaluate a pre-existing model ==
        saved_model = load_saved_model("./model/best_model.zip", env)
        eval_model(saved_model, env)

        # == used to setup the logger and train the DQN model == 
        # tmp_path = "./logs/"
        # # set up logger
        # new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

        # model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=tmp_path)
        # model.set_logger(new_logger)

        # checkpoint_callback = CheckpointCallback(
        #     save_freq=10000,
        #     save_path="./logs/",
        #     name_prefix="rl_model",
        #     save_replay_buffer=True,
        #     save_vecnormalize=True,
        # )

        # # Use deterministic actions for evaluation
        # eval_callback = EvalCallback(env, best_model_save_path="./eval_logs/",
        #                              log_path="./eval_logs/", eval_freq=5000,
        #                              deterministic=True, render=True, verbose=1)

        # callbacks = CallbackList([eval_callback, checkpoint_callback])

        # model.learn(total_timesteps=300000, callback=callbacks, tb_log_name='first_run')
        # model.save("dqn_day2")
        # print('--- DONE TRAINING ---')

        # for i in range(10):
        #     print('Testing in ' + str(10 - i))
        #     time.sleep(1)

        # eval_model(model, env)

        env.close()

    threads = [Thread(target=run, args=(i,)) for i in range(number_of_agents)]

    [t.start() for t in threads]
    [t.join() for t in threads]
