from __future__ import print_function

import sys
sys.path.append("../") 

from datetime import datetime
import numpy as np
import gym
import os
import json
from train import *

from agent.bc_agent import BCAgent
from utils import *


def preprocess_state(state):
    state = rgb2gray(state)
    state = state[12:-12, 12:-12]
    return state


def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 

    while True:
        
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        state = preprocess_state(state)

        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because
        #  it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
    

        action_id = torch.argmax(agent.predict(torch.Tensor(state.reshape(1, 1, 72, 72))))
        

        a = id_to_action(action_id)

        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    agent = BCAgent(class_weights=[0.25, 0.25, 0.25, 0.25])
    agent.load("models/bc_agent_0.pt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
