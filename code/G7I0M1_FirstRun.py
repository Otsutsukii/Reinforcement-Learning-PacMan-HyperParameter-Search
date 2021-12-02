import sys
from PIL import Image
import numpy as np
import gym
from gym.utils.play import play
from random import choice
import time
import random
import matplotlib.pyplot as plt
from scipy import signal

def random_move():
    actions = [0,1,2,3,4,5,6,7,8]
    return choice(actions)

def mycallbacks(obs_t, obs_top1, action, rew, done, info):
    print("action = ", action, " reward = ", rew, " done = ", done,"info = ",info, "obs_t",obs_t[0].shape)

def plot_cumulative_rewards(data):
    plt.plot([i for i in range(len(data))],signal.savgol_filter(data,91,2))
    plt.xlabel("Step t")
    plt.ylabel("Cumulative Rewards per episode")
    plt.savefig("random_agent.png")
    plt.show()

def random_agent(nb_episode,nb_step,game = 'MsPacmanDeterministic-v4'):
    """
    this execute the random agents and plot he smoothed cumulative reward for each game
    """
    env = gym.make(game)
    all_rewards = []
    for i in range(nb_episode):
        seed = (int)(time.time())    
        random.seed(seed)
        env.seed(seed)
        observation = env.reset()
        actions = np.random.randint(0,5,size=nb_step)
        cumulative_reward = 0
        for t in range(nb_step):
            env.render()
            observation, reward, done, info = env.step(actions[t])
            cumulative_reward += reward
            if done:
                all_rewards.append(cumulative_reward/3)
                print("Episode {} finished after {} steps".format(i,t+1))
                break
    env.close()
    return all_rewards

if __name__ == "__main__":
    # we choose 10 episode with 8000 step 
    # the running average of cumulative reward is the culumative_reward/the number of timestep reached
    nb_episode = 500
    nb_step = 8000
    data = random_agent(nb_episode,nb_step)
    plot_cumulative_rewards(data)


