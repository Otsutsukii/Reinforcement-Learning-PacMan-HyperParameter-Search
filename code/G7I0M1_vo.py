import sys
from PIL import Image
import numpy as np
import gym
from gym.utils.play import play
from random import choice
import imageio

# For the game pacman we have 5 actions to be choosen by the agent 
#  for french keyboard azert, we have UP = s , LEFT = q, RIGHT = d, DOWN = s, and touch nothing then the agent stay at the same place 
#  the discrete value of the actions are : UP = 1, RIGHT = 2, LEFT = 3, DOWN = 4, Nothing = 0
#  RIGHT + UP = 5, UP + LEFT = 6, RIGHT + DOWN = 7, DOWN + LEFT = 8
#  rewards : 0 if we stay or moving without eating a white point, 10 if we move and eat a white point, 50 if we eat a big flashing white point
#  200 if after we ate the flashing point, we ate a ghost 

# the obs_t is an image of the state, and it has 3 dimension (height, width, channel) 
# so later we can convert this to grayscale and crop the image to keep only relevant part, without game info at the bottom.

def mycallbacks(obs_t, obs_top1, action, rew, done, info):
    print(obs_t.shape)
    img = Image.fromarray(obs_t)
    img = img.resize((84,84)).convert('L')
    img.save("test.jpg")
    print("action = ", action, " reward = ", rew, " done = ", done,"info = ",info, "obs_t",obs_t[0].shape)

if __name__ == "__main__":
    env = gym.make('MsPacmanDeterministic-v4')
    play(env, zoom=3, fps=12,callback=mycallbacks)
    nb_actions = env.action_space.n
    print(env.action_space.n)

