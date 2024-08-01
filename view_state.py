#importing dependencies
from vizdoom import *
#random to sample actions
import random
#needed for "sleep"
import time
#make a set of actions it can take using numpy (identity matrix)
import numpy as np
#import env base class (OpenAI Gym)
from gym import Env
#import gym spaces (Discrete, Box)
from gym.spaces import Discrete, Box
#Import OpenCV (uses for grayscaling observations)
import cv2

#import env class I wrote in VizDoomGym.py
from VizDoomGym import VizDoomGym
from matplotlib import pyplot as plt

env = VizDoomGym(render=True)
state = env.reset()

print(state.shape)
print(np.moveaxis(state, 0, -1).shape)

#shows a plot of the rendered grayscaled and resized frame
plt.imshow(cv2.cvtColor(state, cv2.COLOR_BGR2RGB))
plt.show()


# #Import Environment Checker
# from stable_baselines3.common import env_checker
# #this checks if the environment is correct
# env_checker.check_env(env)














