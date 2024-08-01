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
#this helps save at different stages of training, instead of doing so manually

#import os for file nav
import os

#import callback class from sb3
from stable_baselines3.common.callbacks import BaseCallback

#import ppo for training
from stable_baselines3 import PPO
from VizDoomGym import VizDoomGym
from Setup_Callback import TrainAndLoggingCallback

CHECKPOINT_DIR = './train/train_basic'
LOG_DIR = './logs/log_basic'
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

#non-rendered env
env = VizDoomGym()

model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048)
#n_steps is the batch size of observations, actions, log probabilites, and values stored in buffer for one iteration
#tensorboard_log is where it saves the logs

#having a low n_steps=256 value for this env generally requires a higher clip_range value

model.learn(total_timesteps=100000, callback=callback)

#approx_kl --> measure of how differnet current agent is compared to previous agent
    #ideally should be vary, not the same,
    #but shouldnt spike (unstable training), PPO clips unstable training to preserve the rest of the training
    #you can increase the clip_range and gae_lambda hyperparams, if there is massive spikes in approx_kl
#explained_variance --> measure of how well critic model can explain variance in value function,
    #want it to be postive b/c it ca explain current state in more detail
#policy_gradient_loss --> tells us how well the agent is able to take actions to capitalie on its advantage, want it ot bdecrease which means less loss
#value_loss --> how well agent is able to predict current return based on current state and action, we want it to decrease

#if policy_gradient_loss goes quickly to 0, this is a sign it is not training well.


#to view these returned values on graphs:
    #cd to logs/log_basic/PPO_#
    #tensorboard --logdir=.

    #tensorboard:
    #ep_len_mean, decrease means it could be winning or losing fast
    #ep_rew_mean, want to increase, mean reward per episode
    #approx_kl, how different new policy is from old policy, some is good
    #explained_variance, how well it can predict future