#importing dependencies
from vizdoom import *
#random to sample actions
import random
#needed for "sleep"
import time
#make a set of actions it can take using numpy (identity matrix)
import numpy as np

#-----------------------
#INSTALL VIZDOOM
#pip install vizdoom
#git clone https://github.com/Farama-Foundation/ViZDoom
#-----------------------
#CONFIGURING INTO GYM ENV
#pip install gym
#pip install opencv-python
#pip install matplotlib
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install stable-baselines3[extra]
#pip install 'shimmy>=0.2.1'  (this bridges gym and gymnasium)
#pip install tensorboard

#-----------------------

#import env base class (OpenAI Gym)
from gym import Env
#import gym spaces (Discrete, Box)
from gym.spaces import Discrete, Box
#Import OpenCV (uses for grayscaling observations)
import cv2

#-----------------------------------------------
#OpenAI GYM ENV CONFIGURATION
    #the idea is to use VizDoom as an openAI gym environment using the base class from gym
    #Then we grayscale the game to reduce the amount of "noise"
        #making it easier/faster to train with less confusion

#Discrete Vs. Box
    #Discrete(3).sample()
    #Discrete --> is an index for things like actions
    #Box(low=0, high=5, shape=(5,5)) --> 5 by 5 array with low value of 0 and high of 5
    #Box(low=0, high=5, shape=(5,5), dtype = np.uint8) // Doom use 320 by 240
#---------------------------------------------------------------------------------------

#CREATE VIZDOOM OpenAI GYM ENV
#creating VizDoomGym(Env)
class VizDoomGym(Env):
    # Function called when we start the env
    def __init__(self, render=False):
        #inherit from Ev
        super().__init__()
        #setup the game
        self.game = DoomGame()
        self.game.load_config('github/scenarios/basic.cfg') #change to use diff envs from VizDoom GitHub

        #render frame (or not)
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        self.game.init() #starts game

        #create observation and action space
        #low/high is for pixel
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        #to fiugre out shape, game.get_state().screen_buffer.shape //(3, 240, 320)
        self.action_space = Discrete(3)

    # This takes a step in the env
    # actions is 0 (100 left), 1 (010 right), or 2 (001 shoot)
    def step(self, action):
        #Specify actions and take step
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4)

        #Get all other information for return
        #logic need b/c errors thrown if it is on the end screen
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state) #applies grayscale method
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else:
            state = np.zeros(self.observation_space.shape) #all zeros
            info = 0
        info = {"info":info}
        done = self.game.is_episode_finished()

        return state, reward, done, info

    def render(): #Define how to render game // VizDoom does this already
        pass

    # Grayscales game frame and resizes it
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_RGB2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        #resizes smaller, less pixels to process
        state = np.reshape(resize, (100, 160, 1))
        return state #this removes the color channels, the 3 in the shape

    # This happens when we start a new game
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)

    def close(self): #closes the game window
        self.game.close()


