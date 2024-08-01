#importing dependencies
from vizdoom import *
#random to sample actions
import random
#needed for "sleep"
import time
#make a set of actions it can take using numpy (identity matrix)
import numpy as np

#-----------------------
#pip install vizdoom
#git clone https://github.com/Farama-Foundation/ViZDoom
#-----------------------

#setup
game = DoomGame()
game.load_config('github/scenarios/basic.cfg')
game.init()

#action set for agent in environment
actions = np.identity(3, dtype=np.uint8)
#actions
#left(100) right(010) attack(001)
random.choice(actions)

episodes = 10
for episode in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state() #game state
        img = state.screen_buffer #this displays current state of game frame
        info = state.game_variables #the number relates to AMMO amount (from basic.cfg)
        reward = game.make_action(random.choice(actions)) #takes action
        print('reward:', reward)
        time.sleep(0.02)
    print('Result:', game.get_total_reward())
    time.sleep(2)

game.close()