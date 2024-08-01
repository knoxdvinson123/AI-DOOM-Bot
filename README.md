# AI Doom Bot
<p align="center">
  <img src="https://github.com/user-attachments/assets/6860e699-4e8e-4efa-8824-ef544898fc3d" />
</p>

## Project Overview
This project involves creating an AI model to play a 90s first-person POV alien shooter game using Python. The game simulation is run using VizDoom, a software designed for training and visualizing AI in a virtual environment. The AI model is trained using reinforcement learning, leveraging the OpenAI Gym API, which wraps around the VizDoom environment.
<p align="center">
  <img src="https://github.com/user-attachments/assets/758ef92c-7100-4e1d-933b-d6620b83a676" />
</p>

## Features
- VizDoom Integration: Utilizes VizDoom to simulate the game environment.
- OpenAI Gym API: Wraps the VizDoom environment in the OpenAI Gym API for ease of training.
- Reinforcement Learning: Employs a reinforcement learning approach with both positive and negative feedback mechanisms.
- Efficiency Incentives: The bot receives negative feedback for every step taken and every missed shot to encourage efficient gameplay.

## Requirements
- Python 3.6+
- VizDoom
- OpenAI Gym
- Stable_Baselines_3 (for Policy algorithms)
- Numpy
- TensorFlow or PyTorch (depending on the chosen RL library)

## How It Works
The AI bot is trained using reinforcement learning, where it learns to play the game through a process of trial and error. The bot receives:
- Positive feedback: For actions that bring it closer to achieving the game's objectives.
- Negative feedback: For every step taken and every missed shot, incentivizing the bot to play the game efficiently.
The training process involves running numerous simulations where the bot continuously improves its strategy based on the feedback received.

## Acknowledgments
VizDoom for providing the game simulation environment.
OpenAI Gym for the API used to wrap the VizDoom environment.
Stable Baselines3 for the reinforcement learning algorithms.
And online resources + youtube!
