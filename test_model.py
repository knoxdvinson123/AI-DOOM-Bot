
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from VizDoomGym import VizDoomGym
import time

#load model
model = PPO.load('./train/train_basic/best_model_100000')

#render env
env = VizDoomGym(render=True)

#evaluate mean reward
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

# print(mean_reward)



for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0;
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        time.sleep(0.20)
        total_reward += reward
    print('Total reward for episode {} is {}'.format(total_reward, episode))
    time.sleep(2)