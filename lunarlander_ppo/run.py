# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 20:00:19 2023

@author: WSY
"""

import gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("LunarLander-v2", n_envs=4)

# Search PPO parameter in: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
# How to custom policy network: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
# A default AC net: dict(pi=[64, 64], vf=[64, 64]) 
# (check in: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py)
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[64, 64], vf=[64, 64]))
model = PPO(policy="MlpPolicy", 
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None, # clip for value function
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False, # Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration 
            # (default: False)
            sde_sample_freq=-1, # Sample a new noise matrix every n steps when using gSDE
            # (default: -1 (only sample at the beginning of the rollout))
            target_kl=None,
            tensorboard_log="tb_log", # Log direction
            policy_kwargs=policy_kwargs,
            verbose=1, # Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
            seed=None,
            device="auto",
            _init_setup_model=True
            )

model.learn(total_timesteps=100e4)
model.save("ppo_lunarlander")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_lunarlander")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()