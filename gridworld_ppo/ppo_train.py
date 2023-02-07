# -*- coding: utf-8 -*-
import torch as th
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from env import GridWorldEnv

# Parallel environments
def make_env(envObj, rank, seed=0):
    def _init():
        env = envObj()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Single process: 
env = GridWorldEnv()
env = make_vec_env(lambda: env, n_envs=1)

# Search PPO parameter in: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
# How to custom policy network: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
# A default AC net: dict(pi=[64, 64], vf=[64, 64]) 
# (check in: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py)
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[32, 32], vf=[32, 32]))
model = PPO(policy="MultiInputPolicy", 
            env=env,
            learning_rate=3e-4,
            n_steps=512, # Max steps collected from an episode  
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
            device="cpu",
            _init_setup_model=True
            )

model.learn(total_timesteps=2e4)
model.save("ppo")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo")

env = GridWorldEnv(render_mode="human")
for episode in range(10):
    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
env.close()    





