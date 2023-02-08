# -*- coding: utf-8 -*-
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from env import MazeEnv

def mask_fn(env):
    return env.get_action_mask()        
            
if __name__ == "__main__":
    env = MazeEnv()
    env = ActionMasker(env, mask_fn)
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = MaskablePPO(policy="MlpPolicy", 
                        env=env, 
                        learning_rate=3e-4,
                        n_steps=2048, 
                        batch_size=64,
                        n_epochs=10,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2,
                        normalize_advantage=False,
                        ent_coef=0.01,
                        vf_coef=0.5,
                        max_grad_norm=0.5,
                        target_kl=0.1,
                        tensorboard_log="logs",
                        policy_kwargs=policy_kwargs,
                        seed=32, 
                        verbose=1)

    model.learn(8e4)
    model.save("ppo")
    
    del model
    
    model = MaskablePPO.load("ppo")
    
    env = MazeEnv(render_mode="human")
    for episode in range(1):
        obs = env.reset()
        i = 0
        while True:
            action_masks = env.get_action_mask()
            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, done, info = env.step(action)
            if done:
                break
    env.close()


