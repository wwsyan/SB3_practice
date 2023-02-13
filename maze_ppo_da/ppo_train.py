# -*- coding: utf-8 -*-
import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.utils import obs_as_tensor
from env import MazeEnv


def mask_fn(env):
    return env.get_action_mask()

class DataAugmentCallback(BaseCallback):
    def __init__(self, 
                 verbose=0,
                 env=None,
                 model=None,
                 rollout_buffer=None,
                 print_buffer_data=False, # Whether to print buffer data for debugging
                 drop_episode=False, # Whether to drop undone episode
                 use_DA=False, # Whether to use data augment
                 
    ):
        super().__init__(verbose)
        self.env = env
        self.model = model
        self.rollout_buffer = rollout_buffer
        self.print_buffer_data = print_buffer_data
        self.drop_episode = drop_episode
        self.use_DA = use_DA
        
    def _on_step(self) -> bool:
        # This is an abstractmethod, you need to redefine
        return True
    
    def _print_buffer_data(self):
        print("Rollout_buffer data:")
        for key, value in self.rollout_buffer.__dict__.items():
            print(key, "=")
            print(value, "\n")
            
    def _data_augment(self):
        """
        Data augment by rotating and flipping, which generates extra 7 batch data:
            rotate 90, rotate 90 + fliplr
            rotate 180, rotate 180 + fliplr
            rotate 270, rotate 270 + fliplr
            rotate 360, rotate 360 + fliplr
                
        :method direction_value_trans: 
        :method grid_trans: 
        
        """
        def direction_trans(direction, trans_type, rot_times=0):
            RIGHT, UP, LEFT, DOWN = 0, 1, 2, 3
            
            def step(direction, trans_type) -> int:
                if trans_type == "rot90":
                    if direction == UP: return LEFT
                    if direction == DOWN:  return RIGHT 
                    if direction == LEFT:  return DOWN 
                    if direction == RIGHT:  return UP
                if trans_type == 'fliplr':
                    if direction == UP: return UP
                    if direction == DOWN: return DOWN
                    if direction == LEFT: return RIGHT
                    if direction == RIGHT: return LEFT
            
            if trans_type == "rot90":
                direction_new = direction
                for i in range(rot_times):
                    direction_new = step(direction_new, "rot90")
            if trans_type == "fliplr":
                direction_new = step(direction, "fliplr")
            
            return direction_new
        
        def grid_trans(self, observation, trans_type, rot_times=0):
            SIZE = self.env.size
            
            # Generate observation grid data
            obs_grid = observation.reshape(SIZE, SIZE)
            
            if trans_type == "rot90":
                obs_grid = np.rot90(obs_grid, rot_times)
            if trans_type == "fliplr":
                obs_grid = np.fliplr(obs_grid)
            
            return obs_grid.flatten()
        
        def action_masks_trans(action_masks, trans_type):
            RIGHT, UP, LEFT, DOWN = 0, 1, 2, 3
            
            action_masks_new = action_masks.copy()
            if trans_type == "fliplr":
                action_masks_new[RIGHT] = action_masks[LEFT]
                action_masks_new[LEFT] = action_masks[RIGHT]
            
            return action_masks_new
        
        ############## Data Augment main ##############
        n_steps = self.rollout_buffer.buffer_size
        n_env = 2 # Original data and augmented data: fliplr
        self.rollout_buffer.n_envs = n_env
        
        DA_obs = np.zeros((n_steps, n_env, self.env.observation_space.shape[0]))
        DA_actions = np.zeros((n_steps, n_env, 1), dtype=int)
        DA_action_masks = np.zeros((n_steps, n_env, self.env.action_space.n), dtype=int)
        
        for i in range(n_steps):
            step_obs = self.rollout_buffer.observations[i, 0]
            step_action = self.rollout_buffer.actions[i, 0]
            step_action_masks = self.rollout_buffer.action_masks[i, 0]
            
            DA_obs[i, 0] = step_obs
            DA_obs[i, 1] = grid_trans(self, step_obs, trans_type="fliplr")
            
            DA_actions[i, 0] = step_action
            DA_actions[i, 1] = direction_trans(int(step_action), trans_type="fliplr")
            
            DA_action_masks[i, 0] = step_action_masks
            DA_action_masks[i, 1] = action_masks_trans(step_action_masks, trans_type="fliplr")
            
        # Check validity of augmented data 
        check_DA = False
        if check_DA:
            self._check_augmented_data(DA_obs, DA_actions, DA_action_masks)     
        
        # Rebuild 1.observations, 2.actions and 3.action_masks
        self.rollout_buffer.observations = DA_obs
        self.rollout_buffer.actions = DA_actions
        self.rollout_buffer.action_masks = DA_action_masks
        
        # Recompute 4.values and 5.logprobs 
        # Please check method compute_returns_and_advantage() in stable_baselines3.common.RolloutBuffer
        DA_values = np.zeros((n_steps, n_env))
        DA_log_probs = np.zeros((n_steps, n_env))
        with th.no_grad():
            for batch_rank in range(2):
                DA_obs_tensor = obs_as_tensor(DA_obs, self.model.device)
                DA_actions_tensor = th.tensor(DA_actions, device=self.model.device).long()
                values, log_prob, entropy = self.model.policy.evaluate_actions(DA_obs_tensor[:, batch_rank], 
                                                                               actions=DA_actions_tensor[:, batch_rank].flatten(),  
                                                                               action_masks=DA_action_masks[:, batch_rank]
                                                                               )
                DA_values[:, batch_rank] = values.cpu().numpy().reshape(-1)
                DA_log_probs[:, batch_rank] = log_prob.cpu().numpy()
        
        self.rollout_buffer.values = DA_values
        self.rollout_buffer.log_probs = DA_log_probs
        
        # Rebuild 8.episode_starts and 9.rewards
        self.rollout_buffer.episode_starts = np.tile(self.rollout_buffer.episode_starts, (1, n_env))
        self.rollout_buffer.rewards = np.tile(self.rollout_buffer.rewards, (1, n_env))
        
        # Recompute 6.returns and 7.advantages
        self.rollout_buffer.advantages = np.zeros((self.rollout_buffer.buffer_size, self.rollout_buffer.n_envs), dtype=np.float32)
        # Check: stable_baselines3.common.buffers.RolloutBuffer
        # :param last_values: state value estimation for the last step (one for each env)
        # :param dones: if the last step was a terminal step (one bool for each env).
        last_value, last_done = th.zeros(1), 0
        self.rollout_buffer.compute_returns_and_advantage(last_values=last_value, dones=last_done)
        
        self.rollout_buffer.returns = self.rollout_buffer.returns.astype(np.float32)
        self.rollout_buffer.values = self.rollout_buffer.values.astype(np.float32)
        self.rollout_buffer.log_probs = self.rollout_buffer.log_probs.astype(np.float32)
        
        
    def _check_augmented_data(self, DA_obs, DA_actions, DA_action_masks) -> None:
        n_steps = self.rollout_buffer.episode_starts.size
        ROW, COL = self.env.size, self.env.size
        
        batch_rank = 1 # 0~1
        print("Check augmented data in batch rank:", batch_rank)
        for i in range(n_steps):
            print(DA_obs[i, batch_rank].reshape(ROW, COL))
            legal_actions = np.where(DA_action_masks[i, batch_rank] == 1)[0]
            print("legal actions:", legal_actions)
            print("choose action:", DA_actions[i, batch_rank, 0])
            
        
    def _on_rollout_start(self) -> None:
        """
        This event is triggered before collecting new samples.
        In order to apply data augment, we rebuild the buffer, which will raise unfull mistake: assert self.fill "".
        Recover the buffer size and reset the buffer will solve this.
        """
        self.rollout_buffer.buffer_size = self.model.n_steps
        self.rollout_buffer.n_envs = 1
        self.rollout_buffer.reset()
        
        
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        
        In this case, above datas are delivered to training phase for calculation:
            (please check method _get_samples() in: sb3_contrib.common.maskable.buffers
             and method train() in: sb3_contrib.ppo_mask.ppo_mask.MaskablePPO)
            1. observations
            2. actions
            3. action_masks
            4. old_values/values
            5. old_log_prob/log_probs
            6. returns
            7. advantages
            
        In order to recompute returns and advantages, done information is also needed:
            8. episode_starts
            9. rewards
        """
        # Print data for debugging
        # Remember: keep your augmented data in the same form
        if self.print_buffer_data:
            self._print_buffer_data()
        
        if self.drop_episode:
            
            drop_flag = 0
            for i in reversed(range(self.rollout_buffer.buffer_size)):
                if self.rollout_buffer.episode_starts[i, 0] == 1:
                    drop_flag = i
                    break
            
            if drop_flag != 0:
            # Drop undone data
                self.rollout_buffer.buffer_size = drop_flag
                self.rollout_buffer.observations = self.rollout_buffer.observations[:drop_flag]
                self.rollout_buffer.actions = self.rollout_buffer.actions[:drop_flag]
                self.rollout_buffer.action_masks = self.rollout_buffer.action_masks[:drop_flag]
                self.rollout_buffer.values = self.rollout_buffer.values[:drop_flag]
                self.rollout_buffer.log_probs = self.rollout_buffer.log_probs[:drop_flag]
                self.rollout_buffer.returns = self.rollout_buffer.returns[:drop_flag]
                self.rollout_buffer.advantages = self.rollout_buffer.advantages[:drop_flag] 
                self.rollout_buffer.episode_starts = self.rollout_buffer.episode_starts[:drop_flag] 
                self.rollout_buffer.rewards = self.rollout_buffer.rewards[:drop_flag] 
            
        if self.use_DA:
            self._data_augment()

        
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
                        target_kl=None,
                        tensorboard_log="logs",
                        policy_kwargs=policy_kwargs,
                        seed=1, 
                        verbose=1)
    
    DACallback = DataAugmentCallback(env=env,
                                      model=model, 
                                      rollout_buffer=model.rollout_buffer,
                                      print_buffer_data=False,
                                      drop_episode=True,
                                      use_DA=True)
                                     
    model.learn(15e4, callback=DACallback)
    model.save("ppo")
    
    del model
    
    model = MaskablePPO.load("ppo")
    
    env = MazeEnv(render_mode="human")
    for episode in range(1):
        obs = env.reset_plus(trans_type="fliplr")
        while True:
            action_masks = env.get_action_mask()
            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, done, info = env.step(action)
            if done:
                break
    env.close()


