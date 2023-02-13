# -*- coding: utf-8 -*-
import gym
from gym import spaces
import pygame
import numpy as np
from stable_baselines3.common.env_checker import check_env


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    MAP = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    ]

    def __init__(self, render_mode=None):
        self.size = 10  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        
        # State
        self.state = {
            "map": None,
            "agent_location": None,
            "target_location": None
        }
        
        # -1: wall, 0: ground, 1: agent, 2: target
        self.observation_space = spaces.Box(-1, 2, shape=(self.size*self.size,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        obs = np.array(self.state["map"]) * -1  
        obs[tuple(self.state["agent_location"])] = 1
        obs[tuple(self.state["target_location"])] = 2
        
        return obs.flatten()

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self.state["agent_location"] - self.state["target_location"]
            )
        }
    
    def get_action_mask(self):
        RIGHT, UP, LEFT, DOWN = 0, 1, 2, 3
        
        right_to_location = np.clip(self.state["agent_location"] + self._action_to_direction[RIGHT], 0, self.size - 1)
        up_to_location = np.clip(self.state["agent_location"] + self._action_to_direction[UP], 0, self.size - 1)
        left_to_location = np.clip(self.state["agent_location"] + self._action_to_direction[LEFT], 0, self.size - 1)
        down_to_location = np.clip(self.state["agent_location"] + self._action_to_direction[DOWN], 0, self.size - 1)
            
        action_mask = np.array(
            [
                not np.array_equal(right_to_location, self.state["agent_location"]) and self.state["map"][tuple(right_to_location)] == 0,
                not np.array_equal(up_to_location, self.state["agent_location"]) and self.state["map"][tuple(up_to_location)] == 0,
                not np.array_equal(left_to_location, self.state["agent_location"]) and self.state["map"][tuple(left_to_location)] == 0,
                not np.array_equal(down_to_location, self.state["agent_location"]) and self.state["map"][tuple(down_to_location)] == 0,
            ]
        )
        
        return action_mask

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(np.random.randint(0, 1000))
        
        self.state["map"] = np.array(self.MAP)
        self.state["agent_location"] = np.array([0, 0])
        self.state["target_location"]= np.array([self.size - 1, self.size - 1])

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation
    
    def reset_plus(self, trans_type, seed=None):
        if trans_type == "fliplr":
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed(np.random.randint(0, 1000))
                
            self.state["map"] = np.fliplr(np.array(self.MAP))
            self.state["agent_location"] = np.array([0, self.size - 1])
            self.state["target_location"]= np.array([self.size - 1, 0])
            
            observation = self._get_obs()

            if self.render_mode == "human":
                self._render_frame()

            return observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[int(action)]
        # We use `np.clip` to make sure we don't leave the grid
        next_agent_location = np.clip(
            self.state["agent_location"] + direction, 0, self.size - 1
        )
        
        if self.state["map"][tuple(next_agent_location)] == 0:
            self.state["agent_location"] = next_agent_location
        
        # An episode is done iff the agent has reached the target
        done = np.array_equal(self.state["agent_location"], self.state["target_location"])
        reward = 100 if done else -0.05  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info

    def random_step(self, use_action_mask=True):
        if use_action_mask:
            action_mask = self.get_action_mask()
            valid_action = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_action)
        else:
            action = self.action_space.sample()
        
        return self.step(action)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.state["target_location"],
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.state["agent_location"] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        
        # Then we draw the wall
        for i in range(self.size):
            for j in range(self.size):
                if self.state["map"][i, j] == 1:
                    pygame.draw.rect(
                        canvas,
                        (170, 170, 170),
                        pygame.Rect(
                            pix_square_size * np.array([i, j]),
                            (pix_square_size, pix_square_size)
                        ),
                    )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (120, 120, 120),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                (120, 120, 120),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    env = MazeEnv(render_mode="human")
    obs = env.reset(seed=0)
    check_env(env)
    
    while True:
        obs, reward, done, info = env.random_step()
        if done:
            break
    env.close()
    
    
    
    
    


