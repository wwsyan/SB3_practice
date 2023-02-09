# sb3_practice
Long-term collection of RL practice in the framework of Stable Baselines 3.

Lastest version of SB3: [Installation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html).

Experimental features based on SB3: [SB3 Contrib](https://stable-baselines3.readthedocs.io/en/master/guide/sb3_contrib.html).

Anything about Gym enviroment: [Gym Documentation](https://www.gymlibrary.dev/).

## Cartpole by PPO
Cartpole is a classical Gym enviroment (for details see [here](https://gymnasium.farama.org/environments/classic_control/cart_pole/)).

See codes [here](https://github.com/wwsyan/sb3_practice/tree/main/cartpole_ppo).

### Log
| Average episode length | Average episode reward |
| :---------: | :---------: |
| <img src="images/cartpole_ppo_ep_len_mean.png" width="100%" height="100%"> | <img src="images/cartpole_ppo_ep_rew_mean.png" width="100%" height="100%"> |

## Lunarlander by PPO
Lunarlander is a classical Gym enviroment, aiming at rocket trajectory optimization 
(for details see [here](https://gymnasium.farama.org/environments/box2d/lunar_lander/)).

See codes [here](https://github.com/wwsyan/sb3_practice/tree/main/lunarlander_ppo)

### Log
| Average episode length | Average episode reward |
| :---------: | :---------: |
| <img src="images/lunarlander_ppo_ep_len_mean.png" width="100%" height="100%"> | <img src="images/lunarlander_ppo_ep_rew_mean.png" width="100%" height="100%"> |

## Gridworld by PPO
Gridworld is modified from a custom Gym enviroment (for details see [here](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)),
where an episode ends when the agent reaches the destination.

See codes [here](https://github.com/wwsyan/sb3_practice/tree/main/gridworld_ppo).

### Video
| Random step | PPO agent |
| :---------: | :---------: |
| <img src="images/gridworld_random_0.gif" width="40%" height="40%"> | <img src="images/gridworld_ppo_0.gif" width="40%" height="40%"> |

## Maze by maskable PPO
Maze is a 2d gridworld-like enviroment.
<ul>
<li>Masking invalid actions greatly speeds up the training process of neural network. </li>
<li>1d observation is applied though it's natural to use a image-like observation.
The reason is, CnnPolicy require image data to have a minimum size of 36x36.
Lukily, a flattened observation still works well. </li>
</ul>

See codes [here](https://github.com/wwsyan/sb3_practice/tree/main/maze_ppo).

### Log
| Average episode length | Average episode reward |
| :---------: | :---------: |
| <img src="images/maze_ppo_ep_len_mean.png" width="100%" height="100%"> | <img src="images/maze_ppo_ep_rew_mean.png" width="100%" height="100%"> |

### Video
| Random step | PPO agent |
| :---------: | :---------: |
| <img src="images/maze_random.gif" width="50%" height="50%"> | <img src="images/maze_ppo.gif" width="50%" height="50%"> |









