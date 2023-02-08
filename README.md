# sb3_practice
Long-term collection of RL practice based on Stable Baselines 3.

Lastest version of SB3: [Installation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html).

## Gridworld by PPO
Gridworld is modified from a custom Gym enviroment (for details see [here](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)),
where an episode ends when the agent reaches the destination.

See codes [here](https://github.com/wwsyan/sb3_practice/tree/main/gridworld_ppo).

### Video
| Performance | Description |
| :---------: | :---------: |
| <img src="images/gridworld_random.gif" width="70%" height="70%"> | Random step |
| <img src="images/gridworld_ppo.gif" width="70%" height="70%"> | PPO agent |

## Maze by maskable PPO
Maze is a 2d gridworld-like enviroment.
<ul>
<li>Masking invalid actions greatly speeds up the training process of the neural network. </li>
<li>1d observation is applied though it's natural to use a image-like observation.
The reason is, CnnPolicy require image data to have a minimum size of 36x36.
Lukily, 1d observation still works well. </li>
</ul>

See codes [here](https://github.com/wwsyan/sb3_practice/tree/main/maze_ppo).

### Log
| Plot | Description |
| :---------: | :---------: |
| <img src="images/maze_ppo_ep_len_mean.png" width="100%" height="100%"> | Average length of episodes |
| <img src="images/maze_ppo_ep_rew_mean.png" width="100%" height="100%"> | Average rewards in an episode |

### Video
| Performance | Description |
| :---------: | :---------: |
| <img src="images/maze_random.gif" width="70%" height="70%"> | Random step |
| <img src="images/maze_ppo.gif" width="70%" height="70%"> | PPO agent |









