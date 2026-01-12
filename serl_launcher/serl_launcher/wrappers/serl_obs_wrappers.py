import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten, Box


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treats the observation space as a dictionary
    of a flattened state space and the images.
    
    Handles two cases:
    1. Dict state space (Franka hardware): Flattens selected proprio_keys from state dict
    2. Box state space (simulation envs): State already flat, just unpacks images
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        
        # Check if state is Dict or Box (already flat)
        state_space = self.env.observation_space["state"]
        self.state_is_dict = isinstance(state_space, gym.spaces.Dict)
        
        if self.state_is_dict:
            # Original behavior: Dict state space needs flattening
            self.proprio_keys = proprio_keys
            if self.proprio_keys is None:
                self.proprio_keys = list(state_space.keys())
            
            self.proprio_space = gym.spaces.Dict(
                {key: state_space[key] for key in self.proprio_keys}
            )
            
            self.observation_space = gym.spaces.Dict({
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
            })
        else:
            # New behavior: Box state space (already flat)
            # Just unpack images to top level
            self.proprio_keys = None
            self.observation_space = gym.spaces.Dict({
                "state": state_space,  # Keep as-is
                **(self.env.observation_space["images"]),
            })

    def observation(self, obs):
        if self.state_is_dict:
            # Flatten Dict state
            obs = {
                "state": flatten(
                    self.proprio_space,
                    {key: obs["state"][key] for key in self.proprio_keys},
                ),
                **(obs["images"]),
            }
        else:
            # State already flat, just unpack images
            obs = {
                "state": obs["state"],
                **(obs["images"]),
            }
        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info

def flatten_observations(obs, proprio_space, proprio_keys):
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs