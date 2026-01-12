"""Wrapper to extract single arm actions from dual-arm action space."""

import gymnasium as gym
import numpy as np


class SingleArmWrapper(gym.Wrapper):
    """
    Wrapper for dual-arm environments to use only one arm with single-arm agents.
    
    Extracts a subset of actions (7D: pos, rot, grip) from full dual-arm actions (14D).
    The environment still receives 14D actions, but the agent only controls one arm.
    The other arm's actions are set to neutral (zero deltas, maintain grip state).
    """
    
    def __init__(self, env, active_arm="right"):
        """
        Args:
            env: Gym environment with 14D action space [L_pos, L_rot, L_grip, R_pos, R_rot, R_grip]
            active_arm: Which arm to control ("left" or "right")
        """
        super().__init__(env)
        self.active_arm = active_arm
        
        # Original action space should be 14D
        assert env.action_space.shape == (14,), f"Expected 14D actions, got {env.action_space.shape}"
        
        # Agent will control 7D action space
        self.action_space = gym.spaces.Box(
            low=env.action_space.low[:7],
            high=env.action_space.high[:7],
            shape=(7,),
            dtype=env.action_space.dtype
        )
        
        # Initialize idle arm state (last gripper command, used for maintaining grip)
        self.idle_arm_last_grip = 0.0
        
    def reset(self, **kwargs):
        """Reset environment and idle arm state."""
        self.idle_arm_last_grip = 0.0  # Start with neutral grip
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """
        Convert 7D agent action to 14D environment action.
        
        Args:
            action: 7D array [pos(3), rot(3), grip(1)]
        
        Returns:
            obs, reward, done, truncated, info from environment
        """
        assert action.shape == (7,), f"Expected 7D action, got {action.shape}"
        
        # Create full 14D action
        full_action = np.zeros(14, dtype=action.dtype)
        
        if self.active_arm == "right":
            # Right arm (indices 7-13) = agent action
            full_action[7:14] = action
            # Left arm (indices 0-6) = neutral (zero deltas, maintain last grip)
            full_action[0:6] = 0.0  # Zero position/rotation deltas
            full_action[6] = self.idle_arm_last_grip  # Maintain grip state
            
        else:  # left arm
            # Left arm (indices 0-6) = agent action  
            full_action[0:7] = action
            # Right arm (indices 7-13) = neutral
            full_action[7:13] = 0.0
            full_action[13] = self.idle_arm_last_grip
        
        # Execute in environment
        obs, reward, done, truncated, info = self.env.step(full_action)
        
        # Update idle arm grip state (in case we need to maintain it)
        # This keeps the idle gripper in its last state
        if self.active_arm == "right":
            self.idle_arm_last_grip = full_action[6]
        else:
            self.idle_arm_last_grip = full_action[13]
        
        return obs, reward, done, truncated, info
