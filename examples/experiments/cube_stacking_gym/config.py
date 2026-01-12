import os
import jax
import numpy as np
import gymnasium as gym
from trossen_arm_mujoco.utils import make_sim_env
from trossen_arm_mujoco.delta_ee_sim_env_serl_drq_BC import SERLGymWrapper1, CubeStackingEE
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from experiments.config import DefaultTrainingConfig


class TrainConfig(DefaultTrainingConfig):
    # Camera and observation keys - must match your demo format
    image_keys = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    proprio_keys = None
    
    # Training parameters
    batch_size = 256
    max_steps = 20000
    replay_buffer_capacity = 200000  # Must hold all demo transitions
    checkpoint_period = 5000
    encoder_type = "resnet-pretrained"
    bc_checkpoint_path = None  # Will be set from FLAGS
    # setup_mode = "single-arm-fixed-gripper"
    
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        """
        Create simulation environment with wrappers matching demo format.
        
        Demo format (from your inspect output):
            obs = {
                'state': (1, 16),              # Batch dimension from ChunkingWrapper
                'cam_high': (1, 128, 128, 3),  # Images at top level, not in 'images' dict
                'cam_low': (1, 128, 128, 3),
                'cam_left_wrist': (1, 128, 128, 3),
                'cam_right_wrist': (1, 128, 128, 3),
            }
        """
        
        CONTROL_TIMESTEP = 0.02
        PHYSICS_TIMESTEP = 0.002
        cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
        onscreen_render = not fake_env  # Set to True to visualize during eval
        
        # Create dm_control environment
        dm_env = make_sim_env(
            CubeStackingEE,
            task_name="sim_transfer_cube",
            onscreen_render=onscreen_render,
            cam_list=cam_list,
            control_timestep=CONTROL_TIMESTEP,
            physics_timestep=PHYSICS_TIMESTEP,
        )
        
        # Wrap with SERL-compatible wrapper
        env = SERLGymWrapper1(
            env=dm_env,
            state_obs_dim=16,
            max_episode_length=300,
            onscreen_render=onscreen_render,
            cam_list=cam_list,
            action_dim=14,
            time_limit=20.0,
            save_video=save_video,
            action_scale = [0.025, 0.1, 1],
            control_mode="delta",
            video_path=self.bc_checkpoint_path if save_video else None,
        )
        # After this: obs = {'state': (16,), 'images': {'cam_*': (128,128,3)}}

        # Apply wrappers to match your demo format
        # 1. Unpack images from 'images' dict to top level using SERLObsWrapper
        env = SERLObsWrapper(
            env,
            proprio_keys=None  # Will use all state keys (just 'state')
        )
        # Now: obs = {'state': (16,), 'cam_high': (128,128,3), 'cam_low': ..., ...}
        
        # 2. Add temporal dimension (obs_horizon=1 adds batch dim)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # Final: obs = {'state': (1,16), 'cam_high': (1,128,128,3), ...}
        # This now matches your demo format exactly!
        
        return env
    
    def process_demos(self, demo):
        """Process demonstration data if needed."""
        return demo
