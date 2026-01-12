import os
import jax
import numpy as np
import gymnasium as gym
from trossen_arm_mujoco.utils import make_sim_env
# from trossen_arm_mujoco.delta_ee_sim_env_serl_drq_RL import SERLGymWrapper1, CubeStackingEE
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from experiments.config import DefaultTrainingConfig
from trossen_arm_mujoco.envs.cube_stacking.single_arm_env import SERLGymWrapper, CubeStackingEE

class TrainConfig(DefaultTrainingConfig):
    # Camera and observation keys - must match your demo format
    image_keys = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    proprio_keys = None
    
    # Encoder
    encoder_type = "resnet-pretrained"
    
    # ================ RLPD-specific settings: ================
    setup_mode = "single-arm-fixed-gripper"  # Both grippers learned by grasp critic
    
    # Training hyperparams (from ram_insertion example)
    max_steps = 150000  # Total RL training steps 200K
    training_starts = 1000  # Start RL after N transitions
    random_steps = 1000  # Initial random steps for better coverage
    cta_ratio = 4  # Critic-to-actor update ratio (increased from 4 to stabilize critics)
    batch_size = 128  # Reduced from 256 to fit GPU memory (4 cameras + ResNet)
    discount = 0.96  # Changed back to 0.99 (0.97 too low for 400-step episodes)
    
    # Replay buffer sizes (reduced to prevent OOM)
    replay_buffer_capacity = 150000  
    demo_buffer_capacity = 15000     # Reduced from 50k (demos stay in separate buffer)
    
    # Checkpointing
    checkpoint_period = 5000
    steps_per_update = 30  # Publish updated policy every 50 steps

    # actor configs
    buffer_period: int = 3000
    actor_step_freq: int = 9
    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5
    eval_period = 3000  # Periodic evaluation frequency (0 to disable)

    critic_ensemble_size = 10
    critic_subsample_size = 2

    # Logging
    log_period=100

    # =========================================================

    def get_environment(self, fake_env=False, save_video=False, classifier=False, video_save_path=None, render=False):
        """
        Create Trossen simulation environment with wrappers matching demo format.
        
        Args:
            fake_env: If True, create dummy env for learner (fast startup)
            save_video: If True, record episode videos
            classifier: Unused, kept for compatibility
            render: If True, enable onscreen rendering (only for actor)
            
        Returns:
            Wrapped environment with observation format:
            obs = {
                'state': (1, 8),               # RIGHT ARM ONLY: [mocap_pose_right(7), gripper(1)]
                'cam_high': (1, 128, 128, 3),
                'cam_low': (1, 128, 128, 3),
                'cam_left_wrist': (1, 128, 128, 3),
                'cam_right_wrist': (1, 128, 128, 3),
            }
        """
        
        CONTROL_TIMESTEP = 0.02
        PHYSICS_TIMESTEP = 0.002
        MAX_EPISODE_LENGTH = 187  # Matches regenerate_ds_state_from_obs() calculation
        TIME_LIMIT = 20.0
        
        cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
        
        # Video path setup
        video_path = None
        if save_video:
            # Use checkpoint_path directory for videos if available, otherwise create default
            if video_save_path is not None:
                video_dir = video_save_path
            else:
                video_dir = os.path.join(os.getcwd(), "eval_videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = video_dir
            print(f"📹 Video recording enabled → saving to: {video_path}")
        
        # Visualization setup (only for actor with render=True)
        onscreen_render = render and not fake_env
        
        if fake_env:
            print("🔧 Creating FAKE environment (learner mode - fast startup)")
            # For learner: create minimal fake env without MuJoCo initialization
            env = SERLGymWrapper(
                env=None,  # No actual dm_env needed
                state_obs_dim=8,
                fake_env=True,
                image_obs=True,
                max_episode_length=MAX_EPISODE_LENGTH,
                action_dim=7,
                time_limit=TIME_LIMIT,
                action_scale=[0.025, 0.1, 0.005],
                control_mode="delta",
            )
        else:
            print("🔧 Creating REAL environment (actor mode - full physics)")
            # For actor: create full dm_control environment
            dm_env = make_sim_env(
                CubeStackingEE,
                task_name="sim_transfer_cube",
                onscreen_render=onscreen_render,
                cam_list=cam_list,
                control_timestep=CONTROL_TIMESTEP,
                physics_timestep=PHYSICS_TIMESTEP,
            )
            
            env = SERLGymWrapper(
                env=dm_env,
                state_obs_dim=8,  # RIGHT ARM ONLY
                fake_env=False,
                image_obs=True,
                max_episode_length=MAX_EPISODE_LENGTH,
                onscreen_render=onscreen_render,
                cam_list=cam_list,
                action_dim=7,  # 7D delta actions (pos=3, rot=3, grip=1)
                time_limit=TIME_LIMIT,
                save_video=save_video,
                video_path=video_path,
                action_scale=[0.025, 0.1, 0.005],
                control_mode="delta",
                plot_name="eval_env" if save_video else "training_env",  # Different names for eval vs training
            )
            print(f"   • Episode length: {MAX_EPISODE_LENGTH} steps × {CONTROL_TIMESTEP}s = {MAX_EPISODE_LENGTH * CONTROL_TIMESTEP:.2f}s")
            print(f"   • Time limit: {TIME_LIMIT}s → terminated=False, truncated=True (correct!)")
        
        # Apply wrappers to match demo format
        # 1. Unpack images from 'images' dict to top level
        env = SERLObsWrapper(env, proprio_keys=None)
        # Now: obs = {'state': (8,), 'cam_high': (128,128,3), ...}
        
        # Add temporal dimension (obs_horizon=1 adds batch dim)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # Final: obs = {'state': (1,8), 'cam_high': (1,128,128,3), ...}
        
        print("✅ Environment created and wrapped successfully")
        return env
    
    def process_demos(self, demo):
        """Process demonstration data if needed."""
        return demo
