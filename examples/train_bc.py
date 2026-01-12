#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import shutil
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.bc import BCAgent

from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.utils.logging_utils import pretty_print_nested_dict
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import get_config
from experiments.config import DefaultTrainingConfig
from experiments.cube_stacking_gym.config import TrainConfig
import json
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("demo_path", None, "Path to demo pickle file(s). Can use wildcards.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 20_000, "Number of pretraining steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")


flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


##############################################################################

def eval(
    env,
    bc_agent: BCAgent,
    sampling_rng,
    seed=42,
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    success_counter = 0
    time_list = []
    eval_metrics = {}
    
    print(f"Starting evaluation over {FLAGS.eval_n_trajs} episodes...")

    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset(seed = seed)
        done = False
        start_time = time.time()
        
        while not done:
            sampling_rng, key = jax.random.split(sampling_rng)

            # Sample actions
            actions = bc_agent.sample_actions(observations=obs, seed=key, argmax=True)
            
            # Convert JAX array to numpy array for compatibility with env.step()
            actions = np.array(actions)

            next_obs, reward, done, truncated, info = env.step(actions)
            obs = next_obs
            
            if done:
                # --- NEW USER INTERVENTION CODE ---
                print(f"\n--- Episode {episode + 1} Ended ---")
                print(f"Auto-Reward from Env: {reward}")

                # Only ask for input if the auto-reward is 0 (failure)
                if reward == 0:
                    user_input = input("Env reported failure. Enter new reward (e.g. 1 or 0.9) or hit Enter to keep 0: ").strip()
                    
                    if user_input:
                        try:
                            new_reward = float(user_input)
                            reward = new_reward
                            print(f">>> Manual Override: Reward set to {reward}")
                        except ValueError:
                            print(">>> Invalid input. Reward kept at 0.0")
                    else:
                        print(">>> Reward kept at 0.0")
                    eval_metrics[f"episode_{episode + 1}_reward"] = reward
                    eval_metrics[f"episode_{episode + 1}_time"] = time.time() - start_time
                    eval_metrics[f"manual_reward_added"] = True
                else:
                    print(">>> Auto-success confirmed.")
                    eval_metrics[f"episode_{episode + 1}_reward"] = reward
                    eval_metrics[f"episode_{episode + 1}_time"] = time.time() - start_time
                # ----------------------------------

                if reward > 0:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(f"Time taken: {dt:.4f}s")
                
                success_counter += reward
                print(f"Running Score: {success_counter}/{episode + 1}")

    print("\n=== FINAL RESULTS ===")
    success_rate = success_counter / FLAGS.eval_n_trajs
    print(f"Success Rate: {success_rate}")
    eval_metrics["success_rate"] = success_rate
    eval_metrics["num_successes"] = success_counter
    eval_metrics["seed for evaluation"] = seed
    if time_list:
        eval_metrics["average_time"] = np.mean(time_list)
        print(f"Average Time: {eval_metrics['average_time']}")
    else:
        print("Average Time: N/A (No successes)")
    return eval_metrics

##############################################################################


def train(
    bc_agent: BCAgent,
    bc_replay_buffer,
    config: DefaultTrainingConfig,
    wandb_logger=None,
):

    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )

    # Track best checkpoints: list of (step, success_rate) tuples
    best_checkpoints = []
    eval_interval = 5000
    eval_n_trajs = 10
    
    # Pretrain BC policy to get started
    for step in tqdm.tqdm(
        range(FLAGS.train_steps + 1),
        dynamic_ncols=True,
        desc="bc_pretraining",
    ):
        batch = next(bc_replay_iterator)
        bc_agent, bc_update_info = bc_agent.update(batch)
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log({"bc": bc_update_info}, step=step)
        
        # Periodic evaluation
        if step % eval_interval == 0 and step > 0:
            print_yellow(f"\n{'='*60}")
            print_yellow(f"Running evaluation at step {step}...")
            print_yellow(f"{'='*60}")
            
            # Save checkpoint
            checkpoint_path = os.path.abspath(FLAGS.bc_checkpoint_path)
            checkpoints.save_checkpoint(
                checkpoint_path, bc_agent.state, step=step, keep=1000000  # Don't auto-delete during training
            )
            
            # Create eval environment
            eval_env = config.get_environment(
                fake_env=False,
                save_video=False,
                classifier=True,
            )
            eval_env = RecordEpisodeStatistics(eval_env)
            
            # Load checkpoint for evaluation
            bc_ckpt = checkpoints.restore_checkpoint(
                checkpoint_path,
                bc_agent.state,
                step=step,
            )
            eval_agent = bc_agent.replace(state=bc_ckpt)
            
            # Run evaluation
            rng = jax.random.PRNGKey(FLAGS.seed + step)  # Different seed per eval
            sampling_rng = jax.device_put(rng, sharding.replicate())
            
            success_counter = 0
            episode_times = []
            
            for episode in range(eval_n_trajs):
                obs, _ = eval_env.reset()
                done = False
                start_time = time.time()
                
                while not done:
                    rng, key = jax.random.split(sampling_rng)
                    actions = eval_agent.sample_actions(observations=obs, seed=key, argmax=True)
                    actions = np.array(actions)
                    next_obs, reward, done, truncated, info = eval_env.step(actions)
                    obs = next_obs
                    
                    if done:
                        if reward > 0:
                            dt = time.time() - start_time
                            episode_times.append(dt)
                            success_counter += reward
            
            # Calculate success rate (handle division by zero)
            success_rate = float(success_counter) / eval_n_trajs if eval_n_trajs > 0 else 0.0
            avg_time = float(np.mean(episode_times)) if len(episode_times) > 0 else 0.0
            
            print_green(f"Step {step} - Success Rate: {success_rate:.2%} ({success_counter}/{eval_n_trajs})")
            if len(episode_times) > 0:
                print_green(f"Step {step} - Average Time: {avg_time:.2f}s")
            
            # Log to wandb
            if wandb_logger:
                wandb_logger.log({
                    "eval/success_rate": success_rate,
                    "eval/avg_time": avg_time,
                    "eval/num_successes": success_counter,
                }, step=step)
            
            # Update best checkpoints list
            best_checkpoints.append((step, success_rate))
            best_checkpoints.sort(key=lambda x: x[1], reverse=True)  # Sort by success rate (descending)
            
            # Keep only top 10 checkpoints
            if len(best_checkpoints) > 10:
                # Remove the worst checkpoint file
                worst_step, worst_rate = best_checkpoints.pop()
                worst_ckpt_dir = os.path.join(checkpoint_path, f"checkpoint_{worst_step}")
                if os.path.exists(worst_ckpt_dir):
                    import shutil
                    shutil.rmtree(worst_ckpt_dir)
                    print_yellow(f"Removed checkpoint at step {worst_step} (success rate: {worst_rate:.2%})")
            
            print_green(f"Current best checkpoints (top {len(best_checkpoints)}):")
            for rank, (ckpt_step, ckpt_rate) in enumerate(best_checkpoints[:10], 1):
                print_green(f"  #{rank}: Step {ckpt_step} - {ckpt_rate:.2%}")
            
            # Clean up eval environment
            eval_env.close()
            del eval_env
            del eval_agent
            
            print_yellow(f"{'='*60}\n")
        
        # Save final checkpoints
        # if step > FLAGS.train_steps - 100 and step % 10 == 0:
        #     checkpoints.save_checkpoint(
        #         os.path.abspath(FLAGS.bc_checkpoint_path), bc_agent.state, step=step, keep=10
        #     )
    
    print_green("bc pretraining done and saved checkpoint")
    print_green(f"\nFinal best checkpoints:")
    for rank, (ckpt_step, ckpt_rate) in enumerate(best_checkpoints[:10], 1):
        print_green(f"  #{rank}: Step {ckpt_step} - {ckpt_rate:.2%}")


##############################################################################


def main(_):
    config: DefaultTrainingConfig = get_config(FLAGS.exp_name)()
    
    # Set checkpoint path in config for video saving
    if hasattr(config, 'bc_checkpoint_path'):
        config.bc_checkpoint_path = FLAGS.bc_checkpoint_path

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name, "Experiment name must be provided."
    eval_mode = FLAGS.eval_n_trajs > 0
    env = config.get_environment(
        fake_env=not eval_mode,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    bc_agent: BCAgent = make_bc_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    bc_agent: BCAgent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, bc_agent), sharding.replicate()
    )

    if not eval_mode:
        assert not os.path.isdir(
            os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")
        )

        bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
        )

        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        # Load demos from command line path or default location
        if FLAGS.demo_path:
            demo_path = glob.glob(FLAGS.demo_path)
        else:
            # Default: look in demo_data/ folder
            demo_path = glob.glob(os.path.join(os.getcwd(), "demo_data", "*.pkl"))
        
        assert demo_path != [], f"No demo files found! Searched: {FLAGS.demo_path or 'demo_data/*.pkl'}"

        # Import function to get episode boundaries
        import sys
        sys.path.insert(0, os.path.join(os.getcwd(), 'trossen_arm_mujoco'))
        from trossen_arm_mujoco.dataset_utils.delta_action_ds.cleanup_utils import get_episode_boundaries
        
        # Specify which episodes to use for BC training (None = all episodes)
        selected_episodes = None  # e.g., [0, 1, 2, 5, 10] or None for all episodes
        
        for path in demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                
                # Get episode boundaries
                episode_boundaries = get_episode_boundaries(transitions)
                print(f"Found {len(episode_boundaries)} episodes in {path}")
                
                # Filter episodes if specific ones are selected
                if selected_episodes is not None:
                    filtered_transitions = []
                    for ep_idx in selected_episodes:
                        if ep_idx < len(episode_boundaries):
                            start, end = episode_boundaries[ep_idx]
                            filtered_transitions.extend(transitions[start:end])
                        else:
                            print(f"Warning: Episode {ep_idx} does not exist (max: {len(episode_boundaries)-1})")
                    transitions = filtered_transitions
                    print(f"Using {len(selected_episodes)} selected episodes with {len(transitions)} transitions")
                
                # Insert transitions into buffer
                for transition in transitions:
                    if np.linalg.norm(transition['actions']) > 0.0:
                        bc_replay_buffer.insert(transition)
        print(f"bc replay buffer size: {len(bc_replay_buffer)}")

        # learner loop
        print_green("starting learner loop")
        train(
            bc_agent=bc_agent,
            bc_replay_buffer=bc_replay_buffer,
            wandb_logger=wandb_logger,
            config=config,
        )

    else:
            import os  # Ensure os is imported

            rng = jax.random.PRNGKey(FLAGS.seed)
            sampling_rng = jax.device_put(rng, sharding.replicate())

            # 1. Get list of all checkpoint folders
            base_path = FLAGS.bc_checkpoint_path
            all_items = os.listdir(base_path)
            
            # Filter: Must be a directory AND start with "checkpoint_"
            ckpt_folders = [
                d for d in all_items 
                if os.path.isdir(os.path.join(base_path, d)) and d.startswith("checkpoint_")
            ]

            # 2. Sort them numerically by step number (e.g., checkpoint_100 before checkpoint_200)
            # This handles the string sorting issue where '1000' comes before '200'
            def get_step_num(name):
                try:
                    return int(name.split('_')[-1])
                except ValueError:
                    return float('inf') # Put malformed names at the end
            
            ckpt_folders.sort(key=get_step_num)

            print(f"Found {len(ckpt_folders)} checkpoints to evaluate: {ckpt_folders}")

            total_eval_metrics = {}

            print(f"checkpoint folders to evaluate: {ckpt_folders}")

            # 3. Iterate and Evaluate
            for ckpt_name in ckpt_folders[]:
                seed = FLAGS.seed 
                full_ckpt_path = os.path.join(base_path, ckpt_name)
                
                print_green(f"\n==================================================")
                print_green(f"Evaluating Checkpoint: {ckpt_name}")
                print_green(f"Path: {full_ckpt_path}")
                print_green(f"==================================================")

                # Restore the specific checkpoint
                # We use bc_agent.state as the template structure
                loaded_state = checkpoints.restore_checkpoint(
                    full_ckpt_path,
                    bc_agent.state,
                )
                
                # Update video path by unwrapping to base environment
                # Use .unwrapped to get to the innermost environment (SERLGymWrapper1)
                base_env = env.unwrapped
                base_env.video_path = full_ckpt_path
                base_env._video_run_dir = None  # Reset cached directory so it recreates on next reset()
                print_yellow(f"[Debug] Updated video_path to: {base_env.video_path}")
                
                # Create a temporary agent with these loaded weights for evaluation
                eval_agent = bc_agent.replace(state=loaded_state)

                checkpoint_eval_metrics = eval(
                    env=env,
                    bc_agent=eval_agent,
                    sampling_rng=sampling_rng,
                    seed=seed,
                )
                total_eval_metrics[ckpt_name] = checkpoint_eval_metrics

            print_green(f"\n================= EVALUATION SUMMARY =================")
            pretty_print_nested_dict(total_eval_metrics)
            with open(f"{base_path}/bc_evaluation_summary_{seed}.json", "w") as f:
                json.dump(total_eval_metrics, f, indent=4)
            print_green(f"\n=====================================================")

if __name__ == "__main__":
    app.run(main)


"""
python examples/train_bc.py \
    --exp_name=cube_stacking_gym \
    --demo_path="/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_fulldelta_act_scaled_act_epsilon_binarized_regenerated_deleted_episodes_epsilon.pkl" \
    --bc_checkpoint_path=/home/qte9489/personal_abhi/temp/hil-serl/bc_checkpoints/cube_stacking/random_cube_pose/v6_properly_cleaned \
    --train_steps=50000


python examples/train_bc.py \
    --exp_name=cube_stacking_gym \
    --bc_checkpoint_path=/home/qte9489/personal_abhi/temp/hil-serl/bc_checkpoints/cube_stacking/random_cube_pose/v6_properly_cleaned \
    --eval_n_trajs=10 \
    --save_video


    > actor_debug_log.txt 2>&1



# SUCCESS DS and CHECKPOINT
1. python examples/train_bc.py \
    --exp_name=cube_stacking_gym \
    --demo_path="/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_binarized_gripper.pkl" \
    --bc_checkpoint_path=./bc_checkpoints/cube_stacking/full_scaled_delta_actions \
    --train_steps=50000

    Final best checkpoints:
    #1: Step 10000 - 100.00%
    #2: Step 40000 - 100.00%
    #3: Step 20000 - 0.00%
    #4: Step 30000 - 0.00%

    run = nannuriabhi2000-hochschule-schmalkalden/hil-serl/runs/cube_stacking_gym_20260103_132521

    checkpoints: 10k, 40k, 40k+

2. python examples/train_bc.py \
    --exp_name=cube_stacking_gym \
    --demo_path="/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_binarized_gripper_regenerated_states_FINAL.pkl" \
    --bc_checkpoint_path=./bc_checkpoints/cube_stacking/full_scaled_delta_actions_regenerated_states \
    --train_steps=50000

    Final best checkpoints:
    #1: Step 10000 - 100.00%
    #2: Step 20000 - 100.00%
    #3: Step 30000 - 100.00%
    #4: Step 40000 - 0.00%
    #5: Step 50000 - 0.00%

    run = nannuriabhi2000-hochschule-schmalkalden/hil-serl/runs/cube_stacking_gym_20260103_235539

3. python examples/train_bc.py \
    --exp_name=cube_stacking_gym \
    --demo_path="/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped_delta_act_scaled_epsilon_binarized_gripper_regenerated_states_FINAL.pkl" \
    --bc_checkpoint_path=./bc_checkpoints/cube_stacking/random_cube_pose/v2 \
    --train_steps=100000

     bc pretraining done and saved checkpoint
        Final best checkpoints:
        #1: Step 10000 - 0.00%
        #2: Step 20000 - 0.00%
        #3: Step 30000 - 0.00%
        #4: Step 40000 - 0.00%
        #5: Step 50000 - 0.00%
        #6: Step 60000 - 0.00%
        #7: Step 70000 - 0.00%
        #8: Step 80000 - 0.00%
        #9: Step 90000 - 0.00%
        #10: Step 100000 - 0.00%
        wandb: 
        wandb: 🚀 View run cube_stacking_gym_20260106_042840 at: 

4. 
    --demo_path="/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_fulldelta_act_scaled_act_epsilon_binarized_regenerated_deleted_episodes.pkl" \
    run = cube_stacking_gym_20260106_205302
    action_scale = [0.025, 0.1, 0.02]
    NOT REALLY GOOD RESULTS (Trained for 30k steps and 5k checkpoint got 10% success rate, others 0%)

    python examples/train_bc.py \
        --exp_name=cube_stacking_gym \
        --demo_path="/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_fulldelta_act_scaled_act_epsilon_binarized_regenerated_deleted_episodes_epsilon.pkl" \
        --bc_checkpoint_path=./bc_checkpoints/cube_stacking/random_cube_pose/v6_properly_cleaned \
        --train_steps=50000    
    run = cube_stacking_gym_20260106_221338
    action_scale = [0.025, 0.1, 1]
    SLIGHTLY BETTER RESULTS
    Final best checkpoints:
    #1: Step 20000 - 100.00%
    #2: Step 40000 - 70.00%
    #3: Step 35000 - 60.00%
    #4: Step 10000 - 50.00%
    #5: Step 45000 - 30.00%
    #6: Step 15000 - 20.00%
    #7: Step 5000 - 10.00%
    #8: Step 30000 - 10.00%
    #9: Step 25000 - 0.00%
    #10: Step 50000 - 0.00%
    wandb: 
    wandb: 🚀 View run cube_stacking_gym_20260106_221338 at: 



"""