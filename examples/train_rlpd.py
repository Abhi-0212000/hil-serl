#!/usr/bin/env python3

import glob
import time
import json
from datetime import datetime
from functools import partial
from collections import defaultdict
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.logging_utils import RecordEpisodeStatistics
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import get_config


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("render", True, "Render the environment.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    Features:
    - Periodic evaluation with separate eval environment
    - Detailed logging at log_period intervals
    - Eval stats saved to JSON file
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,  # Use argmax for evaluation
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions), copy=True)

                next_obs, reward, done, truncated, info = env.step(actions)
                done = done or truncated
                obs = next_obs

                if done:
                    is_success = info.get("is_success", False)
                    if is_success:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(f"Episode {episode + 1}: SUCCESS in {dt:.2f}s")

                    success_counter += int(is_success)
                    print(f"Success rate so far: {success_counter}/{episode + 1}")

        print(f"\n🎯 Final success rate: {success_counter / FLAGS.eval_n_trajs:.1%}")
        if time_list:
            print(f"⏱️  Average success time: {np.mean(time_list):.2f}s")
        return  # after done eval, return and exit
    
    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path) and glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    # Setup evaluation stats file
    eval_stats_file = None
    if FLAGS.checkpoint_path is not None:
        os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
        eval_stats_file = os.path.join(FLAGS.checkpoint_path, "eval_stats.json")
        # Initialize with empty list
        with open(eval_stats_file, 'w') as f:
            json.dump([], f)
        print(f"📊 Evaluation stats will be saved to: {eval_stats_file}")

    transitions = []
    demo_transitions = []

    print(f"🎯 Actor starting with training env. Calling env.reset()...")
    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    episode_length = 0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                # Scale down random actions to avoid wild movements (20% of max range)
                actions = env.action_space.sample()
                action_source = "🎲 RANDOM"
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions), copy=True)
                action_source = "🤖 AGENT"

        # DETAILED LOGGING: Show action source periodically
        if step % config.log_period == 0 or step < 10:
            print(f"\n[Actor Step {step:6d}] {action_source}")

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            episode_length += 1
            
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            
            # CRITICAL: masks = 1.0 - terminated (NOT 1.0 - done!)
            # terminated = True for task completion (bootstrap = 0)
            # truncated = True for time limit (bootstrap = 1)
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - float(terminated),  # Correct mask for RLPD
                dones=done,
            )
            if transition["masks"] == 0.0:
                print_green(f"[Actor Step {step:6d}] 🚩 Termination detected. Mask=0.0")
            
            if 'grasp_penalty' in info:
                transition['grasp_penalty'] = info['grasp_penalty']
            else:
                transition['grasp_penalty'] = 0.0
                
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            # Log episode termination details
            if done:
                is_success = info.get("is_success", False)
                term_reason = "✅ SUCCESS" if is_success else ("⏱️ TRUNCATED" if truncated else "❌ FAILED")
                
                # RecordEpisodeStatistics already added info["episode"] with "r" and "l"
                # Get stats from RecordEpisodeStatistics
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                ep_last_step_reward = reward
                
                print(f"[Actor Step {step:6d}] 🏁 Episode ended → {term_reason}")
                print(f"                 Episode return: {float(ep_return):.3f}")
                print(f"                 Episode length: {int(ep_length)} steps")
                print(f"                 mask={transition['masks']:.1f}, terminated={terminated}, truncated={truncated}\n")
                print(f"                 Last step reward: {float(ep_last_step_reward):.3f}")
                print(f"                 Total episode reward accumulated: {running_return:.3f} over {episode_length} steps")
                # Add custom fields to existing episode dict
                info["episode"]["is_success"] = is_success
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {float(ep_return):.2f}")
                
                # Reset episode tracking
                running_return = 0.0
                episode_length = 0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Periodic policy evaluation
        if step > 0 and config.eval_period > 0 and step % config.eval_period == 0:
            print(f"\n[Actor Step {step:6d}] 🧪 Starting evaluation...")
            print(f"   Creating fresh eval environment with video recording...")
            
            # Create new eval environment with video recording enabled
            eval_env = config.get_environment(fake_env=False, save_video=True, video_save_path=os.path.join(FLAGS.checkpoint_path, "eval_videos") if FLAGS.checkpoint_path is not None else None, render=True)
            eval_env = RecordEpisodeStatistics(eval_env)
            
            with timer.context("eval"):
                # Use fixed seed for reproducible evaluation
                eval_seed = FLAGS.seed + (step // config.eval_period)
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                    seed=eval_seed,
                )
            
            # Close eval environment to free resources
            eval_env.close()
            print(f"   Closed eval environment")
            
            # Send stats to learner for WandB logging
            eval_stats = {"eval": evaluate_info}
            client.request("send-stats", eval_stats)
            
            # Print evaluation results
            success_rate = evaluate_info.get('final.is_success', 0.0)
            avg_return = evaluate_info.get('eval/average_return', 0.0)
            avg_length = evaluate_info.get('eval/average_length', 0)
            
            print(f"[Actor Step {step:6d}] ✅ Evaluation complete:")
            print(f"   • Success rate: {success_rate:.1%}")
            print(f"   • Avg return: {avg_return:.3f}")
            print(f"   • Avg length: {avg_length:.1f} steps")
            
            # Save to JSON file
            if eval_stats_file is not None:
                eval_record = {
                    "step": step,
                    "timestamp": datetime.now().isoformat(),
                    "success_rate": float(success_rate),
                    "average_return": float(avg_return),
                    "average_length": float(avg_length),
                    "full_stats": {k: float(v) if isinstance(v, (np.number, np.floating, np.integer)) else v 
                                   for k, v in evaluate_info.items()}
                }
                
                # Load existing stats, append new one, save
                try:
                    with open(eval_stats_file, 'r') as f:
                        all_stats = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    all_stats = []
                all_stats.append(eval_record)
                with open(eval_stats_file, 'w') as f:
                    json.dump(all_stats, f, indent=2)
                print(f"   • Saved stats to: {eval_stats_file}\n")
            else:
                print()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    # Counters for tracking updates
    total_critic_only_updates = 0
    total_critic_actor_updates = 0
    
    print_green(f"\n🎯 Starting training with CTA_RATIO={config.cta_ratio}")
    print_green(f"   • Critic-only updates per step: {config.cta_ratio - 1}")
    print_green(f"   • Critic+Actor updates per step: 1\n")

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # ========================================================================
        # CRITIC-ONLY UPDATES (CTA_RATIO - 1 updates)
        # Purpose: Update value networks without policy changes
        # ========================================================================
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                online_batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                
                # ========================================================================
                # BATCH COMPOSITION LOGGING (Every log_period steps)
                # ========================================================================
                if step % config.log_period == 0 and critic_step == 0:
                    print(f"\n{'='*80}")
                    print(f"[LEARNER Step {step:6d}] BATCH ANALYSIS")
                    print(f"{'='*80}")
                    
                    # --- Batch Sizes ---
                    online_size = online_batch['actions'].shape[0]
                    demo_size = demo_batch['actions'].shape[0]
                    print(f"\n📦 BATCH SIZES:")
                    print(f"   Online: {online_size:3d} | Demo: {demo_size:3d} | Total: {online_size + demo_size:3d}")
                    
                    # --- Camera Images Check ---
                    # NOTE: SERLObsWrapper unwraps images to top level (not nested under 'images')
                    print(f"\n📸 CAMERA IMAGES (Verifying all 4 cameras):")
                    for batch_name, batch_data in [("Online", online_batch), ("Demo", demo_batch)]:
                        obs = batch_data['observations']
                        print(f"   {batch_name} batch images:")
                        cam_count = 0
                        for cam_name in ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']:
                            if cam_name in obs:
                                img_shape = obs[cam_name].shape
                                img_mean = float(jnp.mean(obs[cam_name]))
                                img_std = float(jnp.std(obs[cam_name]))
                                print(f"      ✓ {cam_name:18s}: shape={img_shape}, mean={img_mean:6.2f}, std={img_std:5.2f}")
                                cam_count += 1
                            else:
                                print(f"      ✗ {cam_name:18s}: MISSING!")
                        print(f"   → {batch_name} total cameras found: {cam_count}/4")
                    
                    # --- Masks/Dones Analysis ---
                    online_masks = online_batch['masks']
                    demo_masks = demo_batch['masks']
                    print(f"\n🎭 MASKS (Bootstrapping Signal):")
                    print(f"   Online: mean={float(jnp.mean(online_masks)):.3f}, min={float(jnp.min(online_masks)):.3f}, max={float(jnp.max(online_masks)):.3f}")
                    print(f"   Demo:   mean={float(jnp.mean(demo_masks)):.3f}, min={float(jnp.min(demo_masks)):.3f}, max={float(jnp.max(demo_masks)):.3f}")
                    
                    # --- Rewards Analysis ---
                    online_rewards = online_batch['rewards']
                    demo_rewards = demo_batch['rewards']
                    print(f"\n🎁 REWARDS:")
                    print(f"   Online: mean={float(jnp.mean(online_rewards)):.4f}, min={float(jnp.min(online_rewards)):.4f}, max={float(jnp.max(online_rewards)):.4f}")
                    print(f"   Demo:   mean={float(jnp.mean(demo_rewards)):.4f}, min={float(jnp.min(demo_rewards)):.4f}, max={float(jnp.max(demo_rewards)):.4f}")
                
                batch = concat_batches(online_batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )
                total_critic_only_updates += 1

        # ========================================================================
        # CRITIC + ACTOR UPDATE (1 update per step)
        # Purpose: Update both value networks AND policy
        # ========================================================================
        with timer.context("train"):
            online_batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            
            batch = concat_batches(online_batch, demo_batch, axis=0)
            
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
            total_critic_actor_updates += 1
        
        # ========================================================================
        # TRAINING METRICS LOGGING (Every log_period steps)
        # ========================================================================
        if step % config.log_period == 0:
            print(f"\n{'='*80}")
            print(f"[LEARNER Step {step:6d}] TRAINING METRICS")
            print(f"{'='*80}")
            
            # --- Update Counts ---
            update_ratio = total_critic_only_updates / (total_critic_actor_updates + 1e-8)
            print(f"\n📊 UPDATE STATISTICS:")
            print(f"   Critic-only updates: {total_critic_only_updates:7d}")
            print(f"   Critic+Actor updates: {total_critic_actor_updates:7d}")
            print(f"   Ratio: {update_ratio:.2f} (expected: {config.cta_ratio-1:.2f})")
            
            # --- Buffer Status ---
            print(f"\n💾 BUFFER STATUS:")
            print(f"   Replay buffer: {len(replay_buffer):6d} / {replay_buffer._capacity:6d} ({100*len(replay_buffer)/replay_buffer._capacity:.1f}%)")
            print(f"   Demo buffer:   {len(demo_buffer):6d} / {demo_buffer._capacity:6d} ({100*len(demo_buffer)/demo_buffer._capacity:.1f}%)")
            print(f"{'='*80}\n")
            
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            # Add update counts to wandb
            update_info_extended = {
                **update_info,
                "training/critic_only_updates": total_critic_only_updates,
                "training/critic_actor_updates": total_critic_actor_updates,
                "training/update_ratio": total_critic_only_updates / (total_critic_actor_updates + 1e-8),
            }
            wandb_logger.log(update_info_extended, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    config = get_config(FLAGS.exp_name)()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=True if FLAGS.eval_checkpoint_step > 0 else False,
        classifier=True,
        video_save_path=os.path.join(FLAGS.checkpoint_path, "eval_videos") if FLAGS.checkpoint_path is not None else None,
        render=FLAGS.render,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            critic_ensemble_size=config.critic_ensemble_size,
            critic_subsample_size=config.critic_subsample_size,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        # Check if there are actual checkpoint files
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt is not None:
            input("Checkpoint path already exists. Press Enter to resume training.")
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
            )
            agent = agent.replace(state=ckpt)
            ckpt_number = os.path.basename(latest_ckpt)[11:]
            print_green(f"Loaded previous checkpoint at step {ckpt_number}.")
        else:
            print_green(f"Checkpoint directory exists but is empty. Starting fresh training.")
            # Create directory if it doesn't exist
            os.makedirs(FLAGS.checkpoint_path, exist_ok=True)

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    # Handle grasp_penalty for hybrid agents
                    if include_grasp_penalty:
                        if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                            transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                        else:
                            # For BC demos without grasp_penalty, set to 0 (no penalty)
                            transition['grasp_penalty'] = 0.0
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
