# Cube Stacking with RLPD (Right Arm Only)

**Task**: Single-arm cube stacking using delta end-effector control  
**Algorithm**: RLPD (Reinforcement Learning with Prior Data)  
**Robot**: Trossen WidowX (right arm active, left arm frozen)

---

## Quick Start

### 1. Download Dataset

```bash
# Install huggingface-cli
pip install huggingface_hub

# Download dataset
huggingface-cli download poolvarine/trossen-robot-data \
  experiments/cube_stacking_gym/RL/only_right_arm_data_regenerated_deleted.pkl \
  --repo-type dataset \
  --local-dir ./train_data_sets/test1/
```

**Dataset info**: 58 episodes, 9,164 transitions

---

### 2. Run Training

**Terminal 1 - Learner** (GPU):
```bash
cd examples/experiments/cube_stacking_gym
bash run_learner.sh
```

**Terminal 2 - Actor** (environment rollouts):
```bash
cd examples/experiments/cube_stacking_gym
bash run_actor.sh
```

---

## Configuration

### Dataset Path
Edit `run_learner.sh` line 6:
```bash
--demo_path="<path_to_downloaded_pkl_file>"
```

### Checkpoint Path
Both scripts save/load from:
```bash
--checkpoint_path=./RLPD_Checkpoints/trial2
```

### Environment Settings
- **Control**: Delta end-effector (7D: pos 3D + rot 3D + gripper 1D)
- **Frequency**: 50Hz (0.02s timestep)
- **Episode Length**: 158 steps (~3.2s)
- **Cameras**: 4 cameras (high, low, left_wrist, right_wrist)

### Training Hyperparameters
- **Demo sampling**: 50%
- **Online sampling**: 50%
- **Critic:Actor ratio**: 4:1
- **Action scaling**: Position 0.025, Rotation 0.1, Gripper 0.005

---

## Results

**HuggingFace**: https://huggingface.co/datasets/poolvarine/trossen-robot-data/tree/main/experiments/cube_stacking_gym/RL

- Dataset: `only_right_arm_data_regenerated_deleted.pkl`
- Checkpoints: `RLPD_Checkpoints/trial2/`
- Logs: `logs/`

---

## Notes

- **Left arm frozen**: Always stays at home position (safety)
- **Reward stages**: Reach → Grasp → Lift → Align → Place
- **Success threshold**: Reward > 4.5 (requires proper stacking)
- **Physics**: Stiff contacts (no cube penetration exploit)
