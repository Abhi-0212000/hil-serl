export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
python ../../train_rlpd.py \
    --exp_name=cube_stacking_gym_rlpd \
    --checkpoint_path=./RLPD_Checkpoints/trial2 \
    --demo_path="./train_data_sets/test1/experiments/cube_stacking_gym/RL/only_right_arm_data_regenerated_deleted.pkl" \
    --learner