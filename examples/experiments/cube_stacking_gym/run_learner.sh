export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
python ../../train_rlpd.py \
    --exp_name=cube_stacking_gym_rlpd \
    --checkpoint_path=./t/ \
    --demo_path="/home/qte9489/personal_abhi/temp/hil-serl/examples/experiments/cube_stacking_gym/Trossen_windowX_DualRobot_only_right_arm_cube_stacking_regenerated_deleted_extra.pkl" \
    --learner