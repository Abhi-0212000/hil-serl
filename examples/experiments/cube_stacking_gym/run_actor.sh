export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py \
    --exp_name=cube_stacking_gym_rlpd \
    --checkpoint_path=./RLPD_Checkpoints/trial2 \
    --ip=localhost \
    --actor