export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_PORT=29500 # default from torch launcher
export CUDA_DEVICE_MAX_CONNECTIONS=1 # TE suggests this for good comm overlap
export NVTE_BATCH_MHA_P2P_COMM=1 # for CP in TE for unusual sizes