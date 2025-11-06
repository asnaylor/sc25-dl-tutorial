
#!/bin/bash

image=nersc/pytorch:25.06.01
tp=${1}

echo "Running example with tp=${tp}"

export MASTER_ADDR=$(hostname)
srun --ntasks-per-node ${tp} --gpus-per-node ${tp} -u shifter --image=$image --module=gpu,nccl-plugin \
    bash -c "
    source export_DDP_vars.sh
    python -m tests.make_mlp_tensor_par --tp ${tp}
    "
