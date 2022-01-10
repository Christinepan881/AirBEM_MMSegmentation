PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    $(dirname "$0")/train.py \
    configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_20k_airbem.py \
    --launcher pytorch \
    --work-dir /data1/chenbin/AirBEM_mm_workdir/deeplabv3plus_r101-d8_512x512_20k_airbem
