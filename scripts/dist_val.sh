PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

PYTHONPATH=./ $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    ./mmdetection/tools/eval.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
