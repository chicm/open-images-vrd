CONFIG=$1
GPUS=$2

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
for i in {0..9}
do
PYTHOHPATH=./ python3  ./mmdetection/tools/train.py --gpus $GPUS $CONFIG ${@:3}
done
