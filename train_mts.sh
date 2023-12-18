#!/bin/bash
mkdir -p checkpoints
# python3 -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 --small
# python3 -u train.py --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/raft-chairs.pth --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001
# python3 -u train.py --name raft-sintel --stage sintel --validation sintel --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --small
# python3 -u train.py --name raft-kitti  --stage kitti --validation kitti --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 1 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85

# python3 -u train.py --name raft-sintel --stage sintel  --validation sintel --gpus 2 3 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 368 496 --wdecay 0.00001 --gamma=0.85 --small
python3 -u train.py --name raft-chairs-small --stage chairs --validation chairs --gpus 0 1 2 3 --num_steps 10000 --batch_size 6 --lr 0.000125 --image_size 368 496 --wdecay 0.00001 --gamma=0.85 --small
