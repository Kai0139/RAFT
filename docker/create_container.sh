#!/bin/sh
IMAGE_NAME=focal_ros_torch
IMAGE_TAG=root

echo hello

DOCKER_CATKINWS=/home/username/workspace/catkin_ws_ov
DOCKER_DATASETS=/data/kaizhang/raft_data
REPO_DIR=/home/kaizhang/repos

docker create --volume ${DOCKER_DATASETS}:${DOCKER_DATASETS} \
    --volume ${REPO_DIR}:/home/user/ \
    --privileged \
    --net host \
    --gpus all \
    --name raft_root_shm16 \
    --interactive \
    --shm-size=16gb \
    ${IMAGE_NAME}:${IMAGE_TAG}

# end
