#!/bin/bash

export UV_PROJECT_ENVIRONMENT=/home/ubuntu/.venvs/audio-df
export HF_HOME=/home/ubuntu/.cache/huggingface
export DATA_ROOT=/mnt/salt/datasets/audio-deepfake
export PROJECT_ROOT=/mnt/drive/rohan/audio-deepfake-detection

source /home/ubuntu/.venvs/audio-df/bin/activate

cd "$PROJECT_ROOT"