#!/bin/bash
# You need to modify the dataset path. 
CPICKLE_DIR="..."

# You can to modify to your own workspace. 
WORKSPACE=`pwd`

# Train & predict. 
KERAS_BACKEND=tensorflow-gpu CUDA_VISIBLE_DEVICES=2 python main.py train --cpickle_dir=$CPICKLE_DIR --workspace=$WORKSPACE


# Compute averaged stats. 
KERAS_BACKEND=tensorflow-gpu CUDA_VISIBLE_DEVICES=1 python main.py get_avg_stats --cpickle_dir=$CPICKLE_DIR --workspace=$WORKSPACE
