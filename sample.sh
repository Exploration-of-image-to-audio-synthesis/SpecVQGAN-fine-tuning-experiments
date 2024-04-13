#!bin/bash

EXPERIMENT_PATH="./logs/2024-04-12T22-08-00_vas_transformer_efficientnet"
SPEC_DIR_PATH="./data/vas/features/*/melspec_10s_22050hz/"
RGB_FEATS_DIR_PATH="./data/vas/features/*/feature_efficientnet_v2_s_dim1280_21.5fps/"
FLOW_FEATS_DIR_PATH=""
SAMPLES_FOLDER="VAS_validation"
SPLITS="\"[validation, ]\""
SAMPLER_BATCHSIZE=4
SAMPLES_PER_VIDEO=10
TOP_K=64 # use TOP_K=512 when evaluating a VAS transformer trained with a VGGSound codebook
NOW=`date +"%Y-%m-%dT%H-%M-%S"`

# Sample
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=62374 \
    --use_env \
        evaluation/generate_samples.py \
        sampler.config_sampler=evaluation/configs/sampler_vas.yaml \
        sampler.model_logdir=$EXPERIMENT_PATH \
        sampler.splits=$SPLITS \
        sampler.samples_per_video=$SAMPLES_PER_VIDEO \
        sampler.batch_size=$SAMPLER_BATCHSIZE \
        sampler.top_k=$TOP_K \
        data.params.spec_dir_path=$SPEC_DIR_PATH \
        data.params.rgb_feats_dir_path=$RGB_FEATS_DIR_PATH \
        data.params.flow_feats_dir_path=$FLOW_FEATS_DIR_PATH \
        sampler.now=$NOW