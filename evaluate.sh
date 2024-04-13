EXPERIMENT_PATH="./logs/"
SPEC_DIR_PATH="./data/vas/features/*/melspec_10s_22050hz/"
RGB_FEATS_DIR_PATH="./data/vas/features/*/feature_efficientnet_v2_s_dim1280_21.5fps/"
SAMPLES_FOLDER="VAS_validation"
SPLITS="\"[validation, ]\""
SAMPLER_BATCHSIZE=4
SAMPLES_PER_VIDEO=10
TOP_K=64 # use TOP_K=512 when evaluating a VAS transformer trained with a VGGSound codebook
NOW=`date +"%Y-%m-%dT%H-%M-%S"`