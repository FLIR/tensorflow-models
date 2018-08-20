# Sample training script

#PIPELINE_CONFIG_PATH=ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config
#PIPELINE_CONFIG_PATH=ssd_densenet_121_coco.config
#PIPELINE_CONFIG_PATH=ssd_mobilenet_v2_coco.config
PIPELINE_CONFIG_PATH=ssd_inception_v2_coco.config
MODEL_DIR=testnet
NUM_TRAIN_STEPS=1
NUM_EVAL_STEPS=2

python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr

echo "==========================="
echo "Freezing the trained graph"
echo "==========================="
# From tensorflow/models/research/
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=$MODEL_DIR/pipeline.config
TRAINED_CKPT_PREFIX=$MODEL_DIR/model.ckpt-$NUM_TRAIN_STEPS

EXPORT_DIR=test_freeze

python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}