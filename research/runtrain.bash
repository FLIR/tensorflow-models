# Sample training script

PIPELINE_CONFIG_PATH=ssd_densenet_121_coco.config
MODEL_DIR=densenet121_test
NUM_TRAIN_STEPS=50
NUM_EVAL_STEPS=20

python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr
