# Sample training script

# Set the gpus to use
user_gpus="$@"
cnd_gpus=
for gpu in $user_gpus
do 
    cnd_gpus=$cnd_gpus,$gpu
done
cnd_gpus=${cnd_gpus#,}

# Set the pipeline config
# This should have everything to run train and evaluation.
PIPELINE_CONFIG_PATH=ssd_inception_v2_coco.config

MODEL_DIR=inception_v2_coco
NUM_TRAIN_STEPS=10
NUM_EVAL_STEPS=2

if [ -d "$MODEL_DIR" ]; then
  echo "Model directory exists"
  echo "THIS MAY CAUSE LOADING ISSUES AND STOP THE MODEL FROM TRAINING"
fi


CUDA_VISIBLE_DEVICES=$cnd_gpus python object_detection/model_main.py \
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