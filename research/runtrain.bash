# Sample training script
# From tensorflow/models/research/

# Setting Default parameters
PIPELINE_CONFIG_PATH=ssd_inception_v2_adas.config
MODEL_DIR=testnet
NUM_TRAIN_STEPS=200100
NUM_EVAL_STEPS=3100
gpus=0

print_usage() {
  printf "Usage: Runs a training, eval, and freezing routine on the model defined by the pipeline config.
        Flags:
            -h: Display this help message.
            -g: The name of gpus to use. Should be seperated by comma with no spaces.(Default $gpus)
            -t: Number of training steps. (Default $NUM_TRAIN_STEPS)
            -e: Number of eval steps. (Default $NUM_EVAL_STEPS)
            -p: Path to pipeline config. (Default $PIPELINE_CONFIG_PATH)
            -m: Model name for the directory of chkpts and such. (Default $MODEL_DIR)
            "
}

while getopts 'hg:t:e:m:p:' flag; do
  case "${flag}" in
    g) gpus="${OPTARG}" ;;
    t) NUM_TRAIN_STEPS="${OPTARG}";;
    e) NUM_EVAL_STEPS="${OPTARG}" ;;
    m) MODEL_DIR="${OPTARG}" ;;
    p) PIPELINE_CONFIG_PATH="${OPTARG}" ;;
    h) print_usage ;;
    *) print_usage
       exit 1 ;;
  esac
done

echo "gpus: $gpus"
echo "train_steps: "$NUM_TRAIN_STEPS
echo "eval_steps: "$NUM_EVAL_STEPS
echo "model: "$MODEL_DIR

if [ -d "$MODEL_DIR" ]; then
  echo "Model directory exists"
  echo "THIS MAY CAUSE LOADING ISSUES AND STOP THE MODEL FROM TRAINING"
fi

CUDA_VISIBLE_DEVICES=$gpus python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr

echo "==========================="
echo "Freezing the trained graph"
echo "==========================="

# Set input type to image
INPUT_TYPE=image_tensor
EXPORT_DIR=$MODEL_DIR"_freeze"

# Pick the most recent config which comes from training
PIPELINE_CONFIG_PATH=$MODEL_DIR/pipeline.config
TRAINED_CKPT_PREFIX=$MODEL_DIR/model.ckpt-$NUM_TRAIN_STEPS

CUDA_VISIBLE_DEVICES=$cnd_gpus python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}