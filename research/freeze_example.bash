# From tensorflow/models/research/

# Assumes idirectory structure from tensorflow-models/reseach/object_detection/model_main.py
# -model_dir/
#   |--model.ckpt-??????
#   |--pipeline.config
#
# Puts the frozen model into the directory
#
# -model_dir/
#   |--model.ckpt-??????
#   |--pipeline.config
#   |--frozen/
#       |-frozen_inference_graph.pb
#
# Example usage
#   bash freeze_example.bash -p path/to/model_dir -c 12345


CKPT_NUM="1"
print_usage() {
  printf "Usage: Runs a training, eval, and freezing routine on the model defined by the pipeline config.
        Flags:
            -h: Display this help message.
            -p: Path to pipeline directory containing checkpoint and pipeline.config. Ex: path/to/model_dir
            -c: Checkpoint number. (Default $CKPT_NUM) Ex: For model at path/to/model.ckpt-1234 set -c 1234
"
}

while getopts 'hp:c:' flag; do
  case "${flag}" in
    p) CKPT_DIR="${OPTARG}" ;;
    c) CKPT_NUM="${OPTARG}" ;;
    h) print_usage && exit;;
    *) print_usage && exit
       exit 1 ;;
  esac
done

PIPELINE_CONFIG_PATH=$CKPT_DIR"/pipeline.config"
TRAINED_CKPT_PREFIX=$CKPT_DIR"/model.ckpt-"$CKPT_NUM
EXPORT_DIR=$CKPT_DIR"/frozen"
INPUT_TYPE=image_tensor
INPUT_TYPE=image_tensor

python object_detection/export_inference_graph.py \
     --input_type=${INPUT_TYPE} \
     --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
     --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
     --output_directory=${EXPORT_DIR}

