# @jroberts: 2/2019
#   The main script for running training of a tf object detector.
#   Requires the data to be saved in a tfrecord format and a
#   object detection pipeline config file. An example of one
#   is in this directory called "flir_sample.config"
#
#   Example Usage:
#     bash runtrain.bash -g 0,1 \\
#                        -p path/to/pipeline.config \\
#                        -t 120 -e 34 \\
#                        -m save/model/here/ \\
#                        -f 
#   will run the experiment defined in path/to/pipeline.config. The 
#   config is where the architecture , training and eval data
#   , fine-tuning checkpoints, and other model details are set.
#   
#   This will use gpus 1 and 0, train for 120 steps and perform 34
#   eval steps after each training epoch. The results of the experiment 
#   will be saved in  /save/model/here and the model will be frozen 
#   for inference. The strucutre of relevant files are below. 
#
#   /save/model/here
#     |-model.ckpt-120.data 
#     |-model.ckpt-120.index
#     |-model.ckpt-120.meta
#     |-pipeline.config
#     |-eval_0
#       |-events.out.tfevents.{timestamp}.pear
#     |-frozen 
#       |-frozen_inference_graph.pb
#   
#   - model.ckpt-120.*: The trained model architecture and weights. Used 
#      create frozen graphs and finetuning. 
#   - pipeline.config: A copy of the config used for this experiment. 
#   - frozen_inference_graph.pb: The proto_buffer of the graph. The variables
#       have been frozen to constants for optimized inference speed.
#       This is the standard TF format for inference. 
#   
#   There are a bunch of extra places where model checkpoint data is 
#   stored in slightly different forms.
#

# Training script
# From tensorflow/models/research/

# Setting Default parameters
PIPELINE_CONFIG_PATH=ssd_inception_v2_adas.config
MODEL_DIR=testnet
NUM_TRAIN_STEPS=200100
NUM_EVAL_STEPS=3100
FREEZE=false
gpus=0

print_usage() {
  printf "Description: Runs a training, eval, and freezing routine on the model defined by the pipeline config.
        Flags:
            -h: Display this help message.
            -g: The name of gpus to use. Should be seperated by comma with no spaces.(Default $gpus)
            -t: Number of training steps. (Default $NUM_TRAIN_STEPS)
            -e: Number of eval steps. (Default $NUM_EVAL_STEPS)
            -p: Path to pipeline config. (Default $PIPELINE_CONFIG_PATH)
            -m: Model name for the directory of chkpts and such. (Default $MODEL_DIR)
            -f: Whether to run a freeze scrip for the model. (Default $FREEZE)
  
  Example Usage:
  
     bash runtrain.bash -p path/to/pipeline.config -m save/model/here -t 120 -e 34 -g 0,1 -f 

   will run the experiment defined in path/to/pipeline.config. The 
   config is where the architecture, training and eval data,
   fine-tuning checkpoints, and other model details are set.
  
   This will use gpus 1 and 0, train for 120 steps and perform 34
   eval steps after each training epoch. The results of the experiment 
   will be saved in  /save/model/here and the model will be frozen 
   for inference. The structure of relevant files are below. 
   This will use gpus 1 and 0, train for 120 steps and perform 34
   eval steps after each training epoch. The results of the experiment 
   will be saved in  /save/model/here and the model will be frozen 
   for inference. The structure of relevant files are below. 

   /save/model/here
     |-model.ckpt-120.data 
     |-model.ckpt-120.index
     |-model.ckpt-120.meta
     |-pipeline.config
     |-eval_0
       |-events.out.tfevents.{timestamp}.pear
     |-frozen 
       |-frozen_inference_graph.pb
   
   - model.ckpt-120.*: The trained model architecture and weights. Used 
      create frozen graphs and finetuning. 
   - pipeline.config: A copy of the config used for this experiment. 
   - frozen_inference_graph.pb: The proto_buffer of the graph. The variables
       have been frozen to constants for optimized inference speed.
       This is the standard TF format for inference. 
   
   There are a bunch of extra places where model checkpoint data is 
   stored in slightly different forms.
"
}

while getopts 'hg:t:e:m:p:f' flag; do
  case "${flag}" in
    g) gpus="${OPTARG}" ;;
    t) NUM_TRAIN_STEPS="${OPTARG}";;
    e) NUM_EVAL_STEPS="${OPTARG}" ;;
    m) MODEL_DIR="${OPTARG}" ;;
    p) PIPELINE_CONFIG_PATH="${OPTARG}" ;;
    f) FREEZE=true ;;
    h) print_usage && exit;;
    *) print_usage && exit
       exit 1 ;;
  esac
done

echo "gpus: $gpus"
echo "train_steps: $NUM_TRAIN_STEPS"
echo "eval_steps: $NUM_EVAL_STEPS"
echo "model: $MODEL_DIR"

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

if $FREEZE  
then
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
fi

