# Runs the tfrecord creator for 
# ADAS data. Assumes all images are 
# in jpeg compression format

# Should be run from tensorflow-models/research/object_detection/adas_utils/

DIR_BASE=$1       # Should not have trailing /
OUTPUT_DIR=$2     #Where to save the tfrecords
MEGER_ANNOS=false

print_usage() {
  printf "Usage: Runs a training, eval, and freezing routine on the model defined by the pipeline config.
        Flags:
            -h: Display this help message.
            -g: The name of gpus to use. Should be seperated by comma with no spaces.(Default $gpus)
            -t: Number of training steps. (Default $NUM_TRAIN_STEPS)
            -e: Number of eval steps. (Default $NUM_EVAL_STEPS)
            -p: Path to pipeline config. (Default $PIPELINE_CONFIG_PATH)
            -m: Model name for the directory of chkpts and such. (Default $MODEL_DIR)
            -f: Whether to run a freeze scrip for the model. (Default $FREEZE)
"
}

while getopts 'hm' flag; do
  case "${flag}" in
    m) MEGER_ANNOS=true ;;
    h) print_usage && exit;;
    *) print_usage && exit
       exit 1 ;;
  esac
done

# Some of the directores put things in 
#           PreviewData/
# Some put it in
#           Data/ 


TRAIN_IMAGE_DIR=$DIR_BASE/train/PreviewData     
VAL_IMAGE_DIR=$DIR_BASE/val/PreviewData

TRAIN_ANNOTATIONS_DIR=$DIR_BASE/train/Annotations/
VAL_ANNOTATIONS_DIR=$DIR_BASE/val/Annotations/

if $MEGER_ANNOS
then
echo "Merging train annotations"
python merge_annotations.py --anno-dir $TRAIN_ANNOTATIONS_DIR \
                              --catids $DIR_BASE/catids.json \
                              --out-name $DIR_BASE/train/merged_annotations.json \
                              --verbose 1

echo "Merging val annotations"
python merge_annotations.py --anno-dir $VAL_ANNOTATIONS_DIR \
                              --catids $DIR_BASE/catids.json \
                              --out-name $DIR_BASE/val/merged_annotations.json \
                              --verbose 1
fi                              
# Make the merged annotations files
# to be compatible with MSCOCO format
TRAIN_ANNOTATIONS_FILE=$DIR_BASE/train/merged_annotations.json    
VAL_ANNOTATIONS_FILE=$DIR_BASE/val/merged_annotations.json        

TRAIN_SHARDS=2    # Sharding a dataset can help throughput during training (Or so TF says....)

echo "Creating tfrecords"
python ../dataset_tools/create_adas_tf_record.py --logtostderr \
      --train_image_dir=$TRAIN_IMAGE_DIR \
      --val_image_dir=$VAL_IMAGE_DIR \
      --test_image_dir=$TEST_IMAGE_DIR \
      --train_annotations_file=$TRAIN_ANNOTATIONS_FILE \
      --val_annotations_file=$VAL_ANNOTATIONS_FILE \
      --train_shards=$TRAIN_SHARDS \
      --output_dir=$OUTPUT_DIR