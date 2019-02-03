# @jroberts: 2/2019
#   Merges the annotation files in the train and val directories of a merged 
#   directory.
#   Should be run from tensorflow-models/research/object_detection/adas_utils/
#   Example Usage:
#       bash merge_annotations.bash DIR_BASE 
#   
#       INPUT DIR STRUCTURE
#   DIR_BASE/
#   |--catids.json
#   |--labelvectors.json
#   | 
#   |--train/
#   |  |--Annotations/
#   |--val/
#      |--Annotations/
#
#       OUTPUT DIR STRUCTURE
#   DIR_BASE/
#   |--catids.json
#   |--labelvectors.json
#   | 
#   |--train/
#   |  |--merged_annotations.json
#   |  |--Annotations/
#   |--val/
#      |--merged_annotations.json
#      |--Annotations/

DIR_BASE=$1

TRAIN_ANNOTATIONS_DIR=$DIR_BASE/train/Annotations/
VAL_ANNOTATIONS_DIR=$DIR_BASE/val/Annotations/

echo "Merging train annotations"
sudo python merge_annotations.py --anno-dir $TRAIN_ANNOTATIONS_DIR \
                              --catids $DIR_BASE/catids.json \
                              --out-name $DIR_BASE/train/merged_annotations.json \
                              --verbose 500 # How often to print progress. Remove or set to -1 if you don't want to know

echo "Merging val annotations"
sudo python merge_annotations.py --anno-dir $VAL_ANNOTATIONS_DIR \
                              --catids $DIR_BASE/catids.json \
                              --out-name $DIR_BASE/val/merged_annotations.json \
                              --verbose 500 # How often to print progress. Remove or set to -1 if you don't want to know
                              