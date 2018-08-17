
IMAGE_DIR_BASE=/home/jroberts/conservator_datasets/ADAS/merged

TRAIN_IMAGE_DIR=$IMAGE_DIR_BASE/train/PreviewData
VAL_IMAGE_DIR=$IMAGE_DIR_BASE/val/PreviewData

TRAIN_ANNOTATIONS_FILE=$IMAGE_DIR_BASE/train/merged_annotations.json
VAL_ANNOTATIONS_FILE=$IMAGE_DIR_BASE/val/merged_annotations.json

OUTPUT_DIR=/home/jroberts/conservator_datasets/ADAS/merged/trecords

TRAIN_SHARDS=2

python ../dataset_tools/create_adas_tf_record.py --logtostderr \
      --train_image_dir=$TRAIN_IMAGE_DIR \
      --val_image_dir=$VAL_IMAGE_DIR \
      --test_image_dir=$TEST_IMAGE_DIR \
      --train_annotations_file=$TRAIN_ANNOTATIONS_FILE \
      --val_annotations_file=$VAL_ANNOTATIONS_FILE \
      --train_shards=$TRAIN_SHARDS \
      --output_dir=$OUTPUT_DIR