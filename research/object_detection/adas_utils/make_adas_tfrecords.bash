
TRAIN_IMAGE_DIR=/home/jroberts/adas_test1000/merged/train/Data
VAL_IMAGE_DIR=/home/jroberts/adas_test1000/merged/val/Data

TRAIN_ANNOTATIONS_FILE=/home/jroberts/adas_test1000/merged/train/merged_annotations.json
VAL_ANNOTATIONS_FILE=/home/jroberts/adas_test1000/merged/val/merged_annotations.json

OUTPUT_DIR=/home/jroberts/adas_test1000/tfrecords

TRAIN_SHARDS=2

python create_adas_tf_record.py --logtostderr \
      --train_image_dir=$TRAIN_IMAGE_DIR \
      --val_image_dir=$VAL_IMAGE_DIR \
      --test_image_dir=$TEST_IMAGE_DIR \
      --train_annotations_file=$TRAIN_ANNOTATIONS_FILE \
      --val_annotations_file=$VAL_ANNOTATIONS_FILE \
      --train_shards=$TRAIN_SHARDS \
      --output_dir=$OUTPUT_DIR