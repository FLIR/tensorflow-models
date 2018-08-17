# Runs the tfrecord creator for 
# ADAS data. Assumes all images are 
# in jpeg compression format


IMAGE_DIR_BASE=/home/jroberts/conservator_datasets/ADAS/merged

TRAIN_IMAGE_DIR=$IMAGE_DIR_BASE/train/PreviewData     # Where the train images are 
VAL_IMAGE_DIR=$IMAGE_DIR_BASE/val/PreviewData         # Where the val images are

TRAIN_ANNOTATIONS_FILE=$IMAGE_DIR_BASE/train/merged_annotations.json    # A merged annoations file as outputed by merge_annoations.py
VAL_ANNOTATIONS_FILE=$IMAGE_DIR_BASE/val/merged_annotations.json        # A merged annoations file as outputed by merge_annoations.py

OUTPUT_DIR=/home/jroberts/conservator_datasets/ADAS/merged/trecords     # Where to save the tfrecords

TRAIN_SHARDS=2    # Sharding a dataset can help throughput during training (Or so TF says....)

python ../dataset_tools/create_adas_tf_record.py --logtostderr \
      --train_image_dir=$TRAIN_IMAGE_DIR \
      --val_image_dir=$VAL_IMAGE_DIR \
      --test_image_dir=$TEST_IMAGE_DIR \
      --train_annotations_file=$TRAIN_ANNOTATIONS_FILE \
      --val_annotations_file=$VAL_ANNOTATIONS_FILE \
      --train_shards=$TRAIN_SHARDS \
      --output_dir=$OUTPUT_DIR