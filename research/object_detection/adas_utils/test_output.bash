IMAGE_DIR="/home/jroberts/conservator_datasets/ADAS/test_adas_jun4_copy_from_tw_07/merged/PreviewData"
ANNO_DIR="/home/jroberts/conservator_datasets/ADAS/test_adas_jun4_copy_from_tw_07/merged/Annotations"
ANNO_FILE="/home/jroberts/conservator_datasets/ADAS/test_adas_jun4_copy_from_tw_07/merged/merged_annotations.json"

echo $IMAGE_DIR

python merge_annotations.py --anno-dir $ANNO_DIR/ \
                    --catids /home/jroberts/conservator_datasets/ADAS/merged/catids.json \
                    --out-name $ANNO_FILE

python ../dataset_tools/create_adas_tf_record.py --logtostderr \
      --train_image_dir=$IMAGE_DIR \
      --val_image_dir=$IMAGE_DIR \
      --test_image_dir=$IMAGE_DIR \
      --train_annotations_file=$ANNO_FILE \
      --val_annotations_file=$ANNO_FILE \
      --test_annotations_file=$ANNO_FILE \
      --train_shards=1 \
      --output_dir=.