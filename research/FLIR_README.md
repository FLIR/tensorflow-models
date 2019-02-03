
# OVERVIEW

These are scripts added for FLIR use to tensorflows' models package. The names and a short description of them is given below. For detailed usage please see the scripts documentation.

Most of these require the tensorflow-models/object_detection requirements installed. For specific requirements see:
            https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md . 

# tensorflow-models
- [make_adas_records.bash](make_adas_records.bash): A wrapper for the full tfrecord formation from a FLIR dataset structure. Assumes the directory structure is 
    ```
    BASE_DIR
    |--Annotations/
    |--(Preview)Data/
    ```
    
            
    Creates the directory(ies) and records:
        
    ```
    BASE_DIR
    |--Annotations/
    |--(Preview)Data/
    |--tfrecords/
        |--adas.record-00000-00001
    |--(Preview)Data_colored*/
    ```

    (*) Created only if -c is set. This should be set if some greyscale images only have 1 channel. 

    Example:
    ```sh
    bash make_adas_records.bash -d ABSOLUTE/PATH/TO/IMAGES -g 3 -p -c
    ```
    Will make tfrecords from the data and annotations in ABSOLUTE/PATH/TO/IMAGES. It will use gpu device 3, the data should be in the folder PreviewData, and it will "colorize" the data in case it is not 3 channels. 



- [runtrain.bash](runtrain.bash): The main script for running training of a tf object detector.   Requires the data to be saved in a tfrecord format and a object detection pipeline config file. An example config file is [flir_sample.config](flir_sample.config).


    Example:
    ```sh
    bash runtrain.bash -p path/to/pipeline.config -m save/model/here -t 120 -e 34 -g 0,1 -f 
    ```

    will run the experiment defined in path/to/pipeline.config, dump the results in save/model/here, after 120 training steps which are intermittedly interrupted to run 34 evaluation steps, gpu devices 0 and 1 will be used, and at the end it will genreate a frozen inference graph. 
    
    The config is where the architecture, training data, eval data, fine-tuning checkpoints, and other experiment details are set.

- [tensorflow_models_setup.source](tensorflow_models_setup.source):
    Sets up the tensorflow virtualenv, unwraps the tensorflow-models proto files, and sets the pythonpath. Can be run before using this package if you have the virtualenv of tensorflow.

    Example
    ```sh
    source tensorflow_models_setup.source
    ```

- [freeze.bash](freeze.bash):
    Takes a model directory and generates a frozen inference graph for a given checkpoint. Assumes model.ckpt-?????.{meta,index,data} files are in the directory.

    Example
    ```sh
    bash freeze_example.bash -p path/to/model_dir -c 12345
    ```

    will freeze model.ckpt-12345 inside of model-dir.

- [freeze_quick.bash](freeze_quick.bash): 
    A wrapper for freeze.bash when you only want the latest model frozen see freeze.bash for directory structure.

# flir_utils/

- [count_records.py](flir_utils/count_records.py): Counts the number of examples in each tfrecord of a given directory.
    
    Example:

    ``` sh
    python count_records.py /$PATH/TO/RECORDS/DIR/
    ```

- [peek_in_tfrecords.py](flir_utils/peek_in_tfrecords.py): 
    Looks inside tfrecords made by [create_adas_tfrecords.py](object_detection/dataset_tools/create_adas_tfrecords.py) and displays or saves the images with the annotation boxes on them.

    Example:

    ```sh
        python peek_in_tfrecords.py --record Path/To/Record.tfrecord \
                                    --catids Path/To/catids.json \
                                    --num-images 10
    ```

    Will display a slide show of 10 images. To go to next image us the space bar. If --save-dir is set then the images will not be displayed. Warning: Loads --num-images into memory. So if this is too high it will be slow.

# object_detection/

- [standard_eval.py](object_detection/standard_eval.py):
    Runs inference from a frozen tensorflow .pb graph for object detection on an image directory. The results are formatted to be compatilbe with:

        caffe/scripts/data_prep/evaluateDetections.sh
        
    Example Usage:
    ```sh
    python standard_eval.py --model-pb-file /path/to/frozen_inference_graph.pb \
                            --image-dir /path/to/images \
                            --output output_results.txt
    ```
    Results are dumped in output_results.txt.

## adas_utils/
    
- [merge_annotations.py](object_detection/adas_utils/merge_annotations.py): 
    Merges annotations found in Annotations directory of an ADAS type datasets into one large annotations json file which is labeled to be consistent with MSCOCO's Annotations jsons.

    Example:
    ```sh
    python merge_annotations.py --anno-dir {$PATH/TO/ANNOTATIONS_DIR} \
                            --catids {$PATH/TO/CATEGORY_IDS.json} \
                            --outname {$PATH/TO/OUTPUT.json} \
                            --verbose 500 
    ```
- [merge_annotations.bash](object_detection/adas_utils/merge_annotations.bash):
    A wrapper for merge_annoations.py for common uses. Should be run from within object_detection/adas_utils/.

    Example:
    ```sh
    bash merge_annotations.bash PATH/TO/DIR_BASE
    ```
- [pseudo_color.py](object_detection/adas_utils/pseudo_color.py):
   Goes through a directory of greyscale images and buffs them to 3 channels. If they are already 3 channels it leaves them alone. Saves ONLY THE MODIFIED images. It is advisable to do this before creating tfrecords to verify data integrity and speed up the records' creation. 
        
    Example:
    ```sh
    python pseudo_color.py --image-dir path/to/images/ --out-dir tmp_dir/
    ```
    If you are okay modifying the existing database then to combine:      

    ```sh
    cp tmp_dir/* path/to/images/ && rn -r tmp_dir
    ```
    
    Otherwise you can make tmp_dir your new working database 
    ```sh
    cp -n path/to/images/* tmp_dir
    ```
## dataset_tools/

- [create_adas_tf_record.py](object_detection/dataset_tools/create_adas_tf_record.py):
    Convert raw adas dataset to TFRecord for object_detection.
    Expects all images in jpeg compression format.
    Please note that this tool creates sharded output files.

    Example usage:
    ```sh
        python create_adas_tf_record.py --logtostderr \
        --train_image_dir="${TRAIN_IMAGE_DIR}" \
        --val_image_dir="${VAL_IMAGE_DIR}" \
        --test_image_dir="${TEST_IMAGE_DIR}" \
        --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
        --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
        --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
        --output_dir="${OUTPUT_DIR}"
    ```
    
    If the a directory flag is omitted it is skipped. Assumes all data is greyscale with 3 channels to agree with mscoco models. If you are not sure if this is the case run adas_utils/pseudo_color.py for your directory. Putting channel checks here severely impacts performance. 
