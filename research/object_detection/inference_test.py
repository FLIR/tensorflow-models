# Runs inference on pictures in test_images/
# Just a slight modification of the ipyb here:
#
#   https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
#
# Use it to test saved models when evaluations are iffy.

import numpy as np
import os
import sys
import time
import argparse
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

# pb files downloaded from
#   https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

PRE_TRAINED_ZOO = {
    'mobilenet_v1_fpn':'/home/jroberts/basil_lab/tflow_zoo/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb',
    'mobilenet_v2': '/home/jroberts/basil_lab/tflow_zoo/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
    'inception_v2':'/home/jroberts/basil_lab/tflow_zoo/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
    'recent':'../testnet_freeze/frozen_inference_graph.pb'
    }

def load_image_into_numpy_array(image,grey=False):
    (im_width, im_height) = image.size
    result = np.array(image.getdata())

    if grey:
        presult = result.reshape((im_height, im_width)).astype(np.uint8)
        
        result = np.zeros((im_height,im_width,3))
        result = np.stack((presult,)*3, axis=-1)
        
    else:
        result = result.reshape((im_height, im_width,3)).astype(np.uint8)
    
    return result
def run_inference_for_single_image(images, graph):
    time_results = []
    detection_results = []
    with graph.as_default():

        for i in range(len(images)):
            image = images[i]
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                            ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                t0 = time.time()
                output_dict = sess.run(tensor_dict,
                                        feed_dict={image_tensor: np.expand_dims(image, 0)})
                t1 = time.time()
                time_results.append(t1-t0)

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        
        detection_results.append(output_dict)

    return detection_results, time_results

def invert_normalized_coord(im_width,im_height,norm_coords):
    ymin, xmin, ymax, xmax = norm_coords
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-name',
                        help='Name of pretrained model to use',
                        type = str,
                        default= 'recent'
                        )
    
    parser.add_argument('--model-pb-file',
                        help='Path to a froze pb file for inference. If model_name is also specified this will be used instead',
                        type=str,
                        default=None
                        )
    
    parser.add_argument('--grey',
                        help='Whether data is greyscale or not. If False assumes image is color. Assumes grey is ADAS data',
                        type=bool,
                        default=False
                        )
    
    parser.add_argument('--img-dir',
                        help='Directory containing test images',
                        type=str,
                        default='test_images'
                        )
    
    parser.add_argument('--plot',
                        help='Boolean whether to plot the results',
                        type=str,
                        default=False
                        )

    args = parser.parse_args()

    names_msg =  """model_name must be from
                    {allowed} 
                    but is {given}"""
    
    if args.model_pb_file is None:
        assert args.model_name in PRE_TRAINED_ZOO.keys(), names_msg.format(allowed=PRE_TRAINED_ZOO.keys()
                                                                    ,given=args.model_name)
        model_path = args.model_name
    else:
        model_path = args.model_pb_file
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = PRE_TRAINED_ZOO[model_path]
    mdl_msg = """
                =======================================
                Using model:
                    {}
                =======================================
                """
    print(mdl_msg.format(PATH_TO_FROZEN_GRAPH))

    # List of the strings that is used to add correct label for each box.
    if args.grey:
        PATH_TO_LABELS = os.path.join('data', 'adas_label_map.pbtxt')    
    else:
        PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90

    detection_graph  = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
 
    PATH_TO_TEST_IMAGES_DIR = args.img_dir
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, x) for x in os.listdir(PATH_TO_TEST_IMAGES_DIR)]
    print(TEST_IMAGE_PATHS)

    # Get just the jp(e)gs. Make this better later
    TEST_IMAGE_PATHS = [x for x in TEST_IMAGE_PATHS if os.path.basename(x).split('.')[-1] in ['jpg','jpeg']]

    test_images = []
    out_results = []
    for  i,image_path in enumerate(TEST_IMAGE_PATHS):
        N = len(TEST_IMAGE_PATHS)
        N = int(np.sqrt(N))
        n = int(np.sqrt(N))
        image = Image.open(image_path)
        
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image,grey=args.grey)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        #image_np_expanded = np.expand_dims(image_np, axis=0)
        image_np_expanded = image_np

        test_images.append(image_np_expanded)
    # Actual detection graph
    detection_graph = tf.get_default_graph()

    # Run the test

    output_dicts, time_results = run_inference_for_single_image(test_images, detection_graph)
    out_results.append(output_dicts)
    print(output_dicts)
    assert False
    print(time_results)
    if args.plot:
        for i,output_dict in enumerate(output_dicts):
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.figure(int(i/N)+1)
            #plt.subplot(n,n,1+i%N)
            plt.subplot(3,3,i+1)
            #plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
    if args.plot:
        plt.show()
    
