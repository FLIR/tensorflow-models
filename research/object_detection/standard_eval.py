_usg =  """
        @jroberts 2018-10-11

        Runs inference from a frozen tensorflow .pb graph for object detection on an image directory. 
        The results are formatted to be compatilbe with 
            caffe/scripts/data_prep/evaluateDetections.sh
        
        Example Usage:
            python standart_eval.py --model-pb-file /path/to/frozen_inferance_graph.pb \
                                    --image-dir /path/to/images
        
        Results are dumped in eval_results.txt.
        """
import numpy as np
import tensorflow as tf

from io import StringIO
from PIL import Image

import time
import os
import sys
import argparse

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    result = np.array(image.getdata())
        
    result = result.reshape((im_height, im_width,3)).astype(np.uint8)
    
    return result
def run_inference_on_images(graph, image_paths, batch_size=1):
    # bboxes are given in 
    # ymin, xmin, ymax, xmax 
    detection_results = []
    num_images = len(image_paths)
    if batch_size >1:
        figit = 1
        if num_images % batch_size == 0:
            figit = 0
        # Wiggle for the last batch
        N = num_images/batch_size +figit
    else:
        N = num_images

    with graph.as_default():
        t0 = time.time()
        with tf.Session() as sess:
            for i in range(N):
                # Batch fun
                if batch_size ==1:
                    img_path = image_paths[i]
                    image = Image.open(img_path)
                    image = load_image_into_numpy_array(image)
                    inputs = np.expand_dims(image, 0)

                else:
                    #inputs = images[i*batch_size:(i+1)*batch_size,...]
                    inputs = []
                    batch_img_paths = image_paths[i*batch_size:(i+1)*batch_size]
                    
                    for  img_path in batch_img_paths:
                        image = Image.open(img_path)
                        image_np = load_image_into_numpy_array(image)
                        inputs.append(image_np)
                    
                    inputs = np.array(inputs)

                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes'
                            ]:
                    tensor_name = key + ':0'

                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
                # Input node
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                        feed_dict={image_tensor:inputs})
                
                # Fix detections
                output_dict['num_detections'] = [int(output_dict['num_detections'][k]) for k in range(len(output_dict['num_detections']))]
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes']
                output_dict['detection_scores'] = output_dict['detection_scores']
                output_dict['image_id'] = batch_img_paths

                detection_results.append(output_dict)
                # Progress report
                if i % 5 == 0:
                    msg = "Completed batch {} of {}. ({:.2f} secs)"
                    print(msg.format(i,N,time.time()-t0))
                    t0 = time.time()

    return detection_results, inputs.shape

def stringy(l):
    # Add zeros for times
    l = l+[0,0]
    N = len(l)
    result = [0]*N
    for i in range(N):
        r = l[i]
        if not isinstance(r,str):
            r = str(r)
        result[i] = r 
    result = ' '.join(result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=_usg)
    
    parser.add_argument('--model-pb-file',
                        help='Path to a froze pb file for inference. Must be a frozen inference graph not a saved_model.pb Esitmator type.',
                        type=str,
                        default=None
                        )
    
    parser.add_argument('--image-dir',
                        help='Directory containing test images. Must ALL be 3-channel even if greyscale.',
                        type=str,
                        default='adas_utils/tmp'
                        )
    
    parser.add_argument('--batch-size',
                        help = 'Batch size to use for images. If greater than 1 assumes images have same resolution',
                        type = int,
                        default = 10)

    parser.add_argument('--threshold',
                        help = '(Deprecated) Threshold for keep detection. If set lower than that of the saved model this argument will have no effect.',
                        type = float,
                        default = None)
    
    parser.add_argument('--output',
                        help = 'Name of output file',
                        type = str,
                        default = 'eval_results.txt')
    
    args = parser.parse_args()
    path_to_graph = args.model_pb_file

    print(path_to_graph)
    # Build the detection graph
    detection_graph  = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_graph, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    image_paths = [ os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) ]
    # Get rid of non jp(e)gs
    image_paths = [x for x in image_paths if os.path.basename(x).split('.')[-1] in ['jpg','jpeg']]
    
    # Actual detection graph
    detection_graph = tf.get_default_graph()

    # Returns a list of detection dicts. One list element for each image.
    output_dicts, last_batch_shape = run_inference_on_images(detection_graph, 
                                            image_paths,
                                            batch_size= args.batch_size)
    img_width, img_height = last_batch_shape[1:3]
    
    #FLIR Format [image_id, label, score, xmin, ymin, xmax, ymax]
    Results = []
    # Get the batch
    t0 = time.time()
    num_batches = len(output_dicts)
    for bb,detection_batch in enumerate(output_dicts):
        b_size = len(detection_batch['image_id'])
        assert b_size == detection_batch['detection_boxes'].shape[0]

        updt_msg = "Processing Detections into FLIR format. Batch {} of {}. ({:.2f} secs)"
        
        if bb % 5 == 0:
            t1= time.time() - t0
            print(updt_msg.format(bb,num_batches,t1))
            t0 = time.time()
        # Unpack the detections
        for i in range(b_size):
            num_detections = detection_batch['num_detections'][i]
            image_id = detection_batch['image_id'][i]
            detection_boxes = detection_batch['detection_boxes'][i,...]
            detection_scores = detection_batch['detection_scores'][i,...]
            detection_classes = detection_batch['detection_classes'][i,...]

            image = Image.open(image_id)
            image_np = load_image_into_numpy_array(image)
            
            # Get all detections
            for j in range(num_detections):
                score = detection_scores[j]
                d_class = detection_classes[j]
                ymin, xmin, ymax, xmax = detection_boxes[j,:]
                # Denormalize
                ymin, ymax = int(img_width*ymin), int(img_width*ymax)
                xmin, xmax = int(img_height*xmin), int(img_height*xmax)
                # threshold and put in FLIR order
                #if score >= args.threshold: Change to VERY low for fair mAP comparison
                if score >= 1e-9:
                    Results.append([image_id,d_class, score, xmin, ymin, xmax, ymax])
                
    # Write results file
    with open(args.output, 'w') as fr:
        for r in Results:
            res = stringy(r)
            fr.write(res+'\n')
    
