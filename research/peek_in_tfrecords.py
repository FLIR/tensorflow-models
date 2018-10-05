import tensorflow as tf 
import numpy as np 
import json    
import matplotlib.pyplot as plt 
from object_detection.utils.visualization_utils import visualize_boxes_and_labels_on_image_array as viz

import os 
import time
import argparse
import multiprocessing as mp 

def adas_parser(serialized_example):
    record_feature_dict = {
      'image/height': tf.FixedLenFeature([],tf.int64),
      'image/width':tf.FixedLenFeature([],tf.int64),
      #'image/filename':tf.FixedLenFeature([],tf.string),
      'image/filename':tf.VarLenFeature(tf.string),
      'image/source_id':tf.FixedLenFeature([],tf.string),
      'image/key/sha256':tf.FixedLenFeature([],tf.string),
      'image/encoded':tf.FixedLenFeature([],tf.string),
      'image/format':tf.FixedLenFeature([],tf.string),
      'image/object/bbox/xmin':tf.VarLenFeature(tf.float32),
      'image/object/bbox/xmax':tf.VarLenFeature(tf.float32),
      'image/object/bbox/ymin':tf.VarLenFeature(tf.float32),
      'image/object/bbox/ymax':tf.VarLenFeature(tf.float32),
      'image/object/class/label':tf.VarLenFeature(tf.int64),
      'image/object/is_crowd':tf.VarLenFeature(tf.int64),
      'image/object/area':tf.VarLenFeature(tf.float32)
        }

    # parse one
    features = tf.parse_single_example(serialized_example,
                                      features=record_feature_dict)
    
    image = tf.image.decode_jpeg(features['image/encoded'])
    
    img_filename = tf.sparse_tensor_to_dense(features['image/filename'],default_value='')
    #img_filename = tf.cast(img_filename, tf.string)
    # Normalized mins and maxes
    # VarLenFeatures are sparse tensors and need to be converted to dense
    xmins = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])
    ymins = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
    xmaxs = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'])
    ymaxs = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'])
    class_labels = tf.sparse_tensor_to_dense(features['image/object/class/label'])
    
    # Resize decoded image
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)

    #image_shape = tf.stack([height, width, 1])
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)

    # object_dection.utils.visualization_utils.py uses ordering
    #   ymin,xmin,ymax,xmax
    # so we return bboxes like that
    annotations = [tf.stack([ymins,xmins,ymaxs,xmaxs], axis = -1),class_labels]
    
    # Datasets get mad at features with lists and not tuples
    features = {"image": image, "annotations": tuple(annotations), "img_filename":img_filename}
    
    return features

def create_dataset_iterator(file_names):
    # Make the dataset
    dataset = tf.data.TFRecordDataset(file_names)

    # Parse it in parallel
    num_slaves = mp.cpu_count()
    dataset = dataset.map(adas_parser)

    # Buffer the batch size of data
    dataset = dataset.prefetch(2)

    # Batch it and make iterator
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    
    features = iterator.get_next()

    return features

def get_all_records(file_names,num_images = 10):
    """
    Decodes the images and annotations in a tfrecord made by 
    create_{adas,coco}_records.py. 
    
    Parameters:
    -----------
        file_names: List of paths to tfrecords.
        num_images: The number of images to return. Reads the record in order;
            however, creating a tfrecord may change the order of original images. 
        
    Returns:
    --------
    (images, bboxes, classes, _img_filenames): tuple of lists
        images[i]: The ith image from the tfrecrod.
        bboxes[i]: List of bounding boxes for image images[i].
            In normalized coordinates with order (ymin,xmin,ymax,xmax).
        classes[i]: List of the class corrsponding to the bboxes[i].
        img_filenames[i]: images[i]'s source id
    
    Returns num_images of the above. 
    """
    # Start session
    with tf.Session() as sess:
        features = create_dataset_iterator(file_names)
        image = features["image"]
        annotations = features["annotations"]
        img_filename = features["img_filename"]
        # initialize operations
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        images = [0]*num_images
        bboxes = [0]*num_images
        classes = [0]*num_images
        img_filenames = [0]*num_images
        for i in range(num_images):
            # Get 'em 
            img, anns, f_names = sess.run([image, annotations, img_filename])
                    
            # Take the first element because of batching
            images[i] = img[0] 
            img_filenames[i] = f_names[0]
            bboxes[i] = anns[0][0]
            classes[i] = anns[1][0]

    return images, bboxes, classes, img_filenames

def peek(record, 
        catids, 
        num_images = 10, 
        plot = True, 
        save_dir = None):
    """
    Views num_images images from tfrecords in the list records. Optional
    plot and save the images. Plots can be closed by keystroke.
    Parameters:
    -----------
        records: List of paths to tfrecord files.
        catids: Path to catids json for whatever categories you care about.
        num_images: How many images to peek at.
        plot: Whether to plot the images. 
        save_dir: If not None then images will be saved with their bboxes
            and labels inside save_dir.
    Returns:
    --------
        None. Either plots or saves the annotated images based on inputs.
    """
    # Convert catids to category_index as using in 
    # visualization_utils.visualize_boxes_and_labels_on_image_array()
    with open(catids,'r') as fj:
        idx = json.load(fj)
    catids = {int(x["id"]):x for x in idx}

    # Get 'em
    imgs, bbxs, labels, f_names = get_all_records([record], num_images=num_images)
    
    scores = [[100.0]*len(c) for c in labels]
    # Save 'em 
    if not save_dir is None:
        print('Saving Images in %s'%save_dir)
        msg = "Saving image {} of {}. ({} secs)"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        t0 = time.time()
        for i in range(num_images):
            # Draw on the array
            vandalized = viz(imgs[i],
                                bbxs[i],
                                labels[i],
                                scores[i],
                                catids,use_normalized_coordinates=True)
            np.save(os.path.join(save_dir,f_names[i])[0],vandalized)
            if i % 25 == 0:
                t = time.time() - t0
                print(msg.format(i,num_images,t))
                t0 = time.time()
    # Draw 'em
    if plot:
        for i in range(num_images):
            # Draw on the array
            vandalized = viz(imgs[i],
                                bbxs[i],
                                labels[i],
                                scores[i],
                                catids,use_normalized_coordinates=True)
            plt.imshow(vandalized)
            ttl = "{} \n {}/{}"
            plt.title(ttl.format(f_names[i],i,num_images))
            while True:
                if plt.waitforbuttonpress():
                    break

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--record',
                        type = str,
                        help = 'Path to ADAS tfrecord file.')

    parser.add_argument('--catids',
                        type = str,
                        help = 'Path to catids file.',
                        default = '/mnt/fruitbasket/users/jroberts/ADAS/thermal_adas_real_15306/merged/catids.json')

    parser.add_argument('--num-images',
                        type = int,
                        default = 10,
                        help = 'Number of images to display for your viewing pleasure.')
    
    parser.add_argument('--save-dir',
                        default = None,
                        help = 'Where to save images. If left None then no images are saved.')
    
    args = parser.parse_args()    

    #hmm = '/home/jroberts/basil_lab/tensorflow-models/research/object_detection/adas_utils/Syncity_test/tfrecords/adas_train.record-00000-of-00001'
    hmm = '/mnt/fruitbasket/users/jroberts/ADAS/thermal_adas_real_15306/tfrecords_clean/adas_train.record-00000-of-00002'
    peek(hmm,args.catids, num_images=args.num_images, save_dir=args.save_dir)