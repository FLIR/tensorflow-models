import numpy as np 
import PIL.Image as Image
import os 
import time 
import argparse

_usg = """ 9/7/2018 @jroberts
        Goes through a directory of greyscale images and buffs them to 3 channels. If they are 
        already 3 channels it leaves them alone. Saves ONLY THE MODIFIED images in --out-dir under 
        their same name. It is advisiable to do this before creating tfrecords to verify data
        integrity and speed up the records' creation. 
        Example:
        $python pseudo_color.py --image-dir path/to/images/ --out-dir tmp_dir/
        
        If you are okay modifying the existing database then to combine:        
        $cp tmp_dir/* path/to/images/ && rn -r tmp_dir

        Otherwise you can make tmp_dir your new working database 
        $cp -n path/to/images/* tmp_dir
        """
def pseudo_color(image_dir,out_dir):
    """
    Goes through image_dir and finds all images which are 1 channels greyscale
    and buffs them up to 3 channel greyscale. 
    """
    baddies = 0
    num_images = len(os.listdir(image_dir))
    t0 = time.time()
    for i,img_name in enumerate(os.listdir(image_dir)):
        full_path = os.path.join(image_dir,img_name)
        image = Image.open(full_path)
        image_array = np.array(image)
        img_shape = list(image_array.shape)

        if i % 100 == 0:
            msg = "On image {} of {}. Found {} baddies. ({:.2f} secs)"
            print(msg.format(i,num_images,baddies,time.time()-t0))
            t0 = time.time()
        assert len(img_shape) > 1, 'Image {} is corrupted. Has shape {}'.format(full_path, img_shape)
        # Check if it is RGB
        if len(img_shape) < 3:
            #print('Found a baddie')
            baddies+=1
            new_image = np.zeros(img_shape+[3]).astype(np.uint8)
            # Make new RGB version of it
            for i in range(3):
                new_image[...,i] = image_array
            
            # Close the original link
            image.close()

            new_image = Image.fromarray(new_image)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir,img_name)
            with open(out_path,'wb') as fb:
                new_image.save(fb)
            
            new_image.close()

    msg = "Found {} baddies out of {} total"
    print(msg.format(baddies,num_images))

if __name__=="__main__":
    parser = argparse.ArgumentParser(usage = _usg)

    parser.add_argument('--image-dir',
                        help='Images you want to pseudo-color',
                        type=str)
    
    parser.add_argument('--out-dir',
                        help='Directory to write the modifeid images. Only modified images are saved.',
                        type = str)
    
    args = parser.parse_args()

    assert args.image_dir, 'Must specify --image-dir'
    assert args.out_dir, 'Must specify --out-dir'
    pseudo_color(args.image_dir,args.out_dir)
    