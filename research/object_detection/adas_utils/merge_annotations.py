_usg = """ 8/20/2018 @jroberts
    Merges annotations found in Annotations directory of an ADAS 
    type datasets into one large annotations json file which is 
    labeled to be consistent with MSCOCO's Annotations jsons. 

Example usage: \n
    python merge_annotations.py --anno-dir {$PATH/TO/ANNOTATIONS_DIR} \r 
                                --catids {$PATH/TO/CATEGORY_IDS.json} \r 
                                --outname {$PATH/TO/OUTPUT.json} \r 
                                --verbose 500 \n

    Will take all annotations in {$PATH/TO/ANNOTATIONS_DIR} and  
    combine them with the category ids in  {$PATH/TO/CATEGORY_IDS.json} 
    into one big annotation file called {$PATH/TO/OUTPUT.json}. By default
    --keep-blanks is set to True. This means images with no ground truths
    are kept.
"""
import json 
import os 
import argparse
import time 

def merge_annotations(anno_dir,catids,out_name, 
                        verbose = -1, keep_blank_annotations = False):
    
    num_images = len(os.listdir(anno_dir))
    images_data = [0]*num_images
    annos_data = [0]*num_images
    blank_annos = []

    # Add the images and annotations
    t0 = time.time()
    for i, anno_file in enumerate(os.listdir(anno_dir)):
        path = os.path.join(anno_dir,anno_file)        
        if i % verbose == 0 and verbose > 0:
            msg = "Processing annotation file {}/{}. ({} secs)"
            print(msg.format(i,num_images,round(time.time()-t0,4)))
            t0 = time.time()
        
        # Get the data from an annotation
        with open(path,'r') as f:
            data = json.load(f)
            img_data = data['image']
            
            # Take only the base name of file_name
            img_data['file_name'] = os.path.basename(img_data['file_name'])            
            if len(data['annotation']) == 0:
                an_data = data['annotation']
                blank_annos.append(i)
            else:
                an_data = data['annotation']
            
            # Add it to the running list
            images_data[i] = img_data
            annos_data[i] = an_data
    
    # Remove blanks if told to 
    if not keep_blank_annotations:
        images_data = [images_data[k] for k in range(num_images) if not k in blank_annos ]
        annos_data = [annos_data[k] for k in range(num_images) if not k in blank_annos ]
    
    # annos_data: list of list of dict -> list of dict
    annos_data = [item for sublist in annos_data for item in sublist]
    with open(catids, 'r') as fcat:
        cats = json.load(fcat)

    # Create the merged annotations file. 
    with open(out_name, 'w') as fw:
        json.dump({
                'images': images_data,
                'annotations': annos_data, 
                'categories': cats},fw, indent=1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(usage=_usg)

    parser.add_argument('--anno-dir',
                        help = 'Directory containing annotation json files.',
                        type = str,
                        required= True)
    
    parser.add_argument('--catids',
                        help='The json file containg the category id to name map.',
                        type = str,
                        required = True)

    parser.add_argument('--outname',
                        help='The full path including extension to save the merged json file.',
                        default = 'merged_annotations.json')
    
    parser.add_argument('--verbose',
                        help='How oftern to print progress. Set to -1 to not.',
                        default = -1,
                        type = int)
    parser.add_argument('--keep-blanks',
                        help='Whether to keep data from images without annotations',
                        default = True,
                        type = bool)
    
    args = parser.parse_args()
    
    merge_annotations(anno_dir = args.anno_dir,
                         catids = args.catids, 
                         out_name = args.outname, 
                         verbose= args.verbose,
                         keep_blank_annotations=args.keep_blanks)
