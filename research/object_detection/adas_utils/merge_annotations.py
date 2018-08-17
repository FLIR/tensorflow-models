import json 
import os 
import argparse

"""
Merges annotations found in Annotations directory of
ADAS datasets into one large annotations json file
which is labeled to be consistent with MSCOCO's
Annotations jsons. 

The catids are from the catids.json from fruitbasket.

Example usage:
python merge_annotations.py --anno-dir {$PATH_TO_ANNOTATIONS}\
                            --catids {$PATH_TO_CATEGORY_IDS}\
"""


def merge_annotations(anno_dir,catids,out_name, 
                        verbose = 0, keep_blank_annotations = False):
    num_images = len(os.listdir(anno_dir))

    images_data = [0]*num_images
    annos_data = [0]*num_images
    blank_annos = []
    # Add the images and annotations

    for i, anno_file in enumerate(os.listdir(anno_dir)):
        path = anno_dir+anno_file
        if i % 100 == 0 and verbose > 0:
            msg = "Processing annotation file %d/%d"%(i,num_images)
            print(msg)
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

            images_data[i] = img_data
            annos_data[i] = an_data
    
    if not keep_blank_annotations:
        images_data = [images_data[k] for k in range(num_images) if not k in blank_annos ]
        annos_data = [annos_data[k] for k in range(num_images) if not k in blank_annos ]
    
    # annos_data: list of list of dict -> list of dict
    annos_data = [item for sublist in annos_data for item in sublist]
    with open(catids, 'r') as fcat:
        cats = json.load(fcat)

    merged_annotations = {'images': images_data, 'annotations': annos_data, 'categories': cats}

    with open(out_name, 'w') as fw:
        json.dump(merged_annotations,fw, indent=1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--anno-dir',
                        help = 'Directory containing annotation json files.',
                        type = str,
                        required= True)
    
    parser.add_argument('--catids',
                        help='The json file containg the category id to name map.',
                        type = str,
                        required = True)

    parser.add_argument('--out-name',
                        help='The full path including extension to save the merged json file.',
                        default = 'merged_annotations.json')
    
    parser.add_argument('--verbose',
                        help='Set to 1 if you want to see progress',
                        default = 0,
                        type = int)
    parser.add_argument('--keep-blanks',
                        help='Whether to keep data from images without annotations',
                        default = False,
                        type = bool)
    args = parser.parse_args()
    catids = args.catids
    anno_dir = args.anno_dir
    out_name = args.out_name
    verbose = args.verbose
    keep_em = args.keep_blanks

    merge_annotations(anno_dir = anno_dir,
                         catids = catids, 
                         out_name = out_name, 
                         verbose= verbose,
                         keep_blank_annotations=keep_em)
