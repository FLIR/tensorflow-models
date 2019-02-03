# A wrapper for the full tfrecord formation from 
# a FLIR dataset structure.
# Assumes the directory structure is 
# 
# BASE_DIR
#     |--Annotations/
#     |--Data/

# Set default parameters
gpus=0
BASE_DIR=GIMME_A_NAME
color=false
preview=false

# Hardcoded where the categorey ids are for adas tasks. 
CATIDS=/mnt/fruitbasket/users/jroberts/ADAS/catids.json

spaces="============================================"
spaces=$spaces$spaces

print_usage() {
  printf "Description: A wrapper for the full tfrecord formation from a FLIR
         dataset structure. Assumes the directory structure is 
            
          BASE_DIR
            |--Annotations/
            |--(Preview)Data/
        
        Must have the tensorflow-models/object_detection requirements installed. 
        See:
          https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
        
        for specifics.

        Creates the directory and records
          BASE_DIR
            |--Annotations/
            |--(Preview)Data/
            |--tfrecords/
              |--adas.record-00000-00001
            |--(Preview)Data_colored/

        Where (Preview)Data_colored if -c is set. 
        
        Flags:         
          -d: Directory to look for data and annotations. Should be absolute path. 
          -c: Whether to run pseudo_color.py for the model. (Default $color)
          -p: Whether to look for PreviewData. (Default $preview)
          -g: Which gpu to use. (Default $gpus)
"
}

while getopts 'hd:g:cp' flag; do
  case "${flag}" in
    d) BASE_DIR="${OPTARG}" ;;
    g) gpus="${OPTARG}" ;;
    c) color=true ;;
    p) preview=true ;;
    h) print_usage && exit;;
    *) print_usage && exit
       exit 1 ;;
  esac
done

#
# Move into object detection.
#
cd object_detection

info_flags="
Flags Found:
  gpus:        $gpus
  colorize:    $color
  Add Preview: $preview
  Base Dir:    $BASE_DIR
  "

echo $info_flags

# Check for Preview
if [ $preview = true ];then
  data_base=PreviewData;
  else
  data_base=Data
fi 

data_dir=$BASE_DIR/$data_base
annotation_dir=$BASE_DIR/Annotations
annotations_file=$BASE_DIR/merged_annotations.json
records_dir=$BASE_DIR/tfrecords

info_directories="
Directories: 
  Data Directory:        $data_dir 
  Annotations Directory: $annotation_dir 
  TFRecord Directory:    $records_dir 
"

echo "
$spaces
$info_directories
$spaces
"

# Look for annotations file
echo "Checking for $annotations_file"
if [ -f $annotations_file ]; then
  echo "Found $annotations_file
  "
else
  echo "Creating $annotations_file
  "

  merge_annos=$PWD/adas_utils/merge_annotations.py
  outname=$annotations_file

  echo "Running:
  python $merge_annos --anno-dir $annotation_dir  \\
                      --catids $CATIDS \\ 
                      --outname $outname
                      "
  python $merge_annos --anno-dir $annotation_dir --catids $CATIDS --outname $outname --verbose 500

fi

echo $spaces

# Color the images if necessary
if [ $color = true ]; then
  data_dir_out=$data_dir"_colored"
  echo "Coloring the data and leaving a copy here: $data_dir_out
        "
  pseudo_color=$PWD/adas_utils/pseudo_color.py 

  echo "Running:
  python $pseudo_color \\
                      --image-dir $data_dir \\
                      --out-dir $data_dir_out \\
                       "
  
  python $pseudo_color --image-dir $data_dir --out-dir $data_dir_out

  echo "Combining new and old directories."

  cp -n $data_dir/* $data_dir_out
  data_dir=$data_dir_out

else
  echo "Not coloring."
  echo "WARNING: If the any greyscale images in the data set
              have only 1 channel then the tfrecord will be 
              formed, but using the tfrecord will crash later.
              "
fi

echo "$spaces
Starting the tfrecord creation.
"

create_adas=$PWD/dataset_tools/create_adas_tf_record.py

echo " 
Running:
python $create_adas \\
            --image_dir=$data_dir \\
            --annotations_file=$annotations_file
            --output_dir=$records_dir
          "

python $create_adas --image_dir=$data_dir --annotations_file=$annotations_file --output_dir=$records_dir

echo $spaces

echo "
This data was created using tensorflow-models/research/make_adas_records.bash.
catids:      $CATIDS

$info_flags
$info_directories
" > $records_dir/INFO.txt


