"""
@jroberts: 2/2019
    Takes a directory of tfrecords and prints the 
    number of examples in each record.
    Example usage:
        python count_records.py Path/TO/RECORDS/DIR/
        
"""
import tensorflow as tf 
import sys
import os 

input_dir = sys.argv[1]

tf_records_filenames = os.listdir(input_dir)
tf_records_filenames = [ os.path.join(input_dir,tfrecord) for tfrecord in tf_records_filenames]


print('Counting records in: {}'.format(tf_records_filenames))

total = 0
for fn in tf_records_filenames:
    c = 0
    for record in tf.python_io.tf_record_iterator(fn):
        c += 1
    print("Record: {} has {} records".format(fn,c))
    total += c

print('Total Records: {}'.format(total))