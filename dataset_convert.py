"""
Training deep mmwave+imu odometry from pseudo ground truth
"""
import pickle
import os
import tensorflow as tf
from os.path import join
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import numpy as np
import h5py
import matplotlib as mpl
import yaml
mpl.use('Agg')
import glob


# utiliy

from utility.data_loader import load_data_multi


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'feature0': _float_feature(feature0),
      'feature1': _float_feature(feature1),
      'feature2': _float_feature(feature2),
      'feature3': _float_feature(feature3),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def tf_serialize_example(f0,f1,f2,f3):
  tf_string = tf.py_function(
    serialize_example,
    (f0, f1, f2, f3),  # Pass these args to the above function.
    tf.string)      # The return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar.

def main():
    print('For mmwave+imu odom!')

    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.full_load(f)

    data_dir = cfg['mvo']['multimodal_data_dir']
    tf_data_dir = cfg['mvo']['tf_data_dir']
    # grap training files
    training_files = sorted(glob.glob(join(data_dir, 'train', '*.h5')))

    for e in range(1):
        for i, training_file in enumerate(training_files):
            print('---> Loading training file: {}', training_file.split('/')[-1])
            n_chunk, x_mm_t, x_imu_t, y_t = load_data_multi(training_file, 'mmwave_middle')
            # generate random length sequences
            len_x_i = x_mm_t[0].shape[0] # ex: length of sequence is 300
            x_mm_1_list = []
            x_mm_2_list = []
            x_imu_list = []
            y_label_list = []
            for j in range(len_x_i-1):
                x_mm_1, x_mm_2, x_imu, y_label = [], [], [], []
                seq_idx_1 = j
                seq_idx_2 = j+1
                x_mm_1.extend(x_mm_t[0][seq_idx_1, :, :, :])
                x_mm_2.extend(x_mm_t[0][seq_idx_2, :, :, :])
                x_imu.extend(x_imu_t[0][seq_idx_2, :, :]) # for 10 imu data
                y_label.extend(y_t[0][seq_idx_1, :])
                x_mm_1, x_mm_2, x_imu, y_label = np.array(x_mm_1), np.array(x_mm_2), \
                                                 np.array(x_imu), np.array(y_label)
                skip = False
                for x in y_label[3:6]:
                    if np.abs(x) > 140:
                        skip = True
                if skip:
                    print(y_label)
                    continue
                x_mm_1_list.append(x_mm_1[0])
                x_mm_2_list.append(x_mm_2[0])
                x_imu_list.append(x_imu)
                y_label_list.append(y_label)
            
            features_dataset = tf.data.Dataset.from_tensor_slices((np.array(x_mm_1_list),np.array(x_mm_2_list), np.array(x_imu_list), np.array(y_label_list)))
                
            path = tf_data_dir + training_file.split("turtle_seq_")[1].split(".")[0] + "/"
            tf.data.experimental.save(
                features_dataset, path, compression='GZIP', 
            )           
            with open(path + 'element_spec', 'wb') as out_:  # also save the element_spec to disk for future loading
                pickle.dump(features_dataset.element_spec, out_)

if __name__ == "__main__":
    os.system("hostname")
    main()
