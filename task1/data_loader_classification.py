import os
import numpy as np
import tensorflow as tf



def read_labeled_image_list(img_list_path, img_dir):
  """Reads a .txt file containing pathes and labeles
  Args:
    img_list_path: a .txt file with one /path/to/image with one label per line
    img_dir: path of directory that contains images
  Returns:
    List with all filenames
  """
  f = open(img_list_path, 'r')
  img_paths = []
  labs = []
  for line in f:
    img_name, lab = line[:-1].split(' ')
    img_paths.append(img_dir + img_name)
    labs.append(int(lab))


  f.close()
  return img_paths, labs

def read_images_from_disk(input_queue):
  """Consumes a single filename and label as a ' '-delimited string
  Args:
    filename_and_label_tensor: A scalar string tensor
  Returns:
    Two tensors: the decoded image, and the string label
  """
  lab = input_queue[1]
  img_path = tf.read_file(input_queue[0])
  img = tf.image.decode_png(img_path, channels=3)
  return img, lab

def get_loader(root, batch_size, config, split=None, shuffle=True):
  """ Get a data loader for tensorflow computation graph
  Args:
    root: Path/to/dataset/root/, a string
    batch_size: Batch size, a integer
    split: Data for train/val/test, a string
    shuffle: If the data should be shuffled every epoch, a boolean
  Returns:
    img_batch: A (float) tensor containing a batch of images.
    lab_batch: A (int) tensor containing a batch of labels.
  """
  img_paths_np, labs_np = read_labeled_image_list(root+'/devkit/'+split+'.txt', root+'/imgs/')

  with tf.device('/cpu:0'):
    img_paths = tf.convert_to_tensor(img_paths_np, dtype=tf.string)
    labs = tf.convert_to_tensor(labs_np, dtype=tf.int64)

    input_queue = tf.train.slice_input_producer([img_paths, labs],
                  shuffle=shuffle, capacity=10*batch_size)

    img, lab = read_images_from_disk(input_queue)

    img.set_shape([32, 32, 3])
    img = tf.cast(img, tf.float32)

    img = tf.image.per_image_standardization(img)
    if (split=='train'):
        img=tf.image.random_flip_left_right(img)
        img = tf.pad(img, [[4,4],[4,4],[0,0]])
        img = tf.random_crop(img, [32,32,3])

    img_batch, lab_batch = tf.train.batch([img, lab], num_threads=1,
                           batch_size=batch_size, capacity=10*batch_size)

  return img_batch, lab_batch
