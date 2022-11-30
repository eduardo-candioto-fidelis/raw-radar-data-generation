import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import re
import os
import gc



#def load_weights(gan, weights_path):
#    gan.built = True
#    gan.load_weights(weights_path)
#
#    timestamp = re.findall('\d+\.\d+', weights_path)[0]
#    x = re.findall('epoch-\d.', weights_path)[0]
#    initial_epoch = int(re.findall('\d+', x)[0])
#
#    return gan, initial_epoch, timestamp


def load_weights(gan, weights_path):
    gan.built = True
    gan.load_weights(weights_path)

    return gan
    

def load_dataset(path, batch_size):
    data = np.load(path)
    tensor = tf.convert_to_tensor(data)
    dataset = tf.data.Dataset.from_tensor_slices(tensor)
    del data
    del tensor
    gc.collect()

    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    print(f'\nNumber of batches: {len(dataset)}\n')

    return dataset


def load_dataset_labaled(path_data, path_label, batch_size):
    data = np.load(path_data)
    labels = np.load(path_label)
    labels_generator = np.asmatrix(labels[0])
    labels_discriminator = np.asmatrix(labels[1])

    data_tensor = tf.convert_to_tensor(data)
    labels_generator_tensor = tf.convert_to_tensor(labels_generator)
    labels_discriminator_tensor = tf.convert_to_tensor(labels_discriminator)
    del data
    del labels
    del labels_discriminator
    del labels_generator
    gc.collect()

    dataset = tf.data.Dataset.from_tensor_slices((data_tensor, labels_generator_tensor, labels_discriminator_tensor))
    del data_tensor
    del labels_generator_tensor
    del labels_discriminator_tensor
    gc.collect()

    dataset = dataset.shuffle(10_000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    print(f'\nNumber of batches: {len(dataset)}')

    return dataset