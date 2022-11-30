from imageio import save
import tensorflow as tf
import numpy as np
import sys

sys.path.append('./src/models/hyperparemeters/')
from model_cwgangp import build_gan
from utils import load_weights

import os

import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.random.set_seed(34643)



def reshape_generations(generated):
    #return tf.reshape(generated, shape=(generated.shape[0], 16, 1024))
    return tf.transpose(generated, [0, 2, 1])


def denormalize(generated, data_min, data_max, a, b):
    return data_min + (generated - a) * (data_max - data_min) / (b - a)


def save_conditional_generations(model, epoch, name):
    gan, _, _, _ = build_gan()
    gan = load_weights(gan, f'./checkpoints/model-{model}/model-{model}-epoch-{epoch}.h5')

    generator = gan.generator
    start_time = time.time()
    distances = np.linspace(25.0, 0.0, 6000).reshape((6000, 1))
    #distances = tf.repeat([[10.0]], repeats=[50], axis=0)
    #distances = tf.repeat([[6.391821368455132]], repeats=[50], axis=0)
    #distances = tf.repeat([[15.0]], repeats=[3], axis=0)
    #distances = tf.repeat([[28.0]], repeats=[50], axis=0)
    #distances = np.load('data/EXP_17_M_labels.npy')[0] 
    #distances = np.array([np.unique(distances)]).T * -1    

    distances_scaled = (distances - 10.981254577636719) / 7.1911773681640625
    
    noise = tf.random.normal(shape=[len(distances), 100])

    noise_and_distances = tf.concat(
        [noise, distances_scaled], axis=1
    )

    generated = reshape_generations(generator.predict(noise_and_distances))
    generated_denormalized = denormalize(generated, -3884.0, 4772.0, -1, 1)
    generated_denormalized = np.round(generated_denormalized, 0)
    print("%s" % (time.time() - start_time))
    np.save(f'./data/generated/{name}.npy', generated_denormalized)



model = sys.argv[1]
epoch = sys.argv[2]
name = sys.argv[3]

save_conditional_generations(model, epoch, name)
