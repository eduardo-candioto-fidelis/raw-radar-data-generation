import os
import sys
sys.path.append('src/models/callbacks/')
sys.path.append('src/models/hyperparemeters/')

import gc
import numpy as np

import tensorflow as tf

from callback_conditional import WandbCallbackGANConditional
from model_cwgangp import build_gan

import wandb


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def load_weights(gan, weights_path):
    gan.built = True
    gan.load_weights(weights_path)

    return gan
    

def load_dataset_labaled(path_data, path_label, batch_size):
    data = np.load(path_data)
    labels_generator, labels_discriminator = np.load(path_label)

    data_tensor = tf.convert_to_tensor(data)
    labels_generator_tensor = tf.convert_to_tensor(labels_generator)
    labels_discriminator_tensor = tf.convert_to_tensor(labels_discriminator)
    del data
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


def main():

    gan, batch_size, epochs, gan_config = build_gan()

    #os.environ["WANDB_RESUME"] = "allow"
    #os.environ["WANDB_RUN_ID"] = "19fh3wpc"

    wandb.init(
        project='test',
        entity='eduardo-candioto',
        name='conditional-nooutiliers',
        config=gan_config,
        #pid="19fh3wpc", 
        #resume="allow"
    )
    
    if wandb.run.resumed:
        print('Resuming Run...')
        model_file = wandb.restore(f'model-{wandb.run.id}.h5').name
        gan = load_weights(gan, model_file)
    
    dataset = load_dataset_labaled('./data/preprocessed/EXP_17_M_chirps_scaled.npy', './data/preprocessed/EXP_17_M_chirps_labels.npy', batch_size)

    print(f'\n\n--------------------- Run: {wandb.run.name} ---------------------------\n\n')

    gan.fit(dataset,
            initial_epoch=wandb.run.step, epochs=epochs, batch_size=batch_size,
            callbacks=[WandbCallbackGANConditional(wandb)])


if __name__ == '__main__':
    main()
