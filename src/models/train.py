import sys

sys.path.append('./utils/')
sys.path.append('./gan/')
sys.path.append('./gan/models/')

import tensorflow as tf

from utils import load_weights, load_dataset
from callbacks import WandbCallbackGAN
from model_swgangp import build_gan

import wandb


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():

    gan, batch_size, epochs, gan_config = build_gan()

    #os.environ["WANDB_RESUME"] = "allow"
    #os.environ["WANDB_RUN_ID"] = "iv7wbhhw"

    wandb.init(
        project='tests',
        entity='eduardo-candioto',
        name='sequential-1',
        config=gan_config,
        #id="iv7wbhhw", 
        #resume="allow"
    )
    
    if wandb.run.resumed:
        print('Resuming Run...')
        model_file = wandb.restore(f'model-{wandb.run.id}.h5').name
        gan = load_weights(gan, model_file)
    
    dataset = load_dataset('./data/real/EXP_17_M_frames_scaled.npy', batch_size)

    print(f'\n\n--------------------- Run: {wandb.run.name} ---------------------------\n\n')

    gan.fit(dataset,
            initial_epoch=wandb.run.step, epochs=epochs, batch_size=batch_size,
            callbacks=[],
            verbose=2)


if __name__ == '__main__':
    main()
