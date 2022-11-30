from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import tensorflow as tf
import os

import wandb



class WandbCallbackGANConditional(Callback):

    def __init__(self, wandb):
        self.wandb = wandb
        self.times = [i for i in range(1024)]
        self.model_path = f'./checkpoints/model-{wandb.run.id}'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        super().__init__()


    def on_epoch_end(self, epoch, logs=None):

        # *** SAVING GENERATIONS ***

        noise = tf.random.normal(shape=(3, 100))
        labels = tf.cast(tf.reshape(tf.linspace(23, 5, 3), shape=(3, 1)), tf.float32)
        labels = (labels - 10.981254577636719) / 7.1911773681640625
        noise_and_labels = tf.concat([noise, labels], axis=1)

        generations = self.model.generator(noise_and_labels)

        a, b = -1, 1
        data_min, data_max = -2444.0, 2544.0
        generations_denormalized = data_min + (generations - a) * (data_max - data_min) / (b - a)

        fig, axes = plt.subplots(1, 3, figsize=(40, 10))
        
        for channel in range(16):
            axes[0].plot(self.times, generations_denormalized[0, :, channel])
            axes[1].plot(self.times, generations_denormalized[1, :, channel])
            axes[2].plot(self.times, generations_denormalized[2, :, channel])

        axes[0].grid(True)
        axes[1].grid(True)
        axes[2].grid(True)

        wandb.log({'epoch': epoch,
                   'discriminator_loss': logs['discriminator_loss'],
                   'generator_loss': logs['generator_loss'],
                   'generations': fig})        

        # *** SAVING MODEL WEIGHTS ***
        
        if (epoch + 1) % 5 == 0:
            self.model.save_weights(os.path.join(self.model_path, f'model-{wandb.run.id}-epoch-{epoch+1}.h5'))
        self.model.save_weights(os.path.join(self.wandb.run.dir, f'model-{wandb.run.id}.h5'))
        self.wandb.save(f'model-{wandb.run.id}.h5')