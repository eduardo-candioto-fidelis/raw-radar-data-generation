import sys 
sys.path.append('./src/models/trainers')

from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv1DTranspose, ReLU, Conv1D, LeakyReLU, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from cwgangp import CWGANGP



def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)

    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


def build_gan():
    BATCH_SIZE = 64
    EPOCHS = 50_000 

    CODINGS_SIZE = 100

    generator = Sequential([
        InputLayer((101,)),
        Dense(4 * 256, input_shape=[CODINGS_SIZE]),
        Reshape([4, 256]),
        ReLU(),
        Conv1DTranspose(128, kernel_size=25, strides=4, padding='same', activation='relu', input_shape=[4, 256]),
        Conv1DTranspose(64, kernel_size=25, strides=4, padding='same', activation='relu', ),
        Conv1DTranspose(32, kernel_size=25, strides=4, padding='same', activation='relu', ),
        Conv1DTranspose(16, kernel_size=25, strides=4, padding='same', activation='tanh', )
    ], name='generator')

    discriminator = Sequential([
        InputLayer((1024, 17)),
        Conv1D(32, kernel_size=25, strides=4, padding='same',  input_shape=[1024, 16]),
        LeakyReLU(0.2),
        Conv1D(64, kernel_size=25, strides=4, padding='same'),
        LeakyReLU(0.2),
        Conv1D(128, kernel_size=25, strides=4, padding='same'),
        LeakyReLU(0.2),
        Conv1D(256, kernel_size=25, strides=4, padding='same'),
        LeakyReLU(0.2),
        Flatten(),
        Dense(1, activation='linear'),
    ], name='discriminator')

    gan = CWGANGP(discriminator, generator, CODINGS_SIZE, discriminator_extra_steps=5)

    gan.compile(
        discriminator_optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
        generator_optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
        discriminator_loss_fn=discriminator_loss,
        generator_loss_fn=generator_loss
    )

    gan_config = {
        'learning_rate': 0.0001,
        'batch_size': BATCH_SIZE,
        'codings_size': CODINGS_SIZE,
        'architecture': {
            'generator': generator.get_config(), 'discriminator': discriminator.get_config()
        },
    }

    return gan, BATCH_SIZE, EPOCHS, gan_config
