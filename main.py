from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from training import models, dataset, train


BUFFER_SIZE = 10000
BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 1
seed = tf.random.normal([num_examples_to_generate, noise_dim])
checkpoint_dir = 'training_checkpoints'

train_dataset = dataset.make_dataset(BUFFER_SIZE, BATCH_SIZE)

models = models.ModelMaking()
generator = models.make_generator_model()
discriminator = models.make_discriminator_model()


checkpoint = tf.train.Checkpoint(generator_optimizer=tf.keras.optimizers.Adam(1e-4),
                                      discriminator_optimizer=tf.keras.optimizers.Adam(1e-4),
                                      generator=generator,
                                      discriminator=discriminator)

trainer = train.Training(batch_size=BATCH_SIZE, noise_dim=noise_dim, generator=generator,
                                 discriminator=discriminator, checkpoint_dir=checkpoint_dir, checkpoint=checkpoint, seed=seed)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
trainer.train_start(train_dataset, EPOCHS)

