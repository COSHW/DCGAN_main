import tensorflow as tf
import os
from IPython import display
import time
import tqdm
from training.generate_image import generate_and_save_images


class Training:

    def __init__(self, batch_size, noise_dim, generator, discriminator, checkpoint_dir, checkpoint, seed):
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = checkpoint
        self.seed = seed

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train_start(self, dataset, epochs):
        for epoch in range(epochs):
            start = time.time()
            i=1
            for image_batch in tqdm.tqdm(dataset):
                i+=1
                self.train_step(image_batch)

            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            generate_and_save_images(self.generator, epoch + 1, self.seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        generate_and_save_images(self.generator, epochs, self.seed)


