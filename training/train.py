import tensorflow as tf
import time
import tqdm
from training.generate_image import generate_and_save_images


class Training:

    def __init__(self, batch_size, noise_dim, generator, discriminator, seed, colors):
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.001)
        self.seed = seed
        self.colors = colors

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

    def train_start(self, dataset, epochs, time_for_training):
        if epochs == "":
            start = time.time()
            self.train_step(dataset[0])
            one_epoch = time.time() - start
            epochs = int((time_for_training*60)/one_epoch)
        else:
            epochs = int(epochs)

        for epoch in range(epochs):
            start = time.time()
            for image_batch in tqdm.tqdm(dataset):
                self.train_step(image_batch)

            generate_and_save_images(self.generator, epoch + 1, self.seed, self.colors)

            print('Эпоха {} тренировалась {} секунд'.format(epoch + 1, time.time() - start))

        generate_and_save_images(self.generator, epochs, self.seed, self.colors)
        self.generator.save(r"model.h5")


