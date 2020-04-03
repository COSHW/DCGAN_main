import tensorflow as tf


class ModelMaking:
    def make_generator_model_28_32_64(self, batch_size, img_size, colors):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(int(img_size/4)*int(img_size/4)*batch_size, use_bias=False, input_shape=(100,)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Reshape((int(img_size/4), int(img_size/4), batch_size)))
        assert model.output_shape == (None, int(img_size/4), int(img_size/4), batch_size)

        model.add(tf.keras.layers.Conv2DTranspose(int(batch_size/2), (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, int(img_size/4), int(img_size/4), int(batch_size/2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(int(batch_size/4), (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(img_size/2), int(img_size/2), batch_size/4)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(colors, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, img_size, img_size, 3)

        return model

    def make_generator_model_128(self, batch_size, img_size, colors):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(int(img_size/16)*int(img_size/16)*batch_size, use_bias=False, input_shape=(100,)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Reshape((int(img_size/16), int(img_size/16), batch_size)))
        assert model.output_shape == (None, int(img_size/16), int(img_size/16), batch_size)

        model.add(tf.keras.layers.Conv2DTranspose(int(batch_size / 2), (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(img_size / 8), int(img_size / 8), int(batch_size / 2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(int(batch_size/4), (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(img_size/4), int(img_size/4), int(batch_size/4))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(int(batch_size/8), (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(img_size/2), int(img_size/2), int(batch_size/8))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(colors, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, img_size, img_size, 3)

        return model

    def make_generator_model_256(self, batch_size, img_size, colors):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(int(img_size/32)*int(img_size/32)*batch_size, use_bias=False, input_shape=(100,)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Reshape((int(img_size/32), int(img_size/32), batch_size)))
        assert model.output_shape == (None, int(img_size/32), int(img_size/32), batch_size)

        model.add(tf.keras.layers.Conv2DTranspose(int(batch_size/2), (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(img_size/16), int(img_size/16), int(batch_size/2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(int(batch_size/4), (5, 5), strides=(4, 4), padding='same', use_bias=False))
        assert model.output_shape == (None, int(img_size/4), int(img_size/4), int(batch_size/4))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(int(batch_size/8 if batch_size==128 else batch_size/4), (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(img_size/2), int(img_size/2), int(batch_size/8 if batch_size==128 else batch_size/4))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(colors, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, img_size, img_size, 3)

        return model

    def make_discriminator_model(self, batch_size, img_size, colors):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(int(batch_size/8), (5, 5), strides=(2, 2), padding='same',
                                         input_shape=[img_size, img_size, colors]))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(int(batch_size/4), (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(int(batch_size/2), (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        if int(img_size) == 128:

            model.add(tf.keras.layers.Conv2D(batch_size, (5, 5), strides=(2, 2), padding='same'))
            model.add(tf.keras.layers.LeakyReLU())
            model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))

        return model

    def make_discriminator_model_256(self, batch_size, img_size, colors):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(int(batch_size/16), (5, 5), strides=(2, 2), padding='same',
                                         input_shape=[img_size, img_size, colors]))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(int(batch_size/8), (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(int(batch_size/4), (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))

        return model




