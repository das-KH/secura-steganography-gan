from keras.layers import Conv2D, Input, Concatenate, LeakyReLU, BatchNormalization, AveragePooling2D, Dense, Flatten
from keras.models import Model
from utils import InceptionBlock, load_data
from numpy import add, zeros, ones


class SecuraSteganographyGAN(object):
  def __init__(self, input_shape):

    self.input_shape = input_shape
    # extracting the height of the image; required for the inception block
    self.image_size = input_shape[0]

    self.generator = self.build_generator()

    # build the discriminator model
    self.discriminator = self.build_discriminator()

    # Compile the discriminator
    self.discriminator.compile(optimizer='adam', loss='binary_crossentropy')

    # build the adversarial model
    cover_image = Input(shape=self.input_shape, name="Cover_Image_Input")
    secret_image = Input(shape=self.input_shape, name="Secret_Image_Input")

    stego_image, reconstructed_image = self.generator([cover_image, secret_image])

    # For the adversarial model, we do not train the discriminator
    self.discriminator.trainable = False

    # The discriminator determines the input is a stego-image or not
    discriminator_preds = self.discriminator(stego_image)

    # Define a coef for the contribution of discriminator loss to total loss
    g = 0.001

    # Build and compile the adversarial model
    self.adversarial = Model(inputs=[cover_image, secret_image], \
                              outputs=[stego_image, reconstructed_image, discriminator_preds], name="Steganography_GAN")

    self.adversarial.compile(optimizer='adam',loss=['mse', 'mse', 'binary_crossentropy'],loss_weights=[1.0, 0.85, g])


    self.adversarial.summary()

    # history variable to plot the loss later on
    self.history = {'epoch':[], 'd_loss': [], 'g_loss': []}

  def build_generator(self):
    # Inputs design
    cover_input = Input(shape=self.input_shape, name='cover_input')
    secret_input = Input(shape=self.input_shape, name='secret_input')

    combined_input = Concatenate(axis=-1)([cover_input, secret_input])

    x = Conv2D(16, (3, 3), padding='same')(combined_input)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0.2)(x)

    x = InceptionBlock(self.image_size, 16, 32, "InceptionBlock_1")(x)
    x = InceptionBlock(self.image_size, 32, 64, "InceptionBlock_2")(x)
    x = InceptionBlock(self.image_size, 64, 128, "InceptionBlock_3")(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0.2)(x)
    stego = Conv2D(3, (1, 1), padding='same', activation='tanh', name="Steganographic_Image")(x)

    # Decoder architecture
    depth = 32
    L1 = Conv2D(depth, 3, padding='same')(stego)
    L1 = BatchNormalization(momentum=0.9)(L1)
    L1 = LeakyReLU(alpha=0.2)(L1)

    L2 = Conv2D(depth*2, 3, padding='same')(L1)
    L2 = BatchNormalization(momentum=0.9)(L2)
    L2 = LeakyReLU(alpha=0.2)(L2)

    L5 = Conv2D(depth, 3, padding='same')(L2)
    L5 = BatchNormalization(momentum=0.9)(L5)
    L5 = LeakyReLU(alpha=0.2)(L5)

    out = Conv2D(3, 1, padding='same', activation='tanh', name="Reconstructed_Secret_Image")(L5)
    decoder_output = out

    generator = Model(inputs=[cover_input, secret_input], outputs=[stego, decoder_output], name="Generator")
    # generator.summary()
    return generator

  def build_discriminator(self):
    img_input = Input(shape=self.input_shape, name='Discriminator_Input')
    L1 = Conv2D(8, 3, padding='same')(img_input)
    L1 = BatchNormalization(momentum=0.9)(L1)
    L1 = LeakyReLU(alpha=0.2)(L1)
    L1 = AveragePooling2D(pool_size=5, strides=2, padding='same')(L1)

    L2 = Conv2D(16, 3, padding='same')(L1)
    L2 = BatchNormalization(momentum=0.9)(L2)
    L2 = LeakyReLU(alpha=0.2)(L2)
    L2 = AveragePooling2D(pool_size=5, strides=2, padding='same')(L2)

    L3 = Conv2D(32, 1, padding='same')(L2)
    L3 = BatchNormalization(momentum=0.9)(L3)
    L3 = AveragePooling2D(pool_size=5, strides=2, padding='same')(L3)

    L4 = Conv2D(64, 1, padding='same')(L3)
    L4 = BatchNormalization(momentum=0.9)(L4)
    L4 = AveragePooling2D(pool_size=5, strides=2, padding='same')(L4)

    L5 = Conv2D(128, 3, padding='same')(L4)
    L5 = BatchNormalization(momentum=0.9)(L5)
    L5 = LeakyReLU(alpha=0.2)(L5)
    L5 = AveragePooling2D(pool_size=5, strides=2, padding='same')(L5)
    F = Flatten()(L5)

    L7 = Dense(128)(F)
    L8 = Dense(1, activation='sigmoid', name="D_output")(L7)

    discriminator = Model(inputs=img_input, outputs=L8, name='Discriminator')
    # discriminator.summary()

    return discriminator

  def train(self, cover_path, secret_path, epochs, batch_size, checkpoint_dir):
    cover_images,secret_images = load_data(cover_path, secret_path)
    # Adversarial ground truths
    original = ones((batch_size, 1))
    encrypted = zeros((batch_size, 1))

    for epoch in range(1, epochs+1):
        batch= 1
        # take one batch of cover and secret images and train them at each iteration
        for batch_cover_images,batch_secret_images in zip(cover_images,secret_images):
            # Predict the generator output for these images
            imgs_stego, _ = self.generator.predict([batch_cover_images, batch_secret_images])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(batch_cover_images, original)

            d_loss_encrypted = self.discriminator.train_on_batch(imgs_stego, encrypted)
            d_loss = 0.5 * add(d_loss_real, d_loss_encrypted)

            # Train the generator
            g_loss = self.adversarial.train_on_batch([batch_cover_images, batch_secret_images], [batch_cover_images, batch_secret_images, original])

            # Print the progress
            print(f'Epoch:{epoch}/{epochs} Batch:{batch}/{len(cover_images)} [D loss: {d_loss}] [G loss: {g_loss[0]}]')
            # increment batch number
            batch +=1
            
        # logging the losses
        self.history['epoch'].append(epoch)
        self.history['d_loss'].append(d_loss)
        self.history['g_loss'].append(g_loss[0])

        # saving the models every 10 epochs
        if epoch % 10  == 1:
            print(f'[INFO] Saving the models to {checkpoint_dir}')
            self.generator.save(checkpoint_dir+f'/Epoch{epoch}/Generator_{epoch}.h5')
            # self.discriminator.save(checkpoint_dir+f'/Epoch{epoch}/Discriminator_{epoch}.h5')
            # self.adversarial.save(checkpoint_dir+f'/Epoch{epoch}/GAN_{epoch}.h5')
