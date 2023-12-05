# from tensorflow.data import AUTOTUNE
# from keras.utils import image_dataset_from_directory
# from keras.layers import Rescaling, Conv2D, Add, MaxPooling2D, Concatenate, Input, Activation
# from keras.models import Model
import argparse


def get_args():
  des = ''' To train the model, pass in the following arguments:
    -i  to define the shape of the input images: HEIGHT WIDTH CHANNELS
    '''
  parser = argparse.ArgumentParser(description=des)
  parser.add_argument("-input-shape", "-i", type=tuple)
  parser.add_argument("-cover-images","-c", type=str)
  parser.add_argument("-secret-images", "-s", type=str)
  parser.add_argument("-batch-size", "-bs", type=int)
  parser.add_argument("-epochs", "-ep", type=int)
  parser.add_argument("-checkpoint-path", "-chkpt", type=str)

  args = parser.parse_args()
  return args

# # fucntion to load the cover and secret images as batches
# # returns  two variables: cover_images and secret Images
# def load_data(cover_image_dir, secret_image_dir,image_size, batch_size):
#   cover_images = image_dataset_from_directory(
#                     cover_image_dir,
#                     validation_split=0,
#                     shuffle=True,
#                     label_mode=None,
#                     image_size=(image_size,image_size),
#                     batch_size=batch_size)

#   secret_images = image_dataset_from_directory(
#                     secret_image_dir,
#                     validation_split=0,
#                     shuffle=True,
#                     label_mode=None,
#                     image_size=(image_size,image_size),
#                     batch_size=batch_size)

#   # to rescale the pixel values of the images [-1 , 1] 
#   normalization_layer = Rescaling(1./127.5, offset=-1)

#   #applying the rescaling layer to all the batches
#   cover_images = cover_images.map(lambda x: (normalization_layer(x)))
#   secret_images = secret_images.map(lambda x: (normalization_layer(x)))
  
#   # caching and prefetching the batches
#   cover_images = cover_images.cache().prefetch(buffer_size=AUTOTUNE)
#   secret_images = secret_images.cache().prefetch(buffer_size=AUTOTUNE)

#   return cover_images,secret_images


# # the inception block used inside the encoder architecture i.e, inside the generator
# def InceptionBlock(image_size, filters_in, filters_out, block_name):
  input_layer = Input(shape=(image_size, image_size, filters_in))  # Adjusted input shape

  tower_filters = int(filters_out / 4)

  tower_1 = Conv2D(tower_filters, 1, padding='same', activation='relu')(input_layer)

  tower_2 = Conv2D(tower_filters, 1, padding='same', activation='relu')(input_layer)
  tower_2 = Conv2D(tower_filters, 3, padding='same', activation='relu')(tower_2)

  tower_3 = Conv2D(tower_filters, 1, padding='same', activation='relu')(input_layer)
  tower_3 = Conv2D(tower_filters, 5, padding='same', activation='relu')(tower_3)

  tower_4 = MaxPooling2D(pool_size=(1, 1), padding='same', strides=(1, 1))(input_layer)
  tower_4 = Conv2D(tower_filters, 1, padding='same', activation='relu')(tower_4)

  concat = Concatenate(axis=-1)([tower_1, tower_2, tower_3, tower_4])

  # Adjusted the number of filters in the Conv2D layer to match concat
  res_link = Conv2D(filters_out, 1, padding='same', activation='relu')(input_layer)

  output = Add()([concat, res_link])
  output = Activation('relu')(output)

  InceptionBlock = Model(inputs=input_layer, outputs=output, name=block_name)
  return InceptionBlock