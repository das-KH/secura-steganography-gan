# from models import SecuraSteganographyGAN
from utils import get_args
args = get_args()

print(args)


input_shape = (256, 256, 3)


# GAN = SecuraSteganographyGAN(input_shape=input_shape)
# 
# GAN.train(cover_path=cover_images_dir, secret_path=secret_images_dir,  batch_size=BATCH_SIZE, epochs= EPOCHS, checkpoint_dir=checkpoint_path)

