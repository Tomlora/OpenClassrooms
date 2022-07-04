import tensorflow as tf
import keras
import torch

def verification_gpu():
    """ Verification que le gpu est disponible"""
    print('Tensorflow : ')
    print(tf.__version__)
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    print(tf.test.is_built_with_cuda())
    print(tf.test.gpu_device_name())
    print(tf.config.list_physical_devices('GPU'))

    print('Torch : ')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
