from skimage.io import imread, imsave
import numpy as np


def image_to_binary(img_path):
    inp_image = imread(img_path)
    img_mnist_format = np.array(inp_image)[:, :, 3] / 255
    mask = img_mnist_format != 0
    img_mnist_format[mask] += 0.9 - img_mnist_format.max()
    img_mnist_format = np.expand_dims(img_mnist_format, axis=3)
    img_mnist_format = np.expand_dims(img_mnist_format, axis=0)
    return img_mnist_format
# image_to_binary('/home/constantin/Desktop/projects/rl/start_project/paint_tool/temp_paint.png')
