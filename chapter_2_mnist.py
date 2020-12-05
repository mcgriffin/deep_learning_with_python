import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras import models
from keras.models import Sequential, load_model
from keras import layers
from keras.utils import to_categorical
from matplotlib import pyplot as plt


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape
train_images[0]
digit = train_images[0]
plt.imshow(digit, cmap=plt.cm.binary)


#building the network architecture
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))


#building out the compilation step
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

#reshape train images and transform to float
train_images_reshaped = train_images.reshape((60000, 28*28))
train_images_reshaped = train_images_reshaped.astype('float32') / 255

#make train labels categorical
train_labels = to_categorical(train_labels)


#reshape test images and transform to float
test_images_reshaped = test_images.reshape((10000, 28*28))
test_images_reshaped = test_images_reshaped.astype('float32') / 255

#make test labels categorical
test_labels = to_categorical(test_labels)

#loss is the mismatch between predictions (y_pred) and actual (y_actual) - lower loss is better
network.fit(train_images_reshaped, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images_reshaped, test_labels)
print('test_acc:', test_acc)

#data representations for neural networks with numpy (TENSORS)

#scalar (aka zero dim tensor)
zero_dim = np.array(12)
zero_dim
zero_dim.ndim
zero_dim.shape
zero_dim.dtype

#vector (aka 1D tensor)
one_dim = np.array([12, 3, 6, 14, 7])
one_dim
one_dim.ndim
one_dim.shape
one_dim.dtype

#matrix (aka 2D tensor)
two_dim = np.array([[5, 78, 2, 34, 0],
                    [6, 79, 3, 35, 1],
                    [7, 80, 4, 36, 2]])
two_dim
two_dim.ndim
two_dim.shape
two_dim.dtype

#cube of numbers (aka 3D tensor)
num_cube = np.array([[[5, 78, 2, 34, 0],
            [6, 79, 3, 35, 1],
            [7, 80, 4, 36, 2]],
            [[5, 78, 2, 34, 0],
            [6, 79, 3, 35, 1],
            [7, 80, 4, 36, 2]],
            [[5, 78, 2, 34, 0],
            [6, 79, 3, 35, 1],
            [7, 80, 4, 36, 2]]])
num_cube
num_cube.ndim
num_cube.shape
num_cube.dtype

#4D tensor
num_quad = np.array([[[
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6]]],
                    [[
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6]]],
                    [[
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6]]]],
                    )
num_quad
num_quad.ndim
num_quad.shape
num_quad.dtype

num_quint = np.array([[[
                        [[1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6]],

                        [[1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6]],

                        [[1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6]],

                        [[1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6]],

                        [[1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6]]],

                        [[[1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6]],

                        [[1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6]],

                        [[1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6]],

                        [[1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6]],

                        [[1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, 5, 6]]],
                        ]])
num_quint
num_quint.ndim
num_quint.shape
num_quint.dtype

#looking back at mnist images (not reshaped)
train_images.shape
train_images.dtype
test_images.shape
test_images.dtype

#choosing the first image (aka selecting specific digit along the first axis)
first_image = train_images[0]
plt.imshow(first_image, cmap=plt.cm.binary)

#manipulating tensors in numpy
#the first axis is aka the BATCH AXIS, and typically represents the samples,
#or what you are trying to identify

#select the first 2 numbers in the set, represented by uint8

my_slice = train_images[0:2]
my_slice.ndim
my_slice.shape
my_slice.dtype
print(my_slice)


#the previous block is equivalent to the following, which outright
#accounts for each axis of the tensor:
my_slice_thorough = train_images[0:2, 0:28, 0:28]
my_slice_thorough.ndim
my_slice_thorough.shape
my_slice_thorough.dtype
print(my_slice_thorough)

#to select the bottom 16 x 16 pixels in the first image
bottom_16_16_pixels = train_images[1, 16:, 16:]
plt.imshow(bottom_16_16_pixels, cmap=plt.cm.binary)

#to select the bottom 16 x 16 pixels in all images
bottom_16_16_pixels_all = train_images[:, 16:, 16:]
#plt.imshow(bottom_16_16_pixels_all)

#to select the same bottom right pixels in all images using negative indexing
bottom_16_16_pixels_all_negative_indexing = train_images[:, -12:, -12:]
#plt.imshow(bottom_16_16_pixels_all_negative_indexing)

#tensor operations
#RELU: this function returns a positive number, or 0 if negative number
#found within 2D array
def naive_rectified_linear_unit(x):
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

two_d_tensor = np.array([[10, -2, -10, -3, -3],
                        [-5, -7, -8, -9, -10],
                        [1, 0, 1, 0, -1]])
naive_rectified_linear_unit(two_d_tensor)

#or, using numpy, element-wise relu:
rectified_linear_unit_operation = np.maximum(two_d_tensor, 0.)
print(rectified_linear_unit_operation)

#this function takes two 2D tensors and adds them together
def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x

naive_add(two_d_tensor, two_d_tensor)

#or, using numpy, element-wise addition:
added_matrices = two_d_tensor + two_d_tensor
print(added_matrices)

def naive_subtract(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] -= y[i, j]
    return x

naive_subtract(two_d_tensor, two_d_tensor)

#or, using numpy, element-wise subtraction:
subtracted_matrices = two_d_tensor - two_d_tensor
print(subtracted_matrices)


#broadcasting
#start with a 2D matrix and vector that you want to add
#they are not the same shape, so you can't add yet
#broadcasting is used to extend y so you get a shape that
#matches X.
#y is simply repeated 32 times to facilitate the addition in this case
X = np.random.randint(0, 10, (32, 10))
y = np.random.randint(0, 10, (10,))

X
y

#loss is the quantity you attempt to minimize during training
#optimizer is the way in which the gradient of the loss will update parameters
    #e.g. RMSProp, SGD with momentum
    
