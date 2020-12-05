#2D layers - densely connected or fully connected (dense layers)
#3D layers - recurrent layers e.g. LSTM
#4D layers - 2D convolution layers e.g. Conv2D

#layers will take tensors of a certain shape and output tensors of a certain shape

from keras import layers
from keras import models
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential, load_model

from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

import altair as alt




#specifying layer that accepts 2D tensor with first dimension (axis 0 = 784),
#and second layer unspecified (could be anything)

#the layer will return a tensor with axis 0 = 32 (transforms from 784 to 32)
layer = layers.Dense(32, input_shape=(784,))
layer

#input_shape argument isn't needed for Keras b/c it is inferred

#a two layer model using the Sequential class (for linear stacks of layers):
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss='mse',
                metrics=['accuracy'])

#would need to have defined the input tensor and target tensor first
model.fit(input_tensor, target_tensor, batch_size=128, epochs=5)

#a two layer model defined with functional API for acyclic graphs of layers (allows customization)
#this is further detailed in ch 7 of text
input_tensor = layers.Input(shape=(784,))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=input_tensor, outputs = output_tensor)

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss='mse',
                metrics=['accuracy'])

#would need to have defined the target tensor
model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)


#binary classification example - positive or negative movie review:

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

train_data[1]
train_labels[1]

max([max(sequence) for sequence in train_data])

word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])

decoded_review = ''.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[1]]
)
decoded_review

#def to one hot encode data
#look at how to do this from original word data and label binnarize using sklearn
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

x_train[0]

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

y_train[0]

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
                loss = 'binary_crossentropy',
                metrics=['accuracy'])


#set 10000 samples apart from original training data to support monitoring model
x_val = x_train[:10000]
partial_x_train=x_train[10000:]

y_val = y_train[:10000]
partial_y_train=y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


#OVERFITTING - use the history to determine if your model is overfitting
#this is a history of the loss, accuracy, val_loss and val_accuracy for each epoch
history_dict = history.history
history_dict.keys()
history_dict

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)
epochs

#plot the loss
plt.plot(epochs, loss_values, 'bo', label='training loss')
plt.plot(epochs, val_loss_values, 'b', label='validation loss')
plt.title('training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

train_val_loss = (
                    plt.figure(figsize=(10, 6)),
                    plt.plot(epochs, loss_values, 'bo', label='training loss'),
                    plt.plot(epochs, val_loss_values, 'b', label='validation loss'),
                    plt.title('training and validation loss'),
                    plt.xlabel('epochs'),
                    plt.ylabel('loss'),
                    plt.legend()
                )

#plot the accuracy
accuracy_values = history_dict['accuracy']
val_accuracy_values = history_dict['val_accuracy']

plt.plot(epochs, accuracy_values, 'bo', label='training accuracy')
plt.plot(epochs, val_accuracy_values, 'b', label='validation accuracy')
plt.title('training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

train_val_accuracy = (
                        plt.figure(figsize=(10,6)),
                        plt.plot(epochs, accuracy_values, 'bo', label='training accuracy'),
                        plt.plot(epochs, val_accuracy_values, 'b', label='validation accuracy'),
                        plt.title('training and validation accuracy'),
                        plt.xlabel('epochs'),
                        plt.ylabel('accuracy'),
                        plt.legend()
                        )



#try to improve upon model (stop overfitting)
model2 = models.Sequential()
model2.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model2.add(layers.Dense(16, activation ='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))

model2.compile(optimizer = 'rmsprop',
                loss = 'binary_crossentropy',
                metrics=['accuracy'])

model2.fit(partial_x_train,
            partial_y_train,
            epochs=4,
            batch_size=512,
            validation_data=(x_val, y_val))

history_dict_2 = model2.history.history
history_dict_2.keys()
history_dict_2

epochs2 = range(1, len(loss_values_2) + 1)
epochs2

loss_values_2 = history_dict_2['loss']
val_loss_values_2 = history_dict_2['val_loss']

train_val_loss_2 = (plt.figure(figsize=(10, 6)),
                    plt.plot(epochs2, loss_values_2, 'bo', label='training loss'),
                    plt.plot(epochs2, val_loss_values_2, 'b', label='validation loss'),
                    plt.title('training and validation loss'),
                    plt.xlabel('epochs'),
                    plt.ylabel('loss'),
                    plt.legend()
                    )

accuracy_values_2 = history_dict_2['accuracy']
val_accuracy_values_2 = history_dict_2['val_accuracy']


train_val_accuracy_2 = (plt.figure(figsize=(10, 6)),
                        plt.plot(epochs2, accuracy_values_2, 'bo', label='training accuracy'),
                        plt.plot(epochs2, val_accuracy_values_2, 'b', label='validation accuracy'),
                        plt.title('training and validation accuracy'),
                        plt.xlabel('epochs'),
                        plt.ylabel('accuracy'),
                        plt.legend()
                        )

#retrain model from scratch using train and test data
model_f = models.Sequential()
model_f.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model_f.add(layers.Dense(16, activation='relu'))
model_f.add(layers.Dense(1, activation='sigmoid'))

model_f.compile(optimizer='rmsprop',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])

model_f.fit(x_train, y_train, epochs=4, batch_size=512)

history_dict_f = model_f.history.history

results = model_f.evaluate(x_test, y_test)
results

#use the model to predict on new data
predictions = model.predict(x_test)
predictions[2]


#also try taking the raw imdb data and using label binnarizer (or whatever the nlp package was) to preprocess
