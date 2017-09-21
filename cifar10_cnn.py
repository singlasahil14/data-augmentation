'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import os
import pickle
import numpy as np
import pandas as pd
import argparse
import time

if K.backend()=='tensorflow':
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

parser = argparse.ArgumentParser(description='Data augmentation parameter search')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to run')
parser.add_argument('--subset', type=float, default=1, help='subset of training data to use')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--result', type=str,  default=randomhash, help='path to save the final model')
parser.add_argument('--transform', type=str, choices=['height', 'width', 'rotation', 'zoom'], 
                    default='width', help='type of transform to apply')
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
num_classes = 10
save_dir = os.path.join(os.getcwd(), args.result)
model_name = 'keras_cifar10_trained_model.h5'

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, y_train = unison_shuffled_copies(x_train, y_train)

num_subset = int(np.floor(args.subset*len(x_train)))
x_train = x_train[:num_subset]
x_test = x_test[:num_subset]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate Adam optimizer
opt = keras.optimizers.Adam(lr=3e-4, decay=1e-6)

# Let's train the model using Adam
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('Not using data augmentation.')
datagen = ImageDataGenerator(featurewise_center=True, 
                             samplewise_center=True)
# Compute (std, mean) quantities required for feature-wise normalization
datagen.fit(x_train)
hist = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True)
print(hist.history.keys())

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
train_metrics = {k: hist.history[k] for k in ('acc','loss')}
pd_train_metrics = pd.DataFrame(train_metrics)
pd_train_metrics.to_csv(os.path.join(save_dir, 'train_metrics.csv'))

eval_metrics = {k: hist.history[k] for k in ('val_acc', 'val_loss')}
pd_eval_metrics = pd.DataFrame(eval_metrics)
pd_eval_metrics.to_csv(os.path.join(save_dir, 'eval_metrics.csv'))

model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Load label names to use in prediction results
label_list_path = 'datasets/cifar-10-batches-py/batches.meta'

keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
datadir_base = os.path.expanduser(keras_dir)
if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
label_list_path = os.path.join(datadir_base, label_list_path)

with open(label_list_path, mode='rb') as f:
    labels = pickle.load(f)

# Evaluate model with test data set and share sample prediction results
evaluation = model.evaluate_generator(datagen.flow(x_test, y_test,
                                      batch_size=batch_size),
                                      steps=x_test.shape[0] // batch_size)

print('Model Accuracy = %.2f' % (evaluation[1]))

