'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)
import random as rnd
rnd.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)
import os
os.environ['PYTHONHASHSEED'] = '0'
import pickle
import pandas as pd
import json
import argparse
import time

import keras
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import Sequence
from keras import backend as K

if K.backend()=='tensorflow':
    cfg = K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=cfg)
    K.set_session(sess)

kw_map = {'height': 'height_shift_range', 'width': 'width_shift_range', 'rotation': 'rotation_range', 
          'zoom': 'zoom_range', 'shear': 'shear_range'}
gap_map = {'height': 0.1, 'width': 0.1, 'rotation': 30, 'zoom': 0.1, 'shear': 0.1}
parser = argparse.ArgumentParser(description='Data augmentation parameter search')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=350, help='number of epochs to run')
parser.add_argument('--subset', type=float, default=1, help='subset of training data to use')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--result', type=str,  default=randomhash, help='path to save the final model')
parser.add_argument('--transform', type=str, choices=['height', 'width', 'rotation', 'zoom', 'shear'], 
                    default='width', help='type of transform to apply')
parser.add_argument('--gap', type=float, default=None, help='gap with which to try transform')
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
num_classes = 10
save_dir = os.path.join(os.getcwd(), args.result)
assert not(os.path.exists(save_dir)), "result dir already exists!"
os.makedirs(save_dir)
transform_kw = kw_map[args.transform]
if args.gap is None:
    transform_gap = gap_map[args.transform]
else:
    transform_gap = args.gap
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

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
    
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def get_model():
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5), padding='same',
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
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

model = get_model()
initial_weights = model.get_weights()
prev_best_loss = np.inf
transform_kw_val = transform_gap
best_loss_map = {}
num_train_batches = -(-x_train.shape[0] // batch_size)
eval_batch_size = 100
num_eval_batches = -(-x_test.shape[0] // eval_batch_size)
val_datagen = ImageDataGenerator(featurewise_center=True, samplewise_center=True)
val_datagen.fit(x_train)
val_generator = val_datagen.flow(x_test, y_test, batch_size=eval_batch_size)
while True:
    model.set_weights(initial_weights)
    if not(transform_gap == 0):
        print(transform_kw, transform_kw_val)
    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    gen_args = {'featurewise_center': True, 'samplewise_center': True, 
                transform_kw: transform_kw_val, 'horizontal_flip': False, 
                'fill_mode': 'reflect'}
    datagen = ImageDataGenerator(**gen_args)
    datagen.fit(x_train)
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 
            steps_per_epoch=num_train_batches,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=num_eval_batches,
            max_queue_size=100*batch_size, workers=3)
    best_val_loss = min(hist.history['val_loss'])
    best_loss_map[transform_kw_val] = best_val_loss
    print('previous best loss', prev_best_loss)
    print('current best loss', best_val_loss)
    
    # Save model and weights
    if not(transform_gap == 0):
        sub_dir = os.path.join(save_dir, str(transform_kw_val))
        os.makedirs(sub_dir)
    else:
        sub_dir = save_dir
    pd_metrics = pd.DataFrame(hist.history)
    pd_metrics.to_csv(os.path.join(sub_dir, 'metrics.csv'))
    
    model_path = os.path.join(sub_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    print('')

    if prev_best_loss < best_val_loss:
        break
    prev_best_loss = best_val_loss

    if transform_gap == 0.:
        break
    transform_kw_val += transform_gap
if not(transform_gap == 0):
    print_str = 'Best loss values for different ranges of ' + args.transform + ' transform'
    best_loss_list = [(transform_kw, 'best validation loss')] + list(best_loss_map.items())
else:
    print_str = 'Best loss value for no transform'
    best_loss_list = [('No transform', 'best validation loss')] + list(best_loss_map.items())
print(print_str)
for ele1,ele2 in best_loss_list:
    print("{:<24}{:<48}".format(ele1,ele2))
print('')

config_str = json.dumps(best_loss_map) + '\n'
config_file = os.path.join(save_dir, 'config')
config_file_object = open(config_file, 'w')
config_file_object.write(config_str)
config_file_object.close()
