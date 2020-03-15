# coding: utf-8
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import keras.backend as K
from keras import optimizers
from keras.activations import relu, sigmoid, softmax
from keras.models import Model
from keras.layers import Dense, LSTM, GRU, Reshape, BatchNormalization, Input, Conv2D, MaxPooling2D, Lambda, Bidirectional, ZeroPadding2D
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import cv2
import os
import string
import pyarabic.araby as araby
import tensorflow as tf
from tensorflow.python.client import device_lib
import h5py
import math
# Check all available devices if GPU is available
print(device_lib.list_local_devices())

# utils

letters = [ch for ch in araby.LETTERS+string.printable+u'٠١٢٣٤٥٦٧٨٩']


def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

# # Loss and train functions, network architecture


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# data loader
def train_data_generator(img_w=432, img_h=32, no_channels=1, text_max_len=40, batch_size=64, train_size=0.8, dataset_path='../dataset/dataset.h5'):
    dataset = h5py.File(dataset_path, 'r')
    train_indexes = list(range(int(train_size*dataset['images'].shape[0])))
    while True:
        images = np.zeros((batch_size, img_h, img_w, no_channels))
        text = np.zeros((batch_size, text_max_len))
        label_length = np.zeros((batch_size, 1), dtype=np.int64)
        input_length = np.ones((batch_size, 1), dtype=np.int64) * 114
        # choose randomly 128 samples of training data from hard disk and load them into memory
        i = 0
        samples_indexes = np.random.choice(train_indexes, size=batch_size)
        for j in samples_indexes:
            images[i] = dataset['images'][j]
            text[i] = dataset['text'][j]
            label_length[i] = dataset['label_length'][j]
            i += 1
        inputs = {
            'input_1': images,
            'the_labels': text,
            'input_length': input_length,
            'label_length': label_length
        }
        outputs = {'ctc': np.zeros(batch_size)}
        yield (inputs, outputs)


def test_data_generator(img_w=432, img_h=32, no_channels=1, text_max_len=40, batch_size=64, train_size=0.8, dataset_path='../dataset/dataset.h5'):
    dataset = h5py.File(dataset_path, 'r')
    test_indexes = list(
        range(int(train_size*dataset['images'].shape[0]), dataset['images'].shape[0]))
    while True:
        images = np.zeros((batch_size, img_h, img_w, no_channels))
        text = np.zeros((batch_size, text_max_len))
        label_length = np.zeros((batch_size, 1), dtype=np.int64)
        input_length = np.ones((batch_size, 1), dtype=np.int64) * 114
        # choose randomly 32 samples of training data from hard disk and load them into memory
        i = 0
        samples_indexes = np.random.choice(test_indexes, size=batch_size)
        for j in samples_indexes:
            images[i] = dataset['images'][j]
            text[i] = dataset['text'][j]
            label_length[i] = dataset['label_length'][j]
            i += 1
        inputs = {
            'input_1': images,
            'the_labels': text,
            'input_length': input_length,
            'label_length': label_length
        }
        outputs = {'ctc': np.zeros(batch_size)}
        yield (inputs, outputs)


# network >>>>> CNN + RNN + CTC loss
# input with shape of height=32 and width=432
inputs = Input(shape=(32, 432, 1))
inputs_padded = ZeroPadding2D((1, 1), input_shape=(32, 432))(inputs)
conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs_padded)
conv_1 = BatchNormalization()(conv_1)
conv1_padded = ZeroPadding2D((1, 1))(conv_1)
conv_1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_padded)
conv_1_2 = BatchNormalization()(conv_1_2)
pool_1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_1_2)
####################################################################################
pool_1_p = ZeroPadding2D((1, 1))(pool_1)
conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1_p)
conv_2 = BatchNormalization()(conv_2)
conv2_padded = ZeroPadding2D((1, 1))(conv_2)
conv_2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_padded)
conv_2_2 = BatchNormalization()(conv_2_2)
pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_2_2)
###################################################################################
pool_2_p = ZeroPadding2D((1, 1))(pool_2)
conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2_p)
conv_3 = BatchNormalization()(conv_3)
conv3_p = ZeroPadding2D((1, 1))(conv_3)
conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3_p)
conv_4 = BatchNormalization()(conv_4)
pool_4 = MaxPooling2D(pool_size=(2, 1))(conv_4)
##################################################################################
conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
conv_5 = BatchNormalization()(conv_5)
batch_norm_5 = BatchNormalization()(conv_5)
conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPooling2D(pool_size=(3, 1))(batch_norm_6)
conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)
conv_7 = BatchNormalization()(conv_7)
# feature maps to sequence
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
# bidirectional LSTM layers with units=256
blstm_1 = Bidirectional(
    LSTM(256, return_sequences=True, dropout=0.2, kernel_initializer='he_normal'))(squeezed)
blstm_1 = BatchNormalization()(blstm_1)
blstm_2 = Bidirectional(LSTM(256, return_sequences=True,
                             dropout=0.2, kernel_initializer='he_normal'))(blstm_1)

outputs = Dense(len(letters)+10, activation='softmax')(blstm_2)

test_model = Model(inputs, outputs)

print(test_model.summary())

test_model.save('test_model.h5')

max_label_len = 40
labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    [outputs, labels, input_length, label_length])

# model to be used at training time
train_model = Model(
    inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
# load weights
# train_model.load_weights("ckpts/CRNN--05--95.947.hdf5")

# load weights
# train_model.load_weights("ckpts/CRNN--15--1.870.hdf5")
epochs = 50
#adam = optimizers.adam(lr=1e-5)
# sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9,
#                      nesterov=True, clipnorm=5)
train_model.compile(
    loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizers.adadelta())
# early_stop = EarlyStopping(
#     monitor='val_loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(
    filepath='ckpts/CRNN--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='val_loss', verbose=1, mode='min', period=5)
train_model.fit_generator(generator=train_data_generator(),
                          validation_data=test_data_generator(),
                          steps_per_epoch=250000//64,
                          validation_steps=50000//64,
                          epochs=epochs,
                          verbose=1,
                          callbacks=[checkpoint])
