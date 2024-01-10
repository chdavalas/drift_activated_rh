#!/usr/bin/env python

# Copyright 2020 chdavalas
#
# chdavalas@gmail.com, cdavalas@hua.gr
#
# This program is free software; you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.
#

from collections import defaultdict
import pandas as pd
import tensorflow_datasets as tfds
import random
from datetime import datetime
from time import time, strftime, sleep
from sys import setrecursionlimit
import argparse
from tensorflow import keras
import numpy as np
import tensorflow as tf
from numpy import argmax
from math import sqrt, log, ceil, log2
from statistics import mean

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CUDNN_DETERMINISTIC"] = "true"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

#tf.debugging.set_log_device_placement(True)
print(datetime.now())



#
# Configuration
#
SEED = 32221100
BATCH = 32
PRE_EPOCHS = 0

LOSS = tf.keras.losses.CategoricalCrossentropy()
LEARNING_RATES = [0.01]
LEARNING_RATE_DECAY_STEPS = 1.0
LEARNING_RATE_DECAY_RATE = 0.05
OPTIMIZERS = ["sgd"]
DATASET_NAME = "cifar10"
MODEL_NAME   = "resnet32"
BATCHES_IN_DATASET = 50000 // BATCH  # 1562
SIZE_OF_STREAM = BATCH * 312 * 5
SIZE_OF_TASK   = BATCH * 312
USE_RANDOM_TASK_SELECTION = False
USE_RANDOM_TASK_SIZE = False
RANDOM_TASK_SIZE_SIGMA = BATCH * 5
#TASK_CLASSES = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
TASK_CLASSES = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
ALL_CLASSES = [cl for classes_per_task in TASK_CLASSES for cl in classes_per_task]

#TASK_CLASSES = [ [0,1,2,3,4,5,6,7,8,9] for _ in range(10) ]
#ALL_CLASSES  = [0,1,2,3,4,5,6,7,8,9]


# Number of per-class images for pretraining/buffer
# if num_per_class=n then the buffer has n*no_of_classes elements with equal parts per class
PRETRAIN_NUM_PER_CLASS = 0
BUFFER_NUM_PER_CLASS   = 50
TEST_BATCH = 10000 // len(TASK_CLASSES)
CONR_N_RH_REPEAT = [1, 10, 25, 50]  # continuous rehearsal with repeats
DRIFTA_MAX_RH_REPEAT = [50]  # drift activated maximum repeat
MIX_LEN = 16  # NUMBER OF OLD ELEMENTS IN A MIXED BATCH
SAVE_INTERMEDIATE_MODELS = False

# Drift parameters
LAM = 0.2  # Lambda hyper parameter. Optimal is 0.2 according to the respective paper
# According to the respective paper 100,400,1000 are the basic variables for this one
# ARL (Average run length) , configures the sensitivity of drift detection (higher means less sensitive)
ARL = 100
MODEL_CACHE = "model_cache/"
# NAME_OF_INIT_MODEL = MODEL_CACHE + DATASET_NAME + str(SEED)
# NAME_OF_INIT_MODEL = MODEL_CACHE + "pretrained-50-per-class-cifar1032221100"
NAME_OF_INIT_MODEL = "model_cache/NO_TRAINING-{}{}".format(DATASET_NAME, SEED)

# Hybrid method parameters
ERROR_THR = 0.2  # If Z_t error exceeds this value, then train no matter what
# If, after MAX_NOTRAIN mini-batches the drift has not been activated, then train no matter what again.
MAX_NOTRAIN = 20



def print_all_parameters():
    print("SEED={}".format(SEED))
    print("BATCH={}".format(BATCH))
    print("PRE_EPOCHS={}".format(PRE_EPOCHS))
    print("DATASET_NAME={}".format(DATASET_NAME))
    print("MODEL_NAME={}".format(MODEL_NAME))
    print("size_of_stream (in images)={}".format(SIZE_OF_STREAM))
    print("size_of_stream (in batches)={}".format(SIZE_OF_STREAM // BATCH))
    print("size of task (in images)={}".format(SIZE_OF_TASK))
    print("size of task (in batches)={}".format(SIZE_OF_TASK // BATCH))
    print("task_classes={}".format(TASK_CLASSES))
    print("all_classes={}".format(ALL_CLASSES))
    print("USE_RANDOM_TASK_SELECTION={}".format(USE_RANDOM_TASK_SELECTION))
    print("USE_RANDOM_TASK_SIZE={}".format(USE_RANDOM_TASK_SIZE))
    print("RANDOM_TASK_SIZE_SIGMA={}".format(RANDOM_TASK_SIZE_SIGMA))
    print("PRETRAIN_NUM_PER_CLASS={}".format(PRETRAIN_NUM_PER_CLASS))
    print("BUFFER_NUM_PER_CLASS={}".format(BUFFER_NUM_PER_CLASS))
    print("TEST_BATCH={}".format(TEST_BATCH))
    print("CONR_N_RH_REPEAT={}".format(CONR_N_RH_REPEAT))
    print("DRIFTA_MAX_RH_REPEAT={}".format(DRIFTA_MAX_RH_REPEAT))
    print("MIX_LEN={}".format(MIX_LEN))
    print("LAM={}".format(LAM))
    print("ARL={}".format(ARL))
    print("ERROR_THR={}".format(ERROR_THR))
    print("MAX_NOTRAIN={}".format(MAX_NOTRAIN))
    print("NAME_OF_INIT_MODEL={}".format(NAME_OF_INIT_MODEL))



def custom_dataset(
    name,
    train_classes,
    test_classes,
    tr_imgs_per_class,
    ts_imgs_per_class,
    seed,
    shuffle=False,
    shuffle_buffer_size=60000,
    perm_set = {} ):
    
    def show_samples(dts, sample_per_row=3):
        import matplotlib.pyplot as plt
        figure = plt.figure(figsize=(10,10))
        
        rand_dts = dts.shuffle(10000,seed=123)\
                      .take(sample_per_row**2)
        
        for i, (img, lbl) in enumerate(rand_dts):
            plt.subplot(\
                sample_per_row, sample_per_row, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            show_img = tf.squeeze(img)
            plt.imshow(show_img)
            plt.title(str(lbl).replace("tf.Tensor",""))
        plt.show()


    if name!="psmnist":
        perm_set = {}  #Permutations only for mnist
    else:
        name = "mnist" #we call the mnist dataset and then add the permutations
    assert name in ["mnist", "cifar10", "cifar100"], "choose: mnist, cifar10, cifar100"


    num_cl = {"mnist": 10, "cifar10": 10, "cifar100": 100}

    for class_tr, class_ts in zip(train_classes, test_classes):
        assert class_tr in range(num_cl[name]) and class_ts in range(num_cl[name]), (
            "choice should be within (0-" + str(num_cl[name]) + ")"
        )

    assert tr_imgs_per_class != 0, "choices !=0 or -1 for all"
    assert tr_imgs_per_class >= -1, "choices !=0 or -1 for all"

    assert ts_imgs_per_class != 0, "choices !=0 or -1 for all"
    assert ts_imgs_per_class >= -1, "choices !=0 or -1 for all"

    print("[Acquire dataset and preproc]\n")

    (train_ds, test_ds), info = tfds.load(
        name, split=["train", "test"], with_info=True, as_supervised=True
    )

    all_classes = [
        info.features["label"].str2int(st) for st in info.features["label"].names
    ]

    original_im_shape = info.features["image"].shape

    print("[Choose images by label name]:" + str(train_classes))

    train_ds_part = None
    test_ds_part  = None


    def prm_if_psmnist(perm, x ,y):

        x = tf.squeeze(x)
        x = tf.reshape(x, [784])
        x = tf.gather(x, perm)
        x = tf.reshape(x, [28,28])
        x = tf.expand_dims(x, -1)
        
        return x, y



    if train_classes != []:
        train_ds_part = train_ds.filter(lambda x, y: y == train_classes[0])\
                                .take(tr_imgs_per_class)

        for cl in train_classes[1:]:
            new_part = train_ds.filter(lambda x, y: y == cl)\
                               .take(tr_imgs_per_class)

            train_ds_part = train_ds_part.concatenate(new_part)


    if test_classes != []:
        test_ds_part = test_ds.filter(lambda x, y: y == test_classes[0])\
                              .take(ts_imgs_per_class)
            
        for cl in test_classes[1:]:
            new_part = test_ds.filter(lambda x, y: y == cl)\
                              .take(ts_imgs_per_class)

            test_ds_part = test_ds_part.concatenate(new_part)


    if perm_set!={} and train_classes != []:

        prm_fun = (lambda x,y :prm_if_psmnist(list(perm_set.values())[0], x ,y))
        tr_perm_ds_part = train_ds_part.map(prm_fun)
        
        print(hash(str(list(perm_set.values())[0])))
        for prm in list(perm_set.values())[1:]:
            print(hash(str(prm)))
            
            prm_fun = (lambda x,y :prm_if_psmnist(prm, x ,y))
            new_perm_ds_part = train_ds_part.map(prm_fun)

            tr_perm_ds_part  = tr_perm_ds_part.concatenate(new_perm_ds_part)
    
        train_ds_part = tr_perm_ds_part #use former variable name for ease
    
    if perm_set!={} and test_classes != []:

        prm_fun = (lambda x,y :prm_if_psmnist(list(perm_set.values())[0], x ,y))
        ts_perm_ds_part = test_ds_part.map(prm_fun)
        
        print(hash(str(list(perm_set.values())[0])))
        for prm in list(perm_set.values())[1:]:
            print(hash(str(prm)))
            
            prm_fun = (lambda x,y :prm_if_psmnist(prm, x ,y))
            new_perm_ds_part = test_ds_part.map(prm_fun)
            
            ts_perm_ds_part  = ts_perm_ds_part.concatenate(new_perm_ds_part)
    
        test_ds_part  = ts_perm_ds_part #use former variable name for ease


    def proc(img, label):
        img = tf.cast(img, dtype=tf.float32)
        img = img / 255.0
        label = tf.one_hot(label, depth=len(all_classes))
        return img, label
    
    
    if train_classes != []:
        train_ds_part = train_ds_part.map(proc)

    if test_classes != []:
        test_ds_part = test_ds_part.map(proc)


    if shuffle:
        train_ds_part = train_ds_part.shuffle(shuffle_buffer_size, seed=seed)

    return train_ds_part, test_ds_part, all_classes, original_im_shape




def residual_block(n_filters, strd, last=False):
    def BLOCK(input_layer):

        shortcut = input_layer

        from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.activations import relu, elu

        layer = Conv2D(
            n_filters,
            kernel_size=(3, 3),
            kernel_regularizer=l2(1e-5),
            padding="same",
            strides=(strd, strd),
        )(input_layer)

        layer = BatchNormalization()(layer)
        layer = Activation(relu)(layer)

        layer = Conv2D(
            n_filters, kernel_size=(3, 3), kernel_regularizer=l2(1e-5), padding="same"
        )(layer)
        layer = BatchNormalization()(layer)

        if strd != 1:

            projection_layer = Conv2D(
                n_filters,
                kernel_size=(1, 1),
                padding="same",
                kernel_regularizer=l2(1e-5),
                strides=(strd, strd),
            )(shortcut)

            block = Add()([layer, projection_layer])
            if not last:
                block = tf.nn.relu(block)
            return block

        else:
            block = Add()([layer, shortcut])
            if not last:
                block = tf.nn.relu(block)
            return block

    return BLOCK




def ResNet18(in_sh, classes_, activ_last=False):

    with tf.device("/GPU:0"):

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input,
            Activation,
            Conv2D,
            BatchNormalization,
        )
        from tensorflow.keras.layers import (
            AveragePooling2D,
            MaxPooling2D,
            Flatten,
            Dense,
        )
        from tensorflow.keras.layers import (
            GlobalAveragePooling2D,
            GlobalMaxPooling2D,
            Dropout,
        )
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.activations import relu, softmax

        input_layer = Input(shape=in_sh)

        layer = Conv2D(
            8, kernel_size=(3, 3), kernel_regularizer=l2(1e-5), padding="same"
        )(input_layer)

        layer = BatchNormalization()(layer)
        layer = Activation(relu)(layer)

        layer = residual_block(8, 1)(layer)
        layer = residual_block(8, 1)(layer)

        layer = residual_block(16, 2)(layer)
        layer = residual_block(16, 1)(layer)

        layer = residual_block(32, 2)(layer)
        layer = residual_block(32, 1)(layer)
        
        layer = residual_block(64, 2)(layer)
        layer = residual_block(64, 1, last=True)(layer)

        layer = GlobalAveragePooling2D()(layer)

        layer = Dense(classes_, kernel_regularizer=l2(1e-5))(layer)
        if activ_last:
            layer = Activation(softmax)(layer)

        model_ = Model(inputs=input_layer, outputs=layer)

        print("\nInput:" + str(input_layer))
        print("Output:" + str(model_.output) + "\n")

        return model_





def ResNet32(in_sh, classes_, activ_last=False):

    with tf.device("/GPU:0"):

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input,
            Activation,
            Conv2D,
            BatchNormalization,
        )
        from tensorflow.keras.layers import (
            AveragePooling2D,
            MaxPooling2D,
            Flatten,
            Dense,
        )
        from tensorflow.keras.layers import (
            GlobalAveragePooling2D,
            GlobalMaxPooling2D,
            Dropout,
        )
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.activations import relu, softmax

        input_layer = Input(shape=in_sh)

        layer = Conv2D(
            16, kernel_size=(3, 3), kernel_regularizer=l2(1e-5), padding="same"
        )(input_layer)

        layer = BatchNormalization()(layer)
        layer = Activation(relu)(layer)

        layer = residual_block(16, 1)(layer)
        layer = residual_block(16, 1)(layer)
        layer = residual_block(16, 1)(layer)
        layer = residual_block(16, 1)(layer)
        layer = residual_block(16, 1)(layer)

        layer = residual_block(32, 2)(layer)
        layer = residual_block(32, 1)(layer)
        layer = residual_block(32, 1)(layer)
        layer = residual_block(32, 1)(layer)
        layer = residual_block(32, 1)(layer)

        layer = residual_block(64, 2)(layer)
        layer = residual_block(64, 1)(layer)
        layer = residual_block(64, 1)(layer)
        layer = residual_block(64, 1)(layer)
        layer = residual_block(64, 1, last=True)(layer)

        layer = GlobalAveragePooling2D()(layer)

        layer = Dense(classes_, kernel_regularizer=l2(1e-5))(layer)
        if activ_last:
            layer = Activation(softmax)(layer)

        model_ = Model(inputs=input_layer, outputs=layer)

        print("\nInput:" + str(input_layer))
        print("Output:" + str(model_.output) + "\n")

        return model_


class ECDDetector:
    def __init__(self, lam=0.2, avg_run_len=400):
        self._lam = lam
        self._avg_run_len = avg_run_len
        self._t = 0
        self._p_0t = 0
        self._Z_t = 0
        self._X_t = 0
        self._diff_X_t = 0
        self._drift = False

    def _L_t(self):
        if self._avg_run_len == 100:
            return (
                2.76
                - 6.23 * self._p_0t
                + 18.12 * pow(self._p_0t, 3)
                - 312.45 * pow(self._p_0t, 5)
                + 1002.18 * pow(self._p_0t, 7)
            )
        elif self._avg_run_len == 400:
            return (
                3.97
                - 6.56 * self._p_0t
                + 48.73 * pow(self._p_0t, 3)
                - 330.13 * pow(self._p_0t, 5)
                + 848.18 * pow(self._p_0t, 7)
            )
        else:
            return (
                1.17
                + 7.56 * self._p_0t
                - 21.24 * pow(self._p_0t, 3)
                + 112.12 * pow(self._p_0t, 5)
                - 987.24 * pow(self._p_0t, 7)
            )

    def predict(self, model, images, labels):
        predicted_labels = argmax(tf.nn.softmax(model(images), axis=-1), axis=-1)
        real_labels = argmax(labels, axis=1)
        n = predicted_labels.shape[0]
        X_t = np.sum(predicted_labels != real_labels)
        self._diff_X_t = X_t / n - self._X_t
        self._X_t = X_t / n

        # X_t is sum, so second term is average
        self._p_0t = (self._t / (self._t + n)) * self._p_0t + (1 / (self._t + n)) * X_t
        self._t += n
        sxt = self._p_0t * (1 - self._p_0t)
        szt = (
            sqrt((self._lam / (2 - self._lam)) * (1 - (1 - self._lam) ** (2 * self._t)))
            * sxt
        )
        L_t = self._L_t()
        self._Z_t = (1 - self._lam) * self._Z_t + self._lam * (X_t / n)
       
        self._drift = (self._Z_t > self._p_0t + L_t * szt) or (
            self._X_t > self._p_0t + 2 * sxt
        )
        #self._drift = (self._Z_t > self._p_0t + L_t * szt) 
        
        return self._drift

    @property
    def X_t(self):
        return self._X_t

    @property
    def diff_X_t(self):
        return self._diff_X_t

    @property
    def Z_t(self):
        return self._Z_t

    @property
    def p_0t(self):
        return self._p_0t

    @property
    def drift(self):
        return self._drift

    @property
    def time(self):
        return self._t


class RehearsalBuffer:
    def __init__(self, model, images, labels):
        self._model = model
        self._images = images
        self._labels = labels

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def update(self, model, images, labels):
        pass


class DynamicRehearsalBuffer(RehearsalBuffer):
    def __init__(self, model, images, labels):
        super().__init__(model, images, labels)
        self._class_avgs = {}
        self._num_per_class = {}
        self._ind_per_class = {}
        self._thetas_per_class = {}
        self._thetas = self._compute_logits(images)

        for i, (theta, label) in enumerate(zip(self._thetas, self._labels)):
            lbl = argmax(label)
            if lbl not in self._class_avgs:
                self._class_avgs[lbl] = theta
                self._num_per_class[lbl] = 1
                self._ind_per_class[lbl] = [i]
                self._thetas_per_class[lbl] = [theta]
            else:
                self._class_avgs[lbl] += theta
                self._num_per_class[lbl] += 1
                self._ind_per_class[lbl].append(i)
                self._thetas_per_class[lbl].append(theta)

        for key in self._class_avgs.keys():
            self._class_avgs[key] = tf.scalar_mul(
                1 / self._num_per_class[key], self._class_avgs[key]
            )

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def update(self, model, new_images, new_labels):
        #k = time()
        a=time()
        new_thetas = self._compute_logits(new_images)
        print("logits:"+str(time()-a))
        buff_upd=0
        
        with tf.device("/GPU:0"):
            for theta_xi, nimg, nlbl in zip(new_thetas, new_images, new_labels):

                yi = argmax(nlbl)
                nyi = self._num_per_class[yi]

                self._class_avgs[yi] *= nyi / (nyi + 1)
                self._class_avgs[yi] += (1 / (nyi + 1)) * theta_xi

                max_dist = None
                max_dist_index = None


                class_thetas  = tf.convert_to_tensor(self._thetas_per_class[yi])
                class_avgs_yi = tf.repeat([self._class_avgs[yi]],class_thetas.shape[0],axis=0)
                
                a=time()
                dists = tf.norm(class_avgs_yi-class_thetas,axis=1)
                buff_upd += time()-a
                print("map eucld:"+str(buff_upd))

                max_dist_index_in_class = tf.math.argmax(dists).numpy()
                max_dist = dists[max_dist_index_in_class]
                max_dist_index = self._ind_per_class[yi][max_dist_index_in_class]

                d_nimg = tf.norm(theta_xi-class_avgs_yi)

                if d_nimg <= max_dist:
                    self._images = self._replace_in_tensor(
                        self._images, nimg, max_dist_index
                    )
                    self._thetas = self._replace_in_tensor(
                        self._thetas, theta_xi, max_dist_index
                    )
                    self._labels = self._replace_in_tensor(
                        self._labels, nlbl, max_dist_index
                    )
                    self._thetas_per_class[yi][max_dist_index_in_class] = theta_xi
                    

        #l = time()
        #print("buffer update time: {}".format(l - k))

    def _replace_in_tensor(self, tensor, element, index_):
        # print("Replacing item {} in tensor with length {}".format(index_, len(tensor)))

        if index_ == 0:
            return tf.concat(
                [
                    tf.convert_to_tensor([element]),
                    tf.gather(tensor, list(range(1, len(tensor))), axis=0),
                ],
                axis=0,
            )
        elif index_ == len(tensor) - 1:
            return tf.concat(
                [
                    tf.gather(tensor, list(range(len(tensor) - 1)), axis=0),
                    tf.convert_to_tensor([element]),
                ],
                axis=0,
            )
        else:
            return tf.concat(
                [
                    tf.gather(tensor, list(range(index_)), axis=0),
                    tf.convert_to_tensor([element]),
                    tf.gather(tensor, list(range(index_ + 1, len(tensor))), axis=0),
                ],
                axis=0,
            )


    def _compute_logits(self, images, batch_sz=256):
        result = None
        logits_fn = tf.function(model, experimental_relax_shapes=True )

        im_num = images.shape[0]

        num_splits = int(im_num/batch_sz)
        last_batch = im_num % batch_sz
        
        splits  = [batch_sz for _ in range(num_splits)]
        splits += [last_batch]

        splitted = tf.split(images, splits, axis=0)
        for im in splitted:
            logits = logits_fn(im)
            
            if result is None:
                result = logits
            else:
                result = tf.concat([result, logits], axis=0)
                
        return result
    


def train_step_initiate_graph_function():
    def train_step(model, images, labels, loss, opt):

        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            soft_logits = tf.nn.softmax(logits)
            loss_value = loss(labels, soft_logits) + sum(model.losses)

        grads = tape.gradient(loss_value, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

        return loss_value

    return train_step



#def distill_step_initiate_graph_function():
    
    #def batch_train_distill(models, images, labels, opt, loss, a=0.5, T=2):
        
        #old_preds          = models[0](images[0], training=False)
        #old_preds_soft_tmp = tf.nn.softmax(old_preds/T, axis=1)
        #LOSS               = sum(model[1].losses)
        
        #with tf.GradientTape() as tape:
                        
            #new_preds          = models[1](images[1], training=True)
            #new_preds_soft     = tf.nn.softmax(new_preds, axis=1)
            #new_preds_soft_tmp = tf.nn.softmax(new_preds/T, axis=1)
            
            #new_loss     = loss(labels[1], new_preds_soft)
            #distill_loss = loss(old_preds_soft_tmp, new_preds_soft_tmp)
            
            ##a=0.5 and T=8
            #LOSS += (1-a)*new_loss + a*distill_loss*(T**2)
                

        #grads = tape.gradient(LOSS, new_model.trainable_weights)
        #opt.apply_gradients(zip(grads, new_model.trainable_weights))
        #return LOSS

    #return train_step




def data_aug_img_layer(images, seed_):

    from tensorflow.keras.layers import Lambda

    def aug_img(img_batch, seed_):
        def random_flip_preproc(img_batch, seed_):
            img_batch = tf.image.random_flip_left_right(img_batch, seed=seed_)
            return img_batch

        def random_crop_preproc(img_batch, seed_):

            original_shape = ors = list(img_batch.shape)

            # Percentage of pixels to crop
            crp_p = random.sample([4, 8, 16], 1)[0]

            crop_size = [
                ors[0],
                ors[1] - int(ors[1] / crp_p),
                ors[2] - int(ors[2] / crp_p),
                ors[3],
            ]

            img_batch = tf.image.random_crop(img_batch, crop_size, seed=seed_)
            img_batch = tf.image.resize(
                img_batch, original_shape[1:3], method="nearest"
            )

            return img_batch

        choice = random.sample([0, 1, 2, 3, 4], 1)[0]

        if choice == 0:
            img_batch = random_flip_preproc(img_batch, seed_)
        elif choice == 1:
            img_batch = random_crop_preproc(img_batch, seed_)
        elif choice == 2:
            img_batch = random_flip_preproc(img_batch, seed_)
            img_batch = random_crop_preproc(img_batch, seed_)
        elif choice == 3:
            img_batch = random_crop_preproc(img_batch, seed_)
            img_batch = random_flip_preproc(img_batch, seed_)
        else:
            pass

        return img_batch

    # Keras Lambda layer
    return Lambda(lambda x: aug_img(x, seed_))(images)


def randomized_buffer_refresh(
    buffer_im, buffer_la, new_im, new_la, cut_size, buf_lim, seed_
):
    CURRENT_BUFFER_SIZE = len(buffer_im.numpy())

    # MAKE USER THE SAMPLE REQUESTED RESPECTS BATCH SIZE
    sample_len = min(len(new_im.numpy()), cut_size)

    # TAKE SAMPLE FROM NEW BATCH
    ind = random.sample(range(len(new_im.numpy())), sample_len)
    new_im = tf.gather(new_im, ind, axis=0)
    new_la = tf.gather(new_la, ind, axis=0)

    # IF BUFFER IS FULL REPLACE OLD ELEMENTS FROM BUFFER WITH NEW ELEMENTS
    # (SAMPLE) FROM BATCH
    if CURRENT_BUFFER_SIZE >= buf_lim:

        ind = random.sample(
            range(CURRENT_BUFFER_SIZE), CURRENT_BUFFER_SIZE - sample_len
        )
        buffer_im = tf.gather(buffer_im, ind, axis=0)
        buffer_la = tf.gather(buffer_la, ind, axis=0)

    # ADD NEW ELEMENTS
    buffer_im = tf.concat([buffer_im, new_im], axis=0)
    buffer_la = tf.concat([buffer_la, new_la], axis=0)

    # SHUFFLE BUFFER
    ind = random.sample(range(len(buffer_im.numpy())), len(buffer_im.numpy()))
    buffer_im = tf.gather(buffer_im, ind, axis=0)[:buf_lim]
    buffer_la = tf.gather(buffer_la, ind, axis=0)[:buf_lim]

    return buffer_im, buffer_la


def mix_batches(
    bufr_im,
    bufr_la,
    bufr_index_start,
    bufr_mix_batch,
    post_im,
    post_la,
    post_index_start,
    post_mix_batch,
    total_batches=None,
    seed=SEED,
):
    batch = bufr_mix_batch + post_mix_batch
    if batch <= 0:
        raise ValueError("Illegal batch size requested: {}".format(batch))

    bufr_len = len(bufr_im)
    post_len = len(post_im)
    if total_batches is None:
        total_batches = min(bufr_len // bufr_mix_batch, post_len // post_mix_batch)

    bufr_cur = bufr_index_start
    post_cur = post_index_start
    batches_cur = 0

    im_mix_batches = []
    la_mix_batches = []

    while batches_cur < total_batches:
        bufr_indxs = tf.constant(
            [v % bufr_len for v in range(bufr_cur, bufr_cur + bufr_mix_batch)]
        )
        bufr_cur = (bufr_cur + bufr_mix_batch) % bufr_len
        bufr_im_mix = tf.gather(bufr_im, bufr_indxs)
        bufr_la_mix = tf.gather(bufr_la, bufr_indxs)

        post_indxs = tf.constant(
            [v % post_len for v in range(post_cur, post_cur + post_mix_batch)]
        )
        post_cur = (post_cur + post_mix_batch) % post_len
        post_im_mix = tf.gather(post_im, post_indxs)
        post_la_mix = tf.gather(post_la, post_indxs)

        # print("Mixed batch postponed {},{} rehearsal".format(post_indxs, bufr_indxs))

        im_mix_batches += [tf.concat([bufr_im_mix, post_im_mix], axis=0)]
        la_mix_batches += [tf.concat([bufr_la_mix, post_la_mix], axis=0)]

        batches_cur += 1

    return im_mix_batches, la_mix_batches, bufr_cur, post_cur

#
#def dataset_training(model, dataset, epochs_, opt, loss, seed_, shuf=False):
#    train = tf.function(train_step_initiate_graph_function())
#    dataset = dataset.shuffle(60000, seed=seed_).cache()
#
#    for i in range(1, epochs_ + 1):
#        for j, (images, labels) in enumerate(dataset):
#            images = data_aug_img_layer(images, seed_ + i + j)
#            train(model, images, labels, loss, opt)
#            print("Epoch:" + str(i) + "|" + str(epochs_) + " btc:"+str(j), end="\r")
#        print()
#

@tf.function
def model_fn(model, x):
    return model(x,training=False)

def stream_testing(model, dts):
    
    accs    = []
    acc_obj = tf.keras.metrics.CategoricalAccuracy()
    
    for x,y in dts:
        preds = model_fn(model, x)
        acc_obj.update_state(y,preds)
        accs += [acc_obj.result().numpy()]
        acc_obj.reset_states()
    
    return [round(acc, 3) for acc in accs]
        
        
    


#def stream_testing(model, dts_list, verbose=0):
    #acc_list = []
    #with tf.device('/GPU:0'):
        #for dts in dts_list:
            #_, acc = model.evaluate(dts, verbose=verbose)
            #acc_list += [round(acc, 3)]
    #return acc_list


class OnlineTraining:
    def __init__(
        self,
        name,
        model,
        optimizer,
        loss,
        scoreboard,
        repeat,
        seed,
        augment_images=False,
    ):
        self._name = name
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._cur_batch = 0
        self._scoreboard = scoreboard
        self._repeat = repeat
        self._seed = seed
        self._augment_images = augment_images
        #self._online_accuracy = tf.keras.metrics.Accuracy()
        self._train = tf.function(train_step_initiate_graph_function())

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    def update(self, images, labels):
        print("\n####################:")
        print("[{}][{}]:".format(self._name, self._cur_batch))

        #self._online_accuracy.update_state(
            #argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1),
            #argmax(labels, axis=1),
        #)
        #self._scoreboard["online_acc_{}".format(self._name)] += [
            #self._online_accuracy.result().numpy()
        #]
        #print("Online accuracy: {}".format(self._online_accuracy.result().numpy()))
        
        preds  = argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1)
        lbls_  = argmax(labels, axis=1)
        correct_obsr=0
        img_count = len(images)
        for pr, lb in zip(preds,lbls_):
            if pr==lb:
                correct_obsr+=1
                
        online_acc = float(correct_obsr)/img_count
            
        self._scoreboard["online_acc_{}".format(self._name)] += [online_acc]
        print("Online accuracy: {}".format(online_acc))


        start = time()

        print("Training {} times".format(self._repeat))

        self._optimizer.iterations.assign(0)
        self._optimizer.learning_rate.decay_steps = 1

        train_batches = 0
        for i in range(self._repeat):
            train_images = images
            if self._augment_images: 
                train_images = data_aug_img_layer(images, self._seed + i + 1)
            self._train(self._model, train_images, labels, self._loss, self._optimizer)
            train_batches += 1
            sys.stdout.flush()

        self._cur_batch += 1
        stop = time()
        self._scoreboard["repeat_{}".format(self._name)] += [self._repeat]
        self._scoreboard["times_{}".format(self._name)] += [stop - start]
        self._scoreboard["train_batches_{}".format(self._name)] += [train_batches]


class ContinuousRehearsal:
    def __init__(
        self,
        name,
        model,
        optimizer,
        loss,
        rehearsal_buffer_images,
        rehearsal_buffer_labels,
        rehearsal_repeats,
        train_every_steps,
        scoreboard,
        seed,
        mix_len,
        augment_images=False,
    ):
        self._name = name
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._cur_batch = 0
        self._trained_on_last_batch = False
        self._postponed_images = None
        self._postponed_labels = None
        self._postponed_buffer_index = 0
        self._rehearsal_buffer = DynamicRehearsalBuffer(
            model, rehearsal_buffer_images, rehearsal_buffer_labels
        )
        self._rehearsal_buffer_index = 0
        self._rehearsal_repeats = rehearsal_repeats
        self._train_every_steps = train_every_steps
        self._scoreboard = scoreboard
        self._seed = seed
        self._mix_len = mix_len
        self._augment_images = augment_images
        #self._online_accuracy = tf.keras.metrics.Accuracy()
        self._train = tf.function(train_step_initiate_graph_function())

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    def update(self, images, labels):
        print("\n####################:")
        print("[{}][{}]:".format(self._name, self._cur_batch))

        #self._online_accuracy.update_state(
            #argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1),
            #argmax(labels, axis=1),
        #)
        #self._scoreboard["online_acc_{}".format(self._name)] += [
            #self._online_accuracy.result().numpy()
        #]
        #print("Online accuracy: {}".format(self._online_accuracy.result().numpy()))
        
        preds  = argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1)
        lbls_  = argmax(labels, axis=1)
        correct_obsr=0
        img_count = len(images)
        for pr, lb in zip(preds,lbls_):
            if pr==lb:
                correct_obsr+=1
                
        online_acc = float(correct_obsr)/img_count
            
        self._scoreboard["online_acc_{}".format(self._name)] += [online_acc]
        print("Online accuracy: {}".format(online_acc))


        start = time()

        if self._cur_batch == 0 or self._trained_on_last_batch:
            self._postponed_images = images
            self._postponed_labels = labels
        else:
            self._postponed_images = tf.concat([self._postponed_images, images], axis=0)
            self._postponed_labels = tf.concat([self._postponed_labels, labels], axis=0)

        train_batches = 0
        rehearsal_repeats = 0

        if self._cur_batch % self._train_every_steps == 0:
            rehearsal_repeats = self._rehearsal_repeats

            print("Training {} times".format(rehearsal_repeats))

            self._optimizer.iterations.assign(0)
            batches_per_rehearsal_repeat = len(self._postponed_images) // self._mix_len
            self._optimizer.learning_rate.decay_steps = batches_per_rehearsal_repeat

            for _ in range(self._rehearsal_repeats):
                (
                    mix_im_list,
                    mix_la_list,
                    self._rehearsal_buffer_index,
                    self._postponed_buffer_index,
                ) = mix_batches(
                    self._rehearsal_buffer.images,
                    self._rehearsal_buffer.labels,
                    self._rehearsal_buffer_index,
                    self._mix_len,
                    self._postponed_images,
                    self._postponed_labels,
                    self._postponed_buffer_index,
                    self._mix_len,
                    total_batches=batches_per_rehearsal_repeat,
                    seed=self._seed,
                )

                for mix_im, mix_la in zip(mix_im_list, mix_la_list):
                    train_images = mix_im
                    if self._augment_images: 
                        train_images = data_aug_img_layer(mix_im, self._seed)
                    self._train(
                        self._model, train_images, mix_la, self._loss, self._optimizer
                    )
                    train_batches += 1
                    sys.stdout.flush()
            self._trained_on_last_batch = True
        else:
            self._trained_on_last_batch = False

        self._rehearsal_buffer.update(self._model, images, labels)

        self._cur_batch += 1
        stop = time()
        self._scoreboard["repeat_{}".format(self._name)] += [rehearsal_repeats]
        self._scoreboard["times_{}".format(self._name)] += [stop - start]
        self._scoreboard["train_batches_{}".format(self._name)] += [train_batches]


class ContinuousRehearsalConverge:
    def __init__(
        self,
        name,
        model,
        optimizer,
        loss,
        rehearsal_buffer_images,
        rehearsal_buffer_labels,
        train_every_steps,
        scoreboard,
        seed,
        mix_len,
        augment_images=False,
        alpha_short=0.5,
        alpha_long=0.05,
        eps=0.005,
    ):
        self._name = name
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._cur_batch = 0
        self._trained_on_last_batch = False
        self._postponed_images = None
        self._postponed_labels = None
        self._postponed_buffer_index = 0
        self._rehearsal_buffer = DynamicRehearsalBuffer(
            model, rehearsal_buffer_images, rehearsal_buffer_labels
        )
        self._rehearsal_buffer_index = 0
        self._train_every_steps = train_every_steps
        self._scoreboard = scoreboard
        self._seed = seed
        self._mix_len = mix_len
        self._augment_images = augment_images
        self._loss_alpha_short = alpha_short
        self._loss_alpha_long = alpha_long
        self._eps = eps
        # TODO: Fixed value for now. Explore option of "warmup" for 5
        # iterations and then monitor difference.
        self._running_avg_loss_short = 1.0
        self._running_avg_loss_long = 1.0
        #self._online_accuracy = tf.keras.metrics.Accuracy()
        self._train = tf.function(train_step_initiate_graph_function())

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    def update(self, images, labels):
        print("\n####################:")
        print("[{}][{}]:".format(self._name, self._cur_batch))

        #self._online_accuracy.update_state(
            #argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1),
            #argmax(labels, axis=1),
        #)
        #self._scoreboard["online_acc_{}".format(self._name)] += [
            #self._online_accuracy.result().numpy()
        #]
        #print("Online accuracy: {}".format(self._online_accuracy.result().numpy()))
        
        preds  = argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1)
        lbls_  = argmax(labels, axis=1)
        correct_obsr=0
        img_count = len(images)
        for pr, lb in zip(preds,lbls_):
            if pr==lb:
                correct_obsr+=1
                
        online_acc = float(correct_obsr)/img_count
            
        self._scoreboard["online_acc_{}".format(self._name)] += [online_acc]
        print("Online accuracy: {}".format(online_acc))


        start = time()

        if self._cur_batch == 0 or self._trained_on_last_batch:
            self._postponed_images = images
            self._postponed_labels = labels
        else:
            self._postponed_images = tf.concat([self._postponed_images, images], axis=0)
            self._postponed_labels = tf.concat([self._postponed_labels, labels], axis=0)

        train_batches = 0
        rehearsal_repeats = 0
        if self._cur_batch % self._train_every_steps == 0:
            print("Starting training until convergence with eps={}".format(self._eps))

            stop_training = False
            self._optimizer.iterations.assign(0)
            batches_per_rehearsal_repeat = len(self._postponed_images) // self._mix_len
            self._optimizer.learning_rate.decay_steps = batches_per_rehearsal_repeat
            while not stop_training:
                (
                    mix_im_list,
                    mix_la_list,
                    self._rehearsal_buffer_index,
                    self._postponed_buffer_index,
                ) = mix_batches(
                    self._rehearsal_buffer.images,
                    self._rehearsal_buffer.labels,
                    self._rehearsal_buffer_index,
                    self._mix_len,
                    self._postponed_images,
                    self._postponed_labels,
                    self._postponed_buffer_index,
                    self._mix_len,
                    seed=self._seed,
                )

                # Avoid huge variable names
                l = self._running_avg_loss_short
                ll = self._running_avg_loss_long
                a = self._loss_alpha_short
                al = self._loss_alpha_long
                for mix_im, mix_la in zip(mix_im_list, mix_la_list):
                    train_images = mix_im
                    if self._augment_images: 
                        train_images = data_aug_img_layer(mix_im, self._seed)
                    loss = self._train(
                        self._model, train_images, mix_la, self._loss, self._optimizer
                    )
                    sys.stdout.flush()
                    train_batches += 1
                    l = (1 - a) * l + a * loss
                    ll = (1 - al) * ll + al * loss
                    # print('Loss: {}, '
                    #       'Running Avg (short): {}, '
                    #       'Running Avg (long): {}, '
                    #       'Abs. Diff: {}'.
                    #       format(loss, l, ll, abs(ll - l)))
                    if abs(ll - l) < self._eps:
                        stop_training = True
                rehearsal_repeats += 1
                self._running_avg_loss_short = l
                self._running_avg_loss_long = ll
                self._trained_on_last_batch = True
                if stop_training:
                    print(
                        "Stopping training after {} batches and {} repeats".format(
                            train_batches, rehearsal_repeats
                        )
                    )
                    sys.stdout.flush()

        else:
            self._trained_on_last_batch = False

        self._rehearsal_buffer.update(self._model, images, labels)
        self._cur_batch += 1
        stop = time()
        self._scoreboard["repeat_{}".format(self._name)] += [rehearsal_repeats]
        self._scoreboard["times_{}".format(self._name)] += [stop - start]
        self._scoreboard["train_batches_{}".format(self._name)] += [train_batches]


class DriftActivatedRehearsal:
    def __init__(
        self,
        name,
        model,
        optimizer,
        loss,
        rehearsal_buffer_images,
        rehearsal_buffer_labels,
        rehearsal_repeats,
        scoreboard,
        seed,
        mix_len,
        err_thr,
        max_notrain,
        use_rehearsal_drift_detector=False,
        rehearsal_drift_detector_batch=10,
        augment_images=False,
        dynamic_initial_learning_rate=True,
        dynamic_rehearsal_repeats=True,
    ):
        self._name = name
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._cur_batch = 0
        self._drift_detector = ECDDetector(avg_run_len=100)
        self._postponed_images = None
        self._postponed_labels = None
        self._postponed_buffer_index = 0
        self._rehearsal_buffer = DynamicRehearsalBuffer(
            model, rehearsal_buffer_images, rehearsal_buffer_labels
        )
        self._rehearsal_buffer_index = 0
        self._rehearsal_repeats = rehearsal_repeats
        self._scoreboard = scoreboard
        self._seed = seed
        self._mix_len = mix_len
        self._err_thr = err_thr
        self._max_notrain = max_notrain
        self._last_train = 0
        self._rehearsal_drift_detector = (
            ECDDetector(avg_run_len=100) if use_rehearsal_drift_detector else None
        )
        self._rehearsal_drift_detector_batch = rehearsal_drift_detector_batch
        self._augment_images = augment_images

        # Store initial learning rate from optimizer
        if callable(self._optimizer.learning_rate):
            self._initial_learning_rate = self._optimizer.learning_rate(0).numpy()
        else:
            self._initial_learning_rate = self._optimizer.learning_rate.numpy()

        self._dynamic_initial_learning_rate = dynamic_initial_learning_rate
        self._dynamic_rehearsal_repeats = dynamic_rehearsal_repeats

        #self._online_accuracy = tf.keras.metrics.Accuracy()
        self._train = tf.function(train_step_initiate_graph_function())

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    def _compute_training_repeat(self, detected_drift, Z_t):
        if self._dynamic_rehearsal_repeats:
            return ceil(2 * self._rehearsal_repeats * log2(1 + Z_t))
        else:
            return self._rehearsal_repeats

    def _compute_initial_learning_rate(self, detected_drift, Z_t):
        if self._dynamic_initial_learning_rate:
            return self._initial_learning_rate * min(100, 5 * np.exp(3 * Z_t))
        else:
            return self._initial_learning_rate

    def update(self, images, labels):
        print("\n####################:")
        print("[{}][{}]:".format(self._name, self._cur_batch))

        #self._online_accuracy.update_state(
            #argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1),
            #argmax(labels, axis=1),
        #)
        #self._scoreboard["online_acc_{}".format(self._name)] += [
            #self._online_accuracy.result().numpy()
        #]
        #print("Online accuracy: {}".format(self._online_accuracy.result().numpy()))
        
        preds  = argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1)
        lbls_  = argmax(labels, axis=1)
        correct_obsr=0
        img_count = len(images)
        for pr, lb in zip(preds,lbls_):
            if pr==lb:
                correct_obsr+=1
                
        online_acc = float(correct_obsr)/img_count
            
        self._scoreboard["online_acc_{}".format(self._name)] += [online_acc]
        print("Online accuracy: {}".format(online_acc))


        start = time()

        if self._cur_batch == 0 or self._cur_batch == self._last_train + 1:
            self._postponed_images = images
            self._postponed_labels = labels
        else:
            self._postponed_images = tf.concat([self._postponed_images, images], axis=0)
            self._postponed_labels = tf.concat([self._postponed_labels, labels], axis=0)

        detected_drift = self._drift_detector.predict(self._model, images, labels)
        actual_Z_t = self._drift_detector.Z_t
        actual_p_0t = self._drift_detector.p_0t
        actual_diff_X_t = self._drift_detector.diff_X_t
        actual_X_t = self._drift_detector.X_t

        if self._rehearsal_drift_detector is not None:
            rehearsal_sample_idxs = random.sample(
                range(tf.shape(self._rehearsal_buffer.images)[0]),
                self._rehearsal_drift_detector_batch,
            )
            rehearsal_sample_idxs = tf.constant(rehearsal_sample_idxs)
            rehearsal_detected_drift = self._rehearsal_drift_detector.predict(
                self._model,
                tf.gather(self._rehearsal_buffer.images, rehearsal_sample_idxs),
                tf.gather(self._rehearsal_buffer.labels, rehearsal_sample_idxs),
            )
            if rehearsal_detected_drift:
                detected_drift = True
                actual_Z_t = max(actual_Z_t, self._rehearsal_drift_detector.Z_t)
                actual_p_0t = max(actual_p_0t, self._rehearsal_drift_detector.p_0t)
                actual_diff_X_t = min(
                    actual_diff_X_t, self._rehearsal_drift_detector.diff_X_t
                )
                actual_X_t = max(actual_X_t, self._rehearsal_drift_detector.X_t)

        print("detected_drift = {}".format(detected_drift))
        print("X_t = {}".format(actual_X_t))
        print("diff_X_t = {}".format(actual_diff_X_t))
        print("Z_t = {}".format(actual_Z_t))
        print("p_0t = {}".format(actual_p_0t))
        sys.stdout.flush()
        self._scoreboard["detected_drift_{}".format(self._name)] += [detected_drift]
        self._scoreboard["diff_X_t_{}".format(self._name)] += [actual_diff_X_t]
        self._scoreboard["Z_t_{}".format(self._name)] += [actual_Z_t]
        self._scoreboard["p_0t_{}".format(self._name)] += [actual_p_0t]

        train_batches = 0
        rehearsal_repeat = 0

        # Conditions to start training
        if (
            detected_drift
            or actual_Z_t > self._err_thr
            or (self._cur_batch - self._last_train) > self._max_notrain
            # or actual_diff_X_t < -self._err_thr
        ):
            rehearsal_repeat = self._compute_training_repeat(detected_drift, actual_Z_t)
            initial_learning_rate = self._compute_initial_learning_rate(
                detected_drift, actual_Z_t
            )

            batches_per_rehearsal_repeat = len(self._postponed_images) // self._mix_len
            self._optimizer.iterations.assign(0)
            self._optimizer.learning_rate.initial_learning_rate = initial_learning_rate
            self._optimizer.learning_rate.decay_steps = batches_per_rehearsal_repeat

            print("Training {} times".format(rehearsal_repeat))
            print(
                "Using {} as initial learning rate".format(
                    self._optimizer.learning_rate.initial_learning_rate
                )
            )

            for _ in range(rehearsal_repeat):
                (
                    mix_im_list,
                    mix_la_list,
                    self._rehearsal_buffer_index,
                    self._postponed_buffer_index,
                ) = mix_batches(
                    self._rehearsal_buffer.images,
                    self._rehearsal_buffer.labels,
                    self._rehearsal_buffer_index,
                    self._mix_len,
                    self._postponed_images,
                    self._postponed_labels,
                    self._postponed_buffer_index,
                    self._mix_len,
                    total_batches=batches_per_rehearsal_repeat,
                    seed=self._seed,
                )

                for mix_im, mix_la in zip(mix_im_list, mix_la_list):
                    train_images = mix_im
                    if self._augment_images: 
                        train_images = data_aug_img_layer(mix_im, self._seed)
                    self._train(
                        self._model, train_images, mix_la, self._loss, self._optimizer
                    )
                    train_batches += 1
                    sys.stdout.flush()
            self._last_train = self._cur_batch

        self._rehearsal_buffer.update(self._model, images, labels)
        self._cur_batch += 1
        stop = time()
        self._scoreboard["repeat_{}".format(self._name)] += [rehearsal_repeat]
        self._scoreboard["times_{}".format(self._name)] += [stop - start]
        self._scoreboard["train_batches_{}".format(self._name)] += [train_batches]


class DriftActivatedRehearsalConverge:
    def __init__(
        self,
        name,
        model,
        optimizer,
        loss,
        rehearsal_buffer_images,
        rehearsal_buffer_labels,
        scoreboard,
        seed,
        mix_len,
        err_thr,
        max_notrain,
        use_rehearsal_drift_detector=False,
        rehearsal_drift_detector_batch=10,
        augment_images=False,
        dynamic_initial_learning_rate=True,
        alpha_short=0.5,
        alpha_long=0.05,
        eps=0.005,
    ):
        self._name = name
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._cur_batch = 0
        self._drift_detector = ECDDetector(avg_run_len=100)
        self._postponed_images = None
        self._postponed_labels = None
        self._postponed_buffer_index = 0
        self._rehearsal_buffer = DynamicRehearsalBuffer(
            model, rehearsal_buffer_images, rehearsal_buffer_labels
        )
        self._rehearsal_buffer_index = 0
        self._scoreboard = scoreboard
        self._seed = seed
        self._mix_len = mix_len
        self._err_thr = err_thr
        self._max_notrain = max_notrain
        self._last_train = 0
        self._rehearsal_drift_detector = (
            ECDDetector(avg_run_len=100) if use_rehearsal_drift_detector else None
        )
        self._rehearsal_drift_detector_batch = rehearsal_drift_detector_batch
        self._augment_images = augment_images
        self._loss_alpha_short = alpha_short
        self._loss_alpha_long = alpha_long
        self._eps = eps
        # Fixed
        self._running_avg_loss_short = 1.0
        self._running_avg_loss_long = 1.0

        # Store initial learning rate from optimizer
        if callable(self._optimizer.learning_rate):
            self._initial_learning_rate = self._optimizer.learning_rate(0).numpy()
        else:
            self._initial_learning_rate = self._optimizer.learning_rate.numpy()

        self._dynamic_initial_learning_rate = dynamic_initial_learning_rate
        #self._online_accuracy = tf.keras.metrics.Accuracy()
        self._train = tf.function(train_step_initiate_graph_function())

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    def _compute_initial_learning_rate(self, detected_drift, Z_t):
        if self._dynamic_initial_learning_rate:
            return self._initial_learning_rate * min(100, 5 * np.exp(3 * Z_t))
        else:
            return self._initial_learning_rate

    def update(self, images, labels):
        print("\n####################:")
        print("[{}][{}]:".format(self._name, self._cur_batch))

        #self._online_accuracy.update_state(
            #argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1),
            #argmax(labels, axis=1),
        #)
        #self._scoreboard["online_acc_{}".format(self._name)] += [
            #self._online_accuracy.result().numpy()
        #]
        #print("Online accuracy: {}".format(self._online_accuracy.result().numpy()))
        
        preds  = argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1)
        lbls_  = argmax(labels, axis=1)
        correct_obsr=0
        img_count = len(images)
        for pr, lb in zip(preds,lbls_):
            if pr==lb:
                correct_obsr+=1
                
        online_acc = float(correct_obsr)/img_count
            
        self._scoreboard["online_acc_{}".format(self._name)] += [online_acc]
        print("Online accuracy: {}".format(online_acc))


        start = time()

        if self._cur_batch == 0 or self._cur_batch == self._last_train + 1:
            self._postponed_images = images
            self._postponed_labels = labels
        else:
            self._postponed_images = tf.concat([self._postponed_images, images], axis=0)
            self._postponed_labels = tf.concat([self._postponed_labels, labels], axis=0)

        detected_drift = self._drift_detector.predict(self._model, images, labels)
        actual_Z_t = self._drift_detector.Z_t
        actual_p_0t = self._drift_detector.p_0t
        actual_diff_X_t = self._drift_detector.diff_X_t
        actual_X_t = self._drift_detector.X_t

        if self._rehearsal_drift_detector is not None:
            rehearsal_sample_idxs = random.sample(
                range(tf.shape(self._rehearsal_buffer.images)[0]),
                self._rehearsal_drift_detector_batch,
            )
            rehearsal_sample_idxs = tf.constant(rehearsal_sample_idxs)
            rehearsal_detected_drift = self._rehearsal_drift_detector.predict(
                self._model,
                tf.gather(self._rehearsal_buffer.images, rehearsal_sample_idxs),
                tf.gather(self._rehearsal_buffer.labels, rehearsal_sample_idxs),
            )
            if rehearsal_detected_drift:
                detected_drift = True
                actual_Z_t = max(actual_Z_t, self._rehearsal_drift_detector.Z_t)
                actual_p_0t = max(actual_p_0t, self._rehearsal_drift_detector.p_0t)
                actual_diff_X_t = min(
                    actual_diff_X_t, self._rehearsal_drift_detector.diff_X_t
                )
                actual_X_t = max(actual_X_t, self._rehearsal_drift_detector.X_t)

        print("detected_drift = {}".format(detected_drift))
        print("X_t = {}".format(actual_X_t))
        print("diff_X_t = {}".format(actual_diff_X_t))
        print("Z_t = {}".format(actual_Z_t))
        print("p_0t = {}".format(actual_p_0t))
        sys.stdout.flush()
        self._scoreboard["detected_drift_{}".format(self._name)] += [detected_drift]
        self._scoreboard["diff_X_t_{}".format(self._name)] += [actual_diff_X_t]
        self._scoreboard["Z_t_{}".format(self._name)] += [actual_Z_t]
        self._scoreboard["p_0t_{}".format(self._name)] += [actual_p_0t]

        train_batches = 0
        rehearsal_repeats = 0
        if (
            detected_drift
            or actual_Z_t > self._err_thr
            or (self._cur_batch - self._last_train) > self._max_notrain
            #or actual_diff_X_t < -self._err_thr
        ):
            print("Starting training until convergence with eps={}".format(self._eps))
            stop_training = False
            initial_learning_rate = self._compute_initial_learning_rate(
                detected_drift, actual_Z_t
            )
            batches_per_rehearsal_repeat = len(self._postponed_images) // self._mix_len
            self._optimizer.iterations.assign(0)
            self._optimizer.learning_rate.initial_learning_rate = initial_learning_rate
            self._optimizer.learning_rate.decay_steps = batches_per_rehearsal_repeat

            while not stop_training:
                (
                    mix_im_list,
                    mix_la_list,
                    self._rehearsal_buffer_index,
                    self._postponed_buffer_index,
                ) = mix_batches(
                    self._rehearsal_buffer.images,
                    self._rehearsal_buffer.labels,
                    self._rehearsal_buffer_index,
                    self._mix_len,
                    self._postponed_images,
                    self._postponed_labels,
                    self._postponed_buffer_index,
                    self._mix_len,
                    total_batches=batches_per_rehearsal_repeat,
                    seed=self._seed,
                )

                # Avoid huge variable names
                l = self._running_avg_loss_short
                ll = self._running_avg_loss_long
                a = self._loss_alpha_short
                al = self._loss_alpha_long
                for mix_im, mix_la in zip(mix_im_list, mix_la_list):
                    train_images = mix_im
                    if self._augment_images: 
                        train_images = data_aug_img_layer(mix_im, self._seed)
                    loss = self._train(
                        self._model, train_images, mix_la, self._loss, self._optimizer
                    )
                    sys.stdout.flush()
                    train_batches += 1
                    l = (1 - a) * l + a * loss
                    ll = (1 - al) * ll + al * loss
                    # print('Loss: {}, '
                    #       'Running Avg (short): {}, '
                    #       'Running Avg (long): {}, '
                    #       'Abs. Diff: {}'.
                    #       format(loss, l, ll, abs(ll - l)))
                    if abs(ll - l) < self._eps:
                        stop_training = True

                rehearsal_repeats += 1
                self._running_avg_loss_short = l
                self._running_avg_loss_long = ll
                self._last_train = self._cur_batch
                if stop_training:
                    print(
                        "Stopping training after {} batches and {} repeats".format(
                            train_batches, rehearsal_repeats
                        )
                    )
                    sys.stdout.flush()

        self._rehearsal_buffer.update(self._model, images, labels)
        self._cur_batch += 1
        stop = time()

        # rehearsal_repeats will always be zero
        self._scoreboard["repeat_{}".format(self._name)] += [rehearsal_repeats]
        self._scoreboard["times_{}".format(self._name)] += [stop - start]
        self._scoreboard["train_batches_{}".format(self._name)] += [train_batches]


def create_tasks(
    stream_size,
    task_size,
    task_classes,
    random_task_selection=False,
    random_task_sizes=False,
    random_task_sizes_sigma=3,
    dataset_name="cifar10",
    train_batch_size=BATCH,
    test_batch_size=TEST_BATCH,
    seed=SEED,
    shuffle_buffer_size=60000,
    perm_set = {}):
    
    avail_tasks = {}
    test_stream = []

    for i, task in enumerate(task_classes):
        task_perm={}
        if perm_set !={}:
            task_perm = {i:perm_set[i]}
        train_task, test_task, _, _ = custom_dataset(
            dataset_name, task, task, -1, -1, seed, True, 60000, task_perm
        )

        avail_tasks[i] = train_task
        
        if i==0:
            test_stream = test_task
        else:
            test_stream = test_stream.concatenate(test_task)
            
    test_stream = test_stream.cache()
    test_stream = test_stream.batch(test_batch_size)
    test_stream = test_stream.apply(\
        tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=2))

    number_of_tasks = len(avail_tasks)
    tasks_sizes = []
    tasks = []

    remaining_size = stream_size

    i = 0
    while remaining_size > 0:
        if random_task_selection:
            current_task = random.randrange(number_of_tasks)
        else:
            current_task = i % number_of_tasks

        if remaining_size < task_size:
            current_task_size = remaining_size
        else:
            if random_task_sizes:
                current_task_size = int(
                    random.gauss(task_size, sigma=random_task_sizes_sigma)
                )
            else:
                current_task_size = task_size

        if current_task_size <= 0:
            continue

        chosen_task = avail_tasks[current_task]
        tasks_sizes += [current_task_size]
        tasks += [current_task]

        chosen_task_stream = chosen_task.shuffle(shuffle_buffer_size, seed=seed).take(
            current_task_size
        )
        if stream_size == remaining_size:
            train_stream = chosen_task_stream
        else:
            train_stream = train_stream.concatenate(chosen_task_stream)

        remaining_size -= current_task_size
        i += 1

    actual_task_per_step = []
    for cur_size, cur_task in zip(tasks_sizes, tasks):
        cur_steps = cur_size // train_batch_size
        actual_task_per_step.extend([cur_task] * cur_steps)
        total_number_of_steps = len(actual_task_per_step)

    return (
        number_of_tasks,
        tasks,
        tasks_sizes,
        total_number_of_steps,
        actual_task_per_step,
        train_stream,
        test_stream,
    )


def clone_initial_model(initial_model, optimizer, loss=LOSS):
    model = tf.keras.models.clone_model(initial_model)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.set_weights(initial_model.get_weights())
    return model


def optimizer_factory(
    optimizer,
    initial_learning_rate,
    decay_rate,
    decay_steps=LEARNING_RATE_DECAY_STEPS,
):
    learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
    )

    if optimizer == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate_fn)
    elif optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    raise ValueError("Not supported")


#
# Main program start
#
print("Starting")


physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) != 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Number of available GPUs: ", len(physical_devices))

print_all_parameters()

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
rng = np.random.RandomState(SEED)

perm_set = {}

if DATASET_NAME == "psmnist":
    print("\nCreate permutations")
    for task_n in range(10):
        perm = rng.permutation(784)
        if task_n==0:
            perm = np.arange(784)

        perm_set[task_n] = perm
        print(task_n, end=":")
        print(hash(str(perm)))
    print()


print("Creating rehearsal buffer")
buffer_dts, _, n_classes, img_sh = custom_dataset(
    DATASET_NAME, ALL_CLASSES, [], BUFFER_NUM_PER_CLASS, -1, SEED, True, 60000, perm_set
)
buffer_size = BUFFER_NUM_PER_CLASS * len(n_classes)

for j, (im, la) in enumerate(buffer_dts.batch(buffer_size)):
    if j == 0:
        buffer_im = im
        buffer_la = la
    else:
        buffer_im = tf.concat([buffer_im, im], axis=0)
        buffer_la = tf.concat([buffer_la, la], axis=0)


print("Creating image sequence/tasks")
(
    number_of_tasks,
    tasks,
    tasks_sizes,
    total_number_of_steps,
    actual_task_per_step,
    train_stream,
    test_stream,
) = create_tasks(
    SIZE_OF_STREAM,
    SIZE_OF_TASK,
    TASK_CLASSES,
    dataset_name=DATASET_NAME,
    random_task_selection=USE_RANDOM_TASK_SELECTION,
    random_task_sizes=USE_RANDOM_TASK_SIZE,
    random_task_sizes_sigma=RANDOM_TASK_SIZE_SIGMA,
    perm_set=perm_set
)


print("Total tasks: {}".format(number_of_tasks))
print("Sequence of tasks: {}".format(tasks))
print("Each task size in images: {}".format(tasks_sizes))
print("Total number of steps: {}".format(total_number_of_steps))


print("INIT & PRETRAIN MODELS")

if not tf.io.gfile.exists(NAME_OF_INIT_MODEL):
    print("A new model has been created, offline training will commence...")
    print("Creating pretraining dataset")
#    pretrain_dts, _, pretrain_n_classes, pretrain_img_shape = custom_dataset(
#        DATASET_NAME, ALL_CLASSES, [], PRETRAIN_NUM_PER_CLASS, -1, SEED, True, 60000, perm_set
#    )
    if MODEL_NAME == "resnet32":
        init_model = ResNet32((32,32,3), 10)
    if MODEL_NAME == "resnet18":
        init_model = ResNet18((32,32,3), 10)
    
    init_model.compile(loss=LOSS, metrics=["accuracy"])
    init_model_opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    with tf.device("/GPU:0"):
#        dataset_training(
#            init_model,
#            pretrain_dts.batch(BATCH),
#            PRE_EPOCHS,
#            opt=init_model_opt,
#            loss=LOSS,
#            seed_=SEED,
#            shuf=True,
#        )
        init_model.save(NAME_OF_INIT_MODEL, overwrite=True)
else:
    print("Loading model from: {}".format(NAME_OF_INIT_MODEL))
    init_model = tf.keras.models.load_model(NAME_OF_INIT_MODEL)

print("Initializing score board")
scoreboard = defaultdict(list)
scoreboard["task"] = actual_task_per_step


# Create methods
print("Creating separate model per method")
METHODS = []


# drift activated
for learning_rate in LEARNING_RATES:
    for optimizer_name in OPTIMIZERS:


        # plain drift activated
        for max_rh_repeat in DRIFTA_MAX_RH_REPEAT:
            optimizer = optimizer_factory(
                optimizer_name, learning_rate, decay_rate=LEARNING_RATE_DECAY_RATE
            )
            model = clone_initial_model(init_model, optimizer)
            METHODS += [
                DriftActivatedRehearsal(
                    name="drifta{}_{}_{}".format(max_rh_repeat,optimizer_name, learning_rate),
                    model=model,
                    optimizer=optimizer,
                    loss=LOSS,
                    rehearsal_buffer_images=buffer_im,
                    rehearsal_buffer_labels=buffer_la,
                    rehearsal_repeats=max_rh_repeat,
                    scoreboard=scoreboard,
                    seed=SEED,
                    mix_len=MIX_LEN,
                    err_thr=ERROR_THR,
                    max_notrain=MAX_NOTRAIN,
                    use_rehearsal_drift_detector=False,
                    augment_images=False,
                    dynamic_initial_learning_rate=True,
                    dynamic_rehearsal_repeats=True,
                )
            ]

        # Drift activated rehearsal until convergence
        optimizer = optimizer_factory(
            optimizer_name, learning_rate, decay_rate=LEARNING_RATE_DECAY_RATE
        )
        model = clone_initial_model(init_model, optimizer)
        METHODS += [
            DriftActivatedRehearsalConverge(
                name="drifta_conv_{}_{}".format(optimizer_name, learning_rate),
                model=model,
                optimizer=optimizer,
                loss=LOSS,
                rehearsal_buffer_images=buffer_im,
                rehearsal_buffer_labels=buffer_la,
                scoreboard=scoreboard,
                seed=SEED,
                mix_len=MIX_LEN,
                err_thr=ERROR_THR,
                max_notrain=MAX_NOTRAIN,
                use_rehearsal_drift_detector=False,
                dynamic_initial_learning_rate=True,
                rehearsal_drift_detector_batch=10,
                augment_images=False,
                alpha_short=0.5,
                alpha_long=0.05,
                eps=0.005,
            )
        ]


# two drift detectors
for learning_rate in LEARNING_RATES:
    for optimizer_name in OPTIMIZERS:

        # 2drifta
        for max_rh_repeat in DRIFTA_MAX_RH_REPEAT:
            optimizer = optimizer_factory(
                optimizer_name, learning_rate, decay_rate=LEARNING_RATE_DECAY_RATE
            )
            model = clone_initial_model(init_model, optimizer)
            METHODS += [
                DriftActivatedRehearsal(
                    name="2drifta{}_{}_{}".format(max_rh_repeat, optimizer_name, learning_rate),
                    model=model,
                    optimizer=optimizer,
                    loss=LOSS,
                    rehearsal_buffer_images=buffer_im,
                    rehearsal_buffer_labels=buffer_la,
                    rehearsal_repeats=max_rh_repeat,
                    scoreboard=scoreboard,
                    seed=SEED,
                    mix_len=MIX_LEN,
                    err_thr=ERROR_THR,
                    max_notrain=MAX_NOTRAIN,
                    use_rehearsal_drift_detector=True,
                    augment_images=False,
                    dynamic_initial_learning_rate=True,
                    dynamic_rehearsal_repeats=True,
                )
            ]


        # 2drifta converge
        optimizer = optimizer_factory(
            optimizer_name, learning_rate, decay_rate=LEARNING_RATE_DECAY_RATE
        )
        model = clone_initial_model(init_model, optimizer)
        METHODS += [
            DriftActivatedRehearsalConverge(
                name="2drifta_conv_{}_{}".format(optimizer_name, learning_rate),
                model=model,
                optimizer=optimizer,
                loss=LOSS,
                rehearsal_buffer_images=buffer_im,
                rehearsal_buffer_labels=buffer_la,
                scoreboard=scoreboard,
                seed=SEED,
                mix_len=MIX_LEN,
                err_thr=ERROR_THR,
                max_notrain=MAX_NOTRAIN,
                use_rehearsal_drift_detector=True,
                dynamic_initial_learning_rate=True,
                rehearsal_drift_detector_batch=10,
                augment_images=False,
                alpha_short=0.5,
                alpha_long=0.05,
                eps=0.005,
            )
        ]



for learning_rate in LEARNING_RATES:
    for optimizer_name in OPTIMIZERS:

        # Continuous rehearsal until convergence
        optimizer = optimizer_factory(
            optimizer_name, learning_rate, decay_rate=LEARNING_RATE_DECAY_RATE
        )
        model = clone_initial_model(init_model, optimizer)
        METHODS += [
            ContinuousRehearsalConverge(
                name="cont_conv_{}_{}".format(optimizer_name, learning_rate),
                model=model,
                optimizer=optimizer,
                loss=LOSS,
                rehearsal_buffer_images=buffer_im,
                rehearsal_buffer_labels=buffer_la,
                train_every_steps=1,
                scoreboard=scoreboard,
                seed=SEED,
                mix_len=MIX_LEN,
                augment_images=False,
                alpha_short=0.5,
                alpha_long=0.05,
                eps=0.005,
            )
        ]


#
# Online Training
#
for learning_rate in LEARNING_RATES:
    for optimizer_name in OPTIMIZERS:
        optimizer = optimizer_factory(
            optimizer_name, learning_rate, decay_rate=LEARNING_RATE_DECAY_RATE
        )
        model = clone_initial_model(init_model, optimizer)
        METHODS += [
            OnlineTraining(
                name="catf_{}_{}".format(optimizer_name, learning_rate),
                model=model,
                optimizer=optimizer,
                loss=LOSS,
                scoreboard=scoreboard,
                repeat=1,
                seed=SEED,
                augment_images=False,
            )
        ]


#
# continuous rehearsal
#
for learning_rate in LEARNING_RATES:
    for optimizer_name in OPTIMIZERS:
        for conr_repeat in CONR_N_RH_REPEAT:
            optimizer = optimizer_factory(
                optimizer_name, learning_rate, decay_rate=LEARNING_RATE_DECAY_RATE
            )
            model = clone_initial_model(init_model, optimizer)
            name = "conr{}_{}_{}".format(conr_repeat, optimizer_name, learning_rate)
            METHODS += [
                ContinuousRehearsal(
                    name=name,
                    model=model,
                    optimizer=optimizer,
                    loss=LOSS,
                    rehearsal_buffer_images=buffer_im,
                    rehearsal_buffer_labels=buffer_la,
                    rehearsal_repeats=conr_repeat,
                    train_every_steps=1,
                    scoreboard=scoreboard,
                    seed=SEED,
                    mix_len=MIX_LEN,
                    augment_images=False,
                )
            ]


print("Computing pretrained model accuracies on all tasks")
pretrained_acc = stream_testing(init_model, test_stream)
print("Pretrained model accuracy on all tasks: {}".format(pretrained_acc))

for t in range(number_of_tasks):
    scoreboard["acc_pretrain_T{}".format(t)] = [
        pretrained_acc[t]
    ] * total_number_of_steps


print("TRAIN & TEST")

# Save model every 100 batches
save_step = 100
full_run=0
with tf.device("/GPU:0"):


    ################ TRAIN STREAM ####################
    stream_ = train_stream.cache()
    stream_ = stream_.batch(BATCH)
    stream_ = stream_\
        .apply(tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=150))
    ##################################################

    for cur_batch, (images, labels) in enumerate(stream_):

        # run all methods
        for method in METHODS:
            TT = time()
            method.update(images, labels)
            ET = time()
            full_run += (ET-TT)/3600
            print( (round(ET-TT,3), round(full_run,3)) )
            
        # test all methods
        for method in METHODS:
            TT = time()
            accuracy_per_task = stream_testing(method.model, test_stream)
            ET = time()
            full_run += (ET-TT)/3600
            # Training code here

            print("\n####################:")
            print(
                "Test method={}, batch={}, acc={}, avg_acc={}".format(
                    method.name,
                    cur_batch,
                    accuracy_per_task,
                    round(mean(accuracy_per_task), 3),
                )
            )
            print( (round(ET-TT,3), round(full_run,3)) )
            sys.stdout.flush()
            for t in range(number_of_tasks):
                scoreboard["acc_{}_T{}".format(method.name, t)].append(
                    accuracy_per_task[t]
                )
            if ((cur_batch + 1) % save_step == 0) and (SAVE_INTERMEDIATE_MODELS):
                method.model.save(
                    MODEL_CACHE + method.name + "_batch_{}".format(cur_batch)
                )


# convert from dict to dataframe
df = pd.DataFrame.from_dict(scoreboard)

# compute average accuracy
for method in METHODS:
    df["acc_avg_{}".format(method.name)] = df[
        ["acc_{}_T{}".format(method.name, t) for t in range(number_of_tasks)]
    ].mean(axis=1)

df["acc_avg_pretrain"] = df[
    ["acc_pretrain_T{}".format(t) for t in range(number_of_tasks)]
].mean(axis=1)

# compute forgetting
for method in METHODS:
    for t in range(number_of_tasks):
        max_acc = pretrained_acc[t]
        forgetting = []

        for step in range(total_number_of_steps):
            value = df.at[step, "acc_{}_T{}".format(method.name, t)]
            diff = 0 if value >= max_acc else max_acc - value
            forgetting.append(diff)
            if value > max_acc:
                max_acc = value

        df["forget_{}_T{}".format(method.name, t)] = forgetting

    df["forget_avg_{}".format(method.name)] = df[
        ["forget_{}_T{}".format(method.name, t) for t in range(number_of_tasks)]
    ].mean(axis=1)


results_filename = "{}.csv".format(strftime("d%d_m%m_y%Y_%H%M%S"))
print("Outputing results to file: {}".format(results_filename))
df.to_csv(results_filename, index=True, index_label="step")
