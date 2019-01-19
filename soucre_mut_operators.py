from __future__ import absolute_import, division, print_function

import os
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

import utils


number_of_train_data = 5000
number_of_test_data = 1000
resize_width = 28
resize_height = 28


def shuffle_in_uni(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def load_data():
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.mnist.load_data()

    train_labels = train_labels[:number_of_train_data]
    test_labels = test_labels[:number_of_test_data]

    train_images = train_images[:number_of_train_data].reshape(
        -1, resize_width * resize_height) / 255.0
    test_images = test_images[:number_of_test_data].reshape(
        -1, resize_width * resize_height) / 255.0

    return (train_images, train_labels), (test_images, test_labels)

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(resize_width * resize_height, )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model = compile_model(model)

    return model


def create_debug_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(resize_width * resize_height, )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model = compile_model(model)

    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model

def load_model(name_of_file):

    file_name = name_of_file + '.h5'
    return keras.models.load_model(file_name)

def train_and_save_normal_model(name_of_file):

    print('Current tensorflow version:', tf.__version__)
    print('')

    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = create_model()

    print('train dataset shape:', train_images.shape)
    print('test dataset shape:', test_images.shape)
    print('network architecture:')
    model.summary()
    print('')

    model.fit(train_images, train_labels, epochs=20, verbose=False)
    loss, acc = model.evaluate(test_images, test_labels)
    print('Trained model, accurancy: {:5.2f}%'.format(100*acc))
    print('')

    file_name = name_of_file + '.h5'
    model.save(file_name)

    print('New model is created and saved as', file_name)




def generate_model_by_DR_mutation(name_of_file, mutation_ratio):
    '''
    Data Reptition(DR): Duplicates training data
    Source-level mutation testing operator
    Target: training data
    Level: Global/Local

    return M'(a new mutated model based on mutated training data and original model)
    '''
    # original data
    (train_images, train_labels), (test_images, test_labels) = load_data()
    print('before data repetition')
    print('train data shape:', train_images.shape)
    print('train labels shape:', train_labels.shape)
    print('')

    # shuffle the original train data
    shuffled_train_images, shuffled_train_labels = shuffle_in_uni(
        train_images, train_labels)

    # select a portion of data and mislabel them
    number_of_duplicate = math.floor(number_of_train_data * mutation_ratio)
    repeated_train_images = shuffled_train_images[:number_of_duplicate]
    repeated_train_labels = shuffled_train_labels[:number_of_duplicate]
    repeated_train_images = np.append(
        train_images, repeated_train_images, axis=0)
    repeated_train_labels = np.append(
        train_labels, repeated_train_labels, axis=0)
    print('after data repetition, where the mutation ratio is', mutation_ratio)
    print('train data shape:', repeated_train_images.shape)
    print('train labels shape:', repeated_train_labels.shape)
    print('')

    model = create_model()

    model.fit(repeated_train_images, repeated_train_labels,
              epochs=20, verbose=False)
    loss, acc = model.evaluate(test_images, test_labels)
    print('DR mutation operator executed')
    print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
    print('')

    file_name = name_of_file + '.h5'
    model.save(file_name)

    print('Mutated model by DR(Data Repetition) is created and saved as', file_name)


def generate_model_by_LE_mutation(name_of_file, mutation_ratio):
    '''
    Label Error(LE): Falisify results (e.g., labels) of data
    Source-level mutation testing operator
    Target: training data
    Level: Global/Local

    return M'(a new mutated model based on mutated training data and original model)
    '''
    # original data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # randomly select a portion of data and mislabel them
    number_of_error_labels = math.floor(number_of_train_data * mutation_ratio)
    permutation = np.random.permutation(len(train_images))
    permutation = permutation[:number_of_error_labels]
    for old_index, new_index in enumerate(permutation):
        train_labels[new_index] = random.randint(0, 9)

    model = create_model()

    model.fit(train_images, train_labels, epochs=20, verbose=False)
    loss, acc = model.evaluate(test_images, test_labels)
    print('LE mutation operator executed')
    print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
    print('')

    file_name = name_of_file + '.h5'
    model.save(file_name)

    print('Mutated model by LE(Label Error) is created and saved as', file_name)


def generate_model_by_DM_mutation(name_of_file, mutation_ratio):
    '''
    Data Missing(DM): Remove selected data
    Source-level mutation testing operator
    Target: training data
    Level: Global/Local

    return M'(a new mutated model based on mutated training data and original model)
    '''
    # original data
    (train_images, train_labels), (test_images, test_labels) = load_data()
    print('before data missing')
    print('train data shape:', train_images.shape)
    print('train labels shape:', train_labels.shape)
    print('')

    # randomly select a portion of data and delete them
    number_of_error_labels = math.floor(number_of_train_data * mutation_ratio)
    permutation = np.random.permutation(len(train_images))
    permutation = permutation[:number_of_error_labels]

    train_images = np.delete(train_images, permutation, 0)
    train_labels = np.delete(train_labels, permutation, 0)

    print('after data missing, where the mutation ratio is', mutation_ratio)
    print('train data shape:', train_images.shape)
    print('train labels shape:', train_labels.shape)
    print('')

    model = create_model()

    model.fit(train_images, train_labels, epochs=20, verbose=False)
    loss, acc = model.evaluate(test_images, test_labels)
    print('DM mutation operator executed')
    print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
    print('')

    file_name = name_of_file + '.h5'
    model.save(file_name)

    print('Mutated model by DM(Data Missing) is created and saved as', file_name)


def generate_model_by_DF_mutation(name_of_file):
    '''
    Data Shuffle(DF): Shuffle selected data
    Source-level mutation testing operator
    Target: training data
    Level: Global/Local

    return M'(a new mutated model based on mutated training data and original model)
    '''
    # original data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # shuffle the original train data
    shuffled_train_images, shuffled_train_labels = shuffle_in_uni(
        train_images, train_labels)

    model = create_model()

    model.fit(shuffled_train_images, shuffled_train_labels,
              epochs=20, verbose=False)
    loss, acc = model.evaluate(test_images, test_labels)
    print('DF mutation operator executed')
    print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
    print('')

    file_name = name_of_file + '.h5'
    model.save(file_name)

    print('Mutated model by DF(Data Shuffle) is created and saved as', file_name)


def generate_model_by_NP_mutation(name_of_file, mutation_ratio):
    '''
    Noise Perturb(NP):
    Source-level mutation testing operator
    Target: training data
    Level: Global/Local

    return M'(a new mutated model based on mutated training data and original model)
    '''
    # original data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # randomly select a portion of data and add noise perturb on them
    number_of_noise_perturbs = math.floor(
        number_of_train_data * mutation_ratio)
    permutation = np.random.permutation(len(train_images))
    permutation = permutation[:number_of_noise_perturbs]
    random_train_images = np.random.standard_normal(train_images.shape) / 10.0
    for old_index, new_index in enumerate(permutation):
        train_images[new_index] = train_images[new_index] + \
            random_train_images[new_index]

    model = create_model()

    model.fit(train_images, train_labels, epochs=20, verbose=False)
    loss, acc = model.evaluate(test_images, test_labels)
    print('NP mutation operator executed')
    print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
    print('')

    file_name = name_of_file + '.h5'
    model.save(file_name)

    print('Mutated model by NP(Noise Perturb) is created and saved as', file_name)


def generate_model_by_LR_mutation(name_of_file, verbose=False):
    '''
    Layer Removal(LR):
    Source-level mutation testing operator
    Target: training program
    Level: Global

    return M'(a new mutated model based on original training data and mutated model)

    Note that according to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, 
    LR operator mainly focuses on layers (e.g., Dense, BatchNormalization layer), whose deletion doesn't make 
    too much difference on the mutated model. 

    Note that according to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, 
    a condition is imposed on LR operator. (The input and output of the removed layer should be the same)

    Note that after modification on model, you should compile the model again. 
    Use model.summary() to check the total amount of parameters 

    Note that for source-level mutation operator, do NOT use the trained network. 
    LR operator should create and modify the untrained model and further train the model. 

    Note that LR should not remove the input or output layer (my implementation)

    Note that LR will remove the first layer which satisfies the condition. It means it has 
    one of specified types and has the same input and output. (my implementation)
    '''

    # original data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # original model (untrained)
    model = create_debug_model()

    # randomly remove specify layer from model
    LR_model = DLPM_Utils.LR_mut(model)

    # training
    LR_model.fit(train_images, train_labels, epochs=20, verbose=False)

    if verbose:
        print('Original untrained model:')
        model.summary()
        print('')

        print('Mutated untrained model:')
        LR_model.summary()
        print('')

        loss, acc = LR_model.evaluate(test_images, test_labels)
        print('LR mutation operator executed')
        print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
        print('')

    file_name = name_of_file + '.h5'
    LR_model.save(file_name)

    print('Mutated model by LR(Layer Removal) is successfully created and saved as', file_name)


def generate_model_by_LAs_mutation(name_of_file, verbose=False):
    '''
    Layer Addition(LA):
    Source-level mutation testing operator
    Target: training program
    Level: Global

    return M'(a new mutated model based on original training data and mutated model)

    Note that according to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, 
    LA operator mainly focuses on layers (e.g., Activation, BatchNormalization layer). it has 
    one of specified types and has the same input and output. (my implementation)

    Note that LAs operator will add the layer to the first space with satisfied condition from 
    input to output. (my implementation)
    '''

    # original data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # original model (untrained)
    model = create_debug_model()

    # Add a layer to DL structure 
    LA_model = DLPM_Utils.LA_mut(model)

    # training
    LA_model.fit(train_images, train_labels, epochs=20, verbose=False)

    if verbose:
        print('Original untrained model:')
        model.summary()
        print('')

        print('Mutated untrained model:')
        LA_model.summary()
        print('')

        loss, acc = LA_model.evaluate(test_images, test_labels)
        print('LA mutation operator executed')
        print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
        print('')

    file_name = name_of_file + '.h5'
    LA_model.save(file_name)

    print('Mutated model by LA(Layer Addition) is successfully created and saved as', file_name)


def generate_model_by_AFRs_mutation(name_of_file, verbose=False):
    '''
    Activation Function Removal(AFR):
    Source-level mutation testing operator
    Target: training program
    Level: Global

    return M'(a new mutated model based on original training data and mutated model)

    Note that AFRs operator randomly remove activation frunction 
    from one of the arbitrary layer except the output layer. (my implementation)
    '''

    # original data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # original model (untrained)
    model = create_debug_model()

    # Add a layer to DL structure 
    AFR_model = DLPM_Utils.AFR_mut(model)

    # training
    AFR_model.fit(train_images, train_labels, epochs=20, verbose=False)

    if verbose:
        print('Original untrained model:')
        model.summary()
        print('')

        print('Mutated untrained model:')
        AFR_model.summary()
        print('')

        loss, acc = AFR_model.evaluate(test_images, test_labels)
        print('LA mutation operator executed')
        print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
        print('')

    file_name = name_of_file + '.h5'
    AFR_model.save(file_name)

    print('Mutated model by AFR(Activation Function Removal) is successfully created and saved as', file_name)

# Ignore some warning message like AVX2 extension
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DLPM_Utils = utils.DLProgramMutationUtils()

# generate_model_by_LR_mutation('LR_model1', False)
# generate_model_by_LAs_mutation('LAs_model1', False)
# generate_model_by_AFRs_mutation('AFRs_model1', False)

generate_model_by_DR_mutation('DR_model1', 0.01)
generate_model_by_LE_mutation('LE_model1', 0.01)