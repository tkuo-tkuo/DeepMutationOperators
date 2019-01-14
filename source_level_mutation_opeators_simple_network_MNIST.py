from __future__ import absolute_import, division, print_function

import os, math, random
import numpy as np
import tensorflow as tf
from tensorflow import keras

number_of_train_data = 5000
number_of_test_data = 1000
resize_width = 28
resize_height = 28

# DR seems not influence the accuracy much
# LE do influence the accuracy dramatically

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

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


def generate_and_save_normal_model(name_of_file):

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


def load_model(name_of_file):

    file_name = name_of_file + '.h5'
    return keras.models.load_model(file_name)


def generate_model_by_DR_mutation(name_of_file, mutation_ratio):
    '''
    Data Reptition(DR):
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
    shuffled_train_images, shuffled_train_labels = shuffle_in_uni(train_images, train_labels)

    # select a portion of data and mislabel them
    number_of_duplicate = math.floor(number_of_train_data * mutation_ratio)
    repeated_train_images = shuffled_train_images[:number_of_duplicate]
    repeated_train_labels = shuffled_train_labels[:number_of_duplicate]
    repeated_train_images = np.append(train_images, repeated_train_images, axis=0)
    repeated_train_labels = np.append(train_labels, repeated_train_labels, axis=0)
    print('after data repetition, where the mutation ratio is', mutation_ratio)
    print('train data shape:', repeated_train_images.shape)
    print('train labels shape:', repeated_train_labels.shape)
    print('')

    model = create_model()

    model.fit(repeated_train_images, repeated_train_labels, epochs=20, verbose=False)
    loss, acc = model.evaluate(test_images, test_labels)
    print('DR mutation operator executed')
    print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
    print('')

    file_name = name_of_file + '.h5'
    model.save(file_name)

    print('Mutated model by DR(Data Repetition) is created and saved as', file_name)

def generate_model_by_LE_mutation(name_of_file, mutation_ratio):
    '''
    Label Error(LE):
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
    Data Missing(DM):
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
    Data Shuffle(DF):
    Source-level mutation testing operator
    Target: training data
    Level: Global/Local

    return M'(a new mutated model based on mutated training data and original model)
    '''
    # original data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # shuffle the original train data
    shuffled_train_images, shuffled_train_labels = shuffle_in_uni(train_images, train_labels)

    model = create_model()

    model.fit(shuffled_train_images, shuffled_train_labels, epochs=20, verbose=False)
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
    number_of_noise_perturbs = math.floor(number_of_train_data * mutation_ratio)
    permutation = np.random.permutation(len(train_images))
    permutation = permutation[:number_of_noise_perturbs]
    random_train_images = np.random.standard_normal(train_images.shape) / 10.0
    for old_index, new_index in enumerate(permutation):
        train_images[new_index] = train_images[new_index] + random_train_images[new_index]

    model = create_model()

    model.fit(train_images, train_labels, epochs=20, verbose=False)
    loss, acc = model.evaluate(test_images, test_labels)
    print('NP mutation operator executed')
    print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
    print('')

    file_name = name_of_file + '.h5'
    model.save(file_name)

    print('Mutated model by NP(Noise Perturb) is created and saved as', file_name)
