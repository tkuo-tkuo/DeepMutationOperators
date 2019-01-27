import tensorflow as tf
import numpy as np

import random
import math
import utils

class ProgramMutationOperators():
    def __init__(self):
        self.LR_mut_candidates = ['Dense', 'BatchNormalization']
        self.LA_mut_candidates = [tf.keras.layers.ReLU(), tf.keras.layers.BatchNormalization()]

    def get_random_layer_LA(self):
        num_of_LA_candidates = len(self.LA_mut_candidates)
        random_index = random.randint(0, num_of_LA_candidates - 1)
        return self.LA_mut_candidates[random_index]

    def LR_mut(self, model):
        new_model = tf.keras.models.Sequential()
        layers = [l for l in model.layers]
        any_layer_removed = False
        for index, layer in enumerate(layers):
            layer_name = type(layer).__name__
            is_in_candidates = layer_name in self.LR_mut_candidates
            has_same_input_output_shape = layer.input.shape.as_list() == layer.output.shape.as_list()
            should_be_removed = is_in_candidates and has_same_input_output_shape

            if index == 0 or index == (len(model.layers) - 1):
                new_model.add(layer)
                continue

            if should_be_removed and not any_layer_removed:
                any_layer_removed = True
                continue

            new_model.add(layer)

        if not any_layer_removed:
            print('None of layers be removed')
            print('LR will only remove the layer with the same input and output')

        return new_model

    def LA_mut(self, model):
        '''
        Add a layer to the DNN structure, focuing on adding layers like Activation, BatchNormalization
        '''
        new_model = tf.keras.models.Sequential()
        layers = [l for l in model.layers]
        any_layer_added = False
        for index, layer in enumerate(layers):

            if index == (len(model.layers) - 1):
                new_model.add(layer)
                continue

            if not any_layer_added:
                any_layer_added = True
                new_model.add(layer)

                # Randomly add one of the specified types of layers
                # Currently, LA mutation operator will randomly add one of specificed types of layers right after input layer
                new_model.add(self.get_random_layer_LA())
                continue

            new_model.add(layer)

        return new_model

    def AFR_mut(self, model):
        '''
        Remova an activation from one of layers in DNN structure.
        '''
        new_model = tf.keras.models.Sequential()
        layers = [l for l in model.layers]
        any_layer_AF_removed = False
        for index, layer in enumerate(layers):

            if index == (len(model.layers) - 1):
                new_model.add(layer)
                continue

            if not any_layer_AF_removed:
                any_layer_AF_removed = True

                # Randomly remove an activation function from one of layers
                # Currently, AFR mutation operator will randomly remove the first activation it meets
                layer.activation = lambda x: x
                new_model.add(layer)
                continue

            new_model.add(layer)

        return new_model


class DataMutationOperators():

    def __init__(self):
        self.utils = utils.GeneralUtils()

    def DR_mut(self, train_datas, train_labels, mutation_ratio):
        number_of_train_data = len(train_datas)

        # shuffle the original train data
        shuffled_train_datas, shuffled_train_labels = self.utils.shuffle_in_uni(train_datas, train_labels)

        # select a portion of data and reproduce
        number_of_duplicate = math.floor(number_of_train_data * mutation_ratio)
        repeated_train_datas = shuffled_train_datas[:number_of_duplicate]
        repeated_train_labels = shuffled_train_labels[:number_of_duplicate]
        repeated_train_datas = np.append(train_datas, repeated_train_datas, axis=0)
        repeated_train_labels = np.append(train_labels, repeated_train_labels, axis=0)
        return repeated_train_datas, repeated_train_labels

    def LE_mut(self, train_datas, train_labels, label_lower_bound, label_upper_bound, mutation_ratio):
        number_of_train_data = len(train_datas)
        number_of_error_labels = math.floor(number_of_train_data * mutation_ratio)
        permutation = np.random.permutation(len(train_datas))
        permutation = permutation[:number_of_error_labels]
        for old_index, new_index in enumerate(permutation):
            train_labels[new_index] = random.randint(label_lower_bound, label_upper_bound)
        return train_datas, train_labels

    def DM_mut(self, train_datas, train_labels, mutation_ratio):
        number_of_train_data = len(train_datas)
        number_of_error_labels = math.floor(number_of_train_data * mutation_ratio)

        assert number_of_train_data >= number_of_error_labels
        permutation = np.random.permutation(number_of_train_data)
        permutation = permutation[:number_of_error_labels]

        train_datas = np.delete(train_datas, permutation, 0)
        train_labels = np.delete(train_labels, permutation, 0)
        return train_datas, train_labels

    def DF_mut(self, train_datas, train_labels):
        return self.utils.shuffle_in_uni(train_datas, train_labels)

    def NP_mut(self, train_datas, train_labels, mutation_ratio):
        number_of_train_data = len(train_datas)
        number_of_noise_perturbs = math.floor(number_of_train_data * mutation_ratio)

        assert number_of_train_data >= number_of_noise_perturbs
        permutation = np.random.permutation(number_of_train_data)
        permutation = permutation[:number_of_noise_perturbs]

        random_train_datas = np.random.standard_normal(train_datas.shape) / 10.0
        for old_index, new_index in enumerate(permutation):
            train_datas[new_index] += random_train_datas[new_index]
        return train_datas, train_labels
