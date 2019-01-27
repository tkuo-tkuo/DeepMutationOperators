import tensorflow as tf
import numpy as np
import math

import utils

class ModelMutationOperators():
    def __init__(self):
        self.utils = utils.GeneralUtils()

    def GF_on_list(self, lst, prob, STD):
        lst = lst.copy()
        for index in range(len(lst)):
            if self.utils.decision(prob):
                lst[index] += np.random.normal() * STD
        return lst

    def WS_on_list(self, lst, output_index):
        lst = lst.copy()
        input_dim, output_dim = lst.shape

        grabbed_lst = np.empty((input_dim), lst.dtype)

        for index in range(input_dim):
            grabbed_lst[index] = lst[index][output_index]
        shuffle_grabbed_lst = self.utils.shuffle(grabbed_lst)
        for index in range(input_dim):
            lst[index][output_index] = shuffle_grabbed_lst[index]

        return lst

    def NAI_on_list(self, lst, output_index):
        lst = lst.copy()
        input_dim, output_dim = lst.shape

        for index in range(input_dim):
            lst[index][output_index] *= -1

        return lst

    def NS_copy_lst_column(self, a, b, a_index, b_index):
        b = b.copy()
        assert a.shape == b.shape
        input_dim, output_dim = a.shape
        for col_index in range(input_dim):
            b[col_index][b_index] = a[col_index][a_index]

        return b

    def NS_on_list(self, lst, mutation_ratio):
        lst = lst.copy()
        copy_lst = lst.copy()
        shuffled_neurons = np.empty(lst.shape, dtype=lst.dtype)
        # calculate the amount of neurons need to be shuffled 
        input_dim, output_dim = lst.shape
        number_of_switch_neurons = math.floor(output_dim * mutation_ratio)
        # produce permutation for shuffling 
        permutation = np.random.permutation(output_dim)
        permutation = permutation[:number_of_switch_neurons]
        shuffled_permutation = self.utils.shuffle(permutation)

        # shuffle neurons 
        for index in range(len(permutation)):
            copy_lst = self.NS_copy_lst_column(lst, copy_lst, permutation[index], shuffled_permutation[index])

        return copy_lst 

    # This function is designed for debugging purpose, 
    # detecting whether mutation operator truely modeifies the weights of given model 
    def diff_count(self, lst, mutated_lst):
        diff_count = 0
        for index in range(len(lst)):
            if mutated_lst[index] != lst[index]:
                diff_count += 1
        return diff_count

    # STD stands for standard deviation 
    def GF_mut(self, model, mutation_ratio, STD=0.1):
        new_model = tf.keras.models.Sequential()
        layers = [l for l in model.layers]
        for index, layer in enumerate(layers):
            weights = np.array(layer.get_weights())
            new_weights = []
            if len(weights) == 0:
                continue

            for val in weights:
                val_shape = val.shape
                flat_val = val.flatten()
                GF_flat_val = self.GF_on_list(flat_val, mutation_ratio, STD)
                GF_val = GF_flat_val.reshape(val_shape)
                assert val.shape == GF_val.shape
                new_weights.append(GF_val)

            new_weights = np.array(new_weights)
            layer.set_weights(new_weights) 
            new_model.add(layer)

        return new_model

    def WS_mut(self, model, mutation_ratio):
        new_model = tf.keras.models.Sequential()
        layers = [l for l in model.layers]
        for index, layer in enumerate(layers):
            weights = np.array(layer.get_weights())
            new_weights = []
            if len(weights) == 0:
                continue
            
            for val in weights:
                val_shape = val.shape
                if len(val.shape) == 2:
                    input_dim, output_dim = val_shape
                    for output_dim_index in range(output_dim):
                        if self.utils.decision(mutation_ratio):
                            val = self.WS_on_list(val, output_dim_index)

                new_weights.append(val)

            new_weights = np.array(new_weights)
            layer.set_weights(new_weights)
            new_model.add(layer)

        return new_model

    def NEB_mut(self, model, mutation_ratio):
        new_model = tf.keras.models.Sequential()
        layers = [l for l in model.layers]
        for index, layer in enumerate(layers):
                weights = np.array(layer.get_weights())
                new_weights = []
                if len(weights) == 0:
                    continue
                
                for val in weights:
                    copy_val = val.copy()
                    val_shape = val.shape
                    if len(val.shape) == 2:
                        input_dim, output_dim = val_shape
                        for input_index in range(input_dim):
                            if self.utils.decision(mutation_ratio):
                                val[input_index] = 0 

                    new_weights.append(val)

                new_weights = np.array(new_weights)
                layer.set_weights(new_weights)
                new_model.add(layer)

        return new_model

    def NAI_mut(self, model, mutation_ratio):
        new_model = tf.keras.models.Sequential()
        layers = [l for l in model.layers]
        for index, layer in enumerate(layers):
            weights = np.array(layer.get_weights())
            new_weights = []
            if len(weights) == 0:
                continue
            
            for val in weights:
                val_shape = val.shape
                if len(val.shape) == 2:
                    val = self.NAI_on_list(val, output_dim_index)

                new_weights.append(val)

            new_weights = np.array(new_weights)
            layer.set_weights(new_weights)
            new_model.add(layer)

        return new_model


    def NS_mut(self, model, mutation_ratio):
        new_model = tf.keras.models.Sequential()
        layers = [l for l in model.layers]
        for index, layer in enumerate(layers):
            weights = np.array(layer.get_weights())
            new_weights = []
            if len(weights) == 0:
                continue
            
            for val in weights:
                val_shape = val.shape
                if len(val.shape) == 2:
                    val = self.NS_on_list(val, mutation_ratio)

                new_weights.append(val)

            new_weights = np.array(new_weights)
            layer.set_weights(new_weights)
            new_model.add(layer)

        return new_model

