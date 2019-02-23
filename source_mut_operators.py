import tensorflow as tf
import numpy as np
import keras

import random
import math
import utils

class SourceMutationOperatorsUtils():

    def __init__(self):
        self.LR_mut_candidates = ['Dense', 'BatchNormalization']
        self.LA_mut_candidates = [keras.layers.ReLU(), keras.layers.BatchNormalization()]

    def LA_get_random_layer(self):
        num_of_LA_candidates = len(self.LA_mut_candidates)
        random_index = random.randint(0, num_of_LA_candidates - 1)
        return self.LA_mut_candidates[random_index]

    def LR_model_scan(self, model):
        index_of_suitable_layers = []
        layers = [l for l in model.layers]
        for index, layer in enumerate(layers):
            layer_name = type(layer).__name__
            is_in_candidates = layer_name in self.LR_mut_candidates
            has_same_input_output_shape = layer.input.shape.as_list() == layer.output.shape.as_list()
            should_be_removed = is_in_candidates and has_same_input_output_shape

            if index == 0 or index == (len(model.layers) - 1):
                continue

            if should_be_removed:
                index_of_suitable_layers.append(index)
        return index_of_suitable_layers

    def LAs_model_scan(self, model):
        index_of_suitable_spots = []
        layers = [l for l in model.layers]
        for index, layer in enumerate(layers):
            if index == (len(model.layers) - 1):
                continue

            index_of_suitable_spots.append(index)
        return index_of_suitable_spots

    def AFRs_model_scan(self, model):
        index_of_suitable_spots = []
        layers = [l for l in model.layers]
        for index, layer in enumerate(layers):
            if index == (len(model.layers) - 1):
                continue

            try:
                if layer.activation is not None:
                    index_of_suitable_spots.append(index)
            except:
                pass
        return index_of_suitable_spots

class SourceMutationOperators():

    def __init__(self):
        self.utils = utils.GeneralUtils()
        self.check = utils.ExaminationalUtils()
        self.model_utils = utils.ModelUtils()
        self.SMO_utils = SourceMutationOperatorsUtils()

    def DR_mut(self, train_dataset, model, mutation_ratio):
        deep_copied_model = self.model_utils.model_copy(model, 'DR')
        train_datas, train_labels = train_dataset
        self.check.mutation_ratio_range_check(mutation_ratio)   
        self.check.training_dataset_consistent_length_check(train_datas, train_labels)     

        # shuffle the original train data
        shuffled_train_datas, shuffled_train_labels = self.utils.shuffle_in_uni(train_datas, train_labels)

        # select a portion of data and reproduce
        number_of_train_data = len(train_datas)
        number_of_duplicate = math.floor(number_of_train_data * mutation_ratio)
        repeated_train_datas = shuffled_train_datas[:number_of_duplicate]
        repeated_train_labels = shuffled_train_labels[:number_of_duplicate]
        repeated_train_datas = np.append(train_datas, repeated_train_datas, axis=0)
        repeated_train_labels = np.append(train_labels, repeated_train_labels, axis=0)
        return (repeated_train_datas, repeated_train_labels), deep_copied_model

    def LE_mut(self, train_dataset, model, label_lower_bound, label_upper_bound, mutation_ratio):
        deep_copied_model = self.model_utils.model_copy(model, 'LE')
        train_datas, train_labels = train_dataset
        LE_train_datas, LE_train_labels = train_datas.copy(), train_labels.copy()
        self.check.mutation_ratio_range_check(mutation_ratio)   
        self.check.training_dataset_consistent_length_check(LE_train_datas, LE_train_labels)
        
        number_of_train_data = len(LE_train_datas)
        number_of_error_labels = math.floor(number_of_train_data * mutation_ratio)
        permutation = np.random.permutation(number_of_train_data)
        permutation = permutation[:number_of_error_labels]
        for old_index, new_index in enumerate(permutation):
            while True:
                val = random.randint(label_lower_bound, label_upper_bound)
                num_of_classes = label_upper_bound - label_lower_bound + 1
                val = keras.utils.np_utils.to_categorical(val, num_of_classes)
                if np.array_equal(LE_train_labels[new_index], val):
                    continue
                else: 
                    LE_train_labels[new_index] = val
                    break
        return (LE_train_datas, LE_train_labels), deep_copied_model

    def DM_mut(self, train_dataset, model, mutation_ratio):
        deep_copied_model = self.model_utils.model_copy(model, 'DM')
        train_datas, train_labels = train_dataset
        DM_train_datas, DM_train_labels = train_datas.copy(), train_labels.copy()
        self.check.mutation_ratio_range_check(mutation_ratio)   
        self.check.training_dataset_consistent_length_check(DM_train_datas, DM_train_labels)

        number_of_train_data = len(DM_train_datas)
        number_of_error_labels = math.floor(number_of_train_data * mutation_ratio)

        permutation = np.random.permutation(number_of_train_data)
        permutation = permutation[:number_of_error_labels]

        DM_train_datas = np.delete(DM_train_datas, permutation, 0)
        DM_train_labels = np.delete(DM_train_labels, permutation, 0)
        return (DM_train_datas, DM_train_labels), deep_copied_model

    def DF_mut(self, train_dataset, model, mutation_ratio):
        deep_copied_model = self.model_utils.model_copy(model, 'DF')
        train_datas, train_labels = train_dataset
        DF_train_datas, DF_train_labels = train_datas.copy(), train_labels.copy()
        self.check.mutation_ratio_range_check(mutation_ratio)   
        self.check.training_dataset_consistent_length_check(DF_train_datas, DF_train_labels)
        
        number_of_train_data = len(DF_train_datas)
        number_of_shuffled_datas = math.floor(number_of_train_data * mutation_ratio)

        permutation = np.random.permutation(number_of_train_data)
        permutation = permutation[:number_of_shuffled_datas]

        DF_train_datas, DF_train_labels = self.utils.shuffle_in_uni_with_permutation(DF_train_datas, DF_train_labels, permutation) 
        return (DF_train_datas, DF_train_labels), deep_copied_model

    def NP_mut(self, train_dataset, model, mutation_ratio, STD=0.1):
        deep_copied_model = self.model_utils.model_copy(model, 'NP')
        train_datas, train_labels = train_dataset
        NP_train_datas, NP_train_labels = train_datas.copy(), train_labels.copy()
        self.check.mutation_ratio_range_check(mutation_ratio)   
        self.check.training_dataset_consistent_length_check(NP_train_datas, NP_train_labels)
        
        number_of_train_data = len(NP_train_datas)
        number_of_noise_perturbs = math.floor(number_of_train_data * mutation_ratio)

        permutation = np.random.permutation(number_of_train_data)
        permutation = permutation[:number_of_noise_perturbs]

        random_train_datas = np.random.standard_normal(NP_train_datas.shape) * STD
        for old_index, new_index in enumerate(permutation):
            NP_train_datas[new_index] += random_train_datas[new_index]
        return (NP_train_datas, NP_train_labels), deep_copied_model

    
    def LR_mut(self, train_dataset, model, mutated_layer_indices=None):
        # Copying and some assertions
        deep_copied_model = self.model_utils.model_copy(model, 'LR')
        train_datas, train_labels = train_dataset
        copied_train_datas, copied_train_labels = train_datas.copy(), train_labels.copy()
        self.check.training_dataset_consistent_length_check(copied_train_datas, copied_train_labels)

        # Randomly select from suitable layers instead of the first one 
        index_of_suitable_layers = self.SMO_utils.LR_model_scan(model)
        number_of_suitable_layers = len(index_of_suitable_layers)
        if number_of_suitable_layers == 0:
            print('None of layers be removed')
            print('LR will only remove the layer with the same input and output')
            print('However, there is no suitable layer for the input model')
            return (copied_train_datas, copied_train_labels), deep_copied_model

        new_model = keras.models.Sequential()
        layers = [l for l in deep_copied_model.layers]


        if mutated_layer_indices == None:
            random_picked_layer_index = index_of_suitable_layers[random.randint(0, number_of_suitable_layers-1)]
            print('Selected layer by LR mutation operator', random_picked_layer_index)
            
            for index, layer in enumerate(layers):
                if index == random_picked_layer_index:
                    continue
                new_model.add(layer)
        else:
            self.check.in_suitable_indices_check(index_of_suitable_layers, mutated_layer_indices)

            for index, layer in enumerate(layers):
                if index in mutated_layer_indices:
                    continue
                new_model.add(layer)

        return (copied_train_datas, copied_train_labels), new_model

    def LAs_mut(self, train_dataset, model, mutated_layer_indices=None):
        # Copying and some assertions
        deep_copied_model = self.model_utils.model_copy(model, 'LAs')
        train_datas, train_labels = train_dataset
        copied_train_datas, copied_train_labels = train_datas.copy(), train_labels.copy()
        self.check.training_dataset_consistent_length_check(copied_train_datas, copied_train_labels)

        # Randomly select from suitable spots instead of the first one 
        index_of_suitable_spots = self.SMO_utils.LAs_model_scan(model)
        number_of_suitable_spots = len(index_of_suitable_spots)
        if number_of_suitable_spots == 0:
            print('No layers be added')
            print('There is no suitable spot for the input model')
            return (copied_train_datas, copied_train_labels), deep_copied_model

        new_model = keras.models.Sequential()
        layers = [l for l in deep_copied_model.layers]

        if mutated_layer_indices == None:
            random_picked_spot_index = index_of_suitable_spots[random.randint(0, number_of_suitable_spots-1)]
            print('Selected layer by LAs mutation operator', random_picked_spot_index)

            for index, layer in enumerate(layers):

                if index == random_picked_spot_index:
                    new_model.add(layer)
                    new_model.add(self.SMO_utils.LA_get_random_layer())
                    continue
                new_model.add(layer)
        else:
            self.check.in_suitable_indices_check(index_of_suitable_spots, mutated_layer_indices)

            for index, layer in enumerate(layers):
                if index in mutated_layer_indices:
                    new_model.add(layer)
                    new_model.add(self.SMO_utils.LA_get_random_layer())
                    continue
                new_model.add(layer)


        return (copied_train_datas, copied_train_labels), new_model

    def AFRs_mut(self, train_dataset, model, mutated_layer_indices=None):
        # Copying and some assertions
        deep_copied_model = self.model_utils.model_copy(model, 'AFRs')
        train_datas, train_labels = train_dataset
        copied_train_datas, copied_train_labels = train_datas.copy(), train_labels.copy()
        self.check.training_dataset_consistent_length_check(copied_train_datas, copied_train_labels)

        # Randomly select from suitable layers instead of the first one 
        index_of_suitable_layers = self.SMO_utils.AFRs_model_scan(model)
        number_of_suitable_layers = len(index_of_suitable_layers)
        if number_of_suitable_layers == 0:
            print('None activation of layers be removed')
            print('There is no suitable layer for the input model')
            return (copied_train_datas, copied_train_labels), deep_copied_model
        
        new_model = keras.models.Sequential()
        layers = [l for l in deep_copied_model.layers]

        if mutated_layer_indices == None:
            random_picked_layer_index = index_of_suitable_layers[random.randint(0, number_of_suitable_layers-1)]
            print('Seleced layer by AFRs mutation operator', random_picked_layer_index)

            for index, layer in enumerate(layers):
                if index == random_picked_layer_index:
                    layer.activation = lambda x: x
                    new_model.add(layer)
                    continue
                new_model.add(layer)
        else:
            self.check.in_suitable_indices_check(index_of_suitable_layers, mutated_layer_indices)
            for index, layer in enumerate(layers):
                if index in mutated_layer_indices:
                    layer.activation = lambda x: x
                    new_model.add(layer)
                    continue
                new_model.add(layer)

        return (copied_train_datas, copied_train_labels), new_model