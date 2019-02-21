import tensorflow as tf
import numpy as np
import keras

import math, random

import utils

class ModelMutationOperatorsUtils():

    def __init__(self):
        self.utils = utils.GeneralUtils()
        self.LD_mut_candidates = ['Dense']
        self.LAm_mut_candidates = ['Dense']

    def GF_on_list(self, lst, mutation_ratio, prob_distribution, STD, lower_bound, upper_bound, lam):
        copy_lst = lst.copy()
        number_of_data = len(copy_lst)
        permutation = self.utils.generate_permutation(number_of_data, mutation_ratio)

        if prob_distribution == 'normal':
            copy_lst[permutation] += np.random.normal(scale=STD, size=len(permutation))
        elif prob_distribution == 'uniform':
            copy_lst[permutation] += np.random.uniform(low=lower_bound, high=upper_bound, size=len(permutation))
        elif prob_distribution == 'exponential':
            assert lam is not 0
            scale = 1 / lam 
            copy_lst[permutation] += np.random.exponential(scale=sclae, size=len(permutation))
        else:
            pass

        return copy_lst

    def WS_on_Dense_list(self, lst, output_index):
        copy_lst = lst.copy()
        grabbed_lst = copy_lst[:, output_index]
        shuffle_grabbed_lst = self.utils.shuffle(grabbed_lst)
        copy_lst[:, output_index] = shuffle_grabbed_lst
        return copy_lst

    def WS_on_Conv2D_list(self, lst, output_channel_index):
        copy_lst = lst.copy()
        filter_width, filter_height, num_of_input_channels, num_of_output_channels = copy_lst.shape

        copy_lst = np.reshape(copy_lst, (filter_width * filter_height * num_of_input_channels, num_of_output_channels))
        grabbled_lst = copy_lst[:, output_channel_index]
        shuffle_grabbed_lst = self.utils.shuffle(grabbled_lst)
        copy_lst[:, output_channel_index] = shuffle_grabbed_lst
        copy_lst = np.reshape(copy_lst, (filter_width, filter_height, num_of_input_channels, num_of_output_channels))
        return copy_lst

    def NS_on_Dense_list(self, lst, mutation_ratio):
        first_copy_lst = lst.copy()
        second_copy_lst = lst.copy()
        input_dim, output_dim = first_copy_lst.shape

        permutation = self.utils.generate_permutation(input_dim, mutation_ratio)
        shuffled_permutation = self.utils.shuffle(permutation)

        for index in range(len(permutation)):
            second_copy_lst[shuffled_permutation[index], :] = first_copy_lst[permutation[index], :]
        return second_copy_lst 

    def NS_on_Conv2D_list(self, lst, mutation_ratio):
        first_copy_lst = lst.copy()
        second_copy_lst = lst.copy()
        filter_width, filter_height, num_of_input_channels, num_of_output_channels = first_copy_lst.shape

        permutation = self.utils.generate_permutation(num_of_input_channels, mutation_ratio)
        shuffled_permutation = self.utils.shuffle(permutation)

        for index in range(len(permutation)):
            second_copy_lst[:, :, shuffled_permutation[index], :] = first_copy_lst[:, :, permutation[index], :]
        return second_copy_lst

    def LD_model_scan(self, model):
        index_of_suitable_layers = []
        layers = [l for l in model.layers]
        for index, layer in enumerate(layers):
            layer_name = type(layer).__name__
            is_in_candidates = layer_name in self.LD_mut_candidates
            has_same_input_output_shape = layer.input.shape.as_list() == layer.output.shape.as_list()
            should_be_removed = is_in_candidates and has_same_input_output_shape
            if index == 0 or index == (len(model.layers) - 1):
                continue
            if should_be_removed:
                index_of_suitable_layers.append(index)
        return index_of_suitable_layers

    def LAm_model_scan(self, model):
        index_of_suitable_spots = []
        layers = [l for l in model.layers]
        for index, layer in enumerate(layers):
            layer_type_name = type(layer).__name__
            is_in_candidates = layer_type_name in self.LAm_mut_candidates
            has_same_input_output_shape = layer.input.shape.as_list() == layer.output.shape.as_list()
            should_be_added = is_in_candidates and has_same_input_output_shape
            if should_be_added:
                index_of_suitable_spots.append(index)

        return index_of_suitable_spots

    def AFRm_model_scan(self, model):
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



class ModelMutationOperators():
    
    def __init__(self):
        self.utils = utils.GeneralUtils()
        self.model_utils = utils.ModelUtils()
        self.check = utils.ExaminationalUtils()
        self.MMO_utils = ModelMutationOperatorsUtils()

    def GF_mut(self, model, mutation_ratio, prob_distribution='normal', STD=0.1, lower_bound=None, upper_bound=None, lam=None, mutated_layer_indices=None):
        self.check.mutation_ratio_range_check(mutation_ratio)  
        
        valid_prob_distribution_types = ['normal', 'uniform']
        assert prob_distribution in valid_prob_distribution_types, 'The probability distribution type ' + prob_distribution + ' is not implemented in GF mutation operator'
        if prob_distribution == 'uniform' and ((lower_bound is None) or (upper_bound is None)):
            raise ValueError('In uniform distribution, users are required to specify the lower bound and upper bound of noises')
        if prob_distribution == 'exponential' and (lam is None):
            raise ValueError('In exponential distribution, users are required to specify the lambda value')

        GF_model = self.model_utils.model_copy(model, 'GF')
        layers = [l for l in GF_model.layers]

        num_of_layers = len(layers)
        self.check.valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        layers_should_be_mutated = self.model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers, mutated_layer_indices)

        for index, layer in enumerate(layers):
            weights = layer.get_weights()
            new_weights = []
            if not (len(weights) == 0) and layers_should_be_mutated[index]:
                for val in weights:
                    val_shape = val.shape
                    flat_val = val.flatten()
                    GF_flat_val = self.MMO_utils.GF_on_list(flat_val, mutation_ratio, prob_distribution, STD, lower_bound, upper_bound, lam)
                    GF_val = GF_flat_val.reshape(val_shape)
                    new_weights.append(GF_val)
                layer.set_weights(new_weights) 

        return GF_model

    def WS_mut(self, model, mutation_ratio, mutated_layer_indices=None):
        self.check.mutation_ratio_range_check(mutation_ratio)

        WS_model = self.model_utils.model_copy(model, 'WS')
        layers = [l for l in WS_model.layers]

        num_of_layers = len(layers)
        self.check.valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        layers_should_be_mutated = self.model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers, mutated_layer_indices)

        for index, layer in enumerate(layers):
            weights = layer.get_weights()
            layer_name = type(layer).__name__
            new_weights = []
            if not (len(weights) == 0):
                for val in weights:
                    val_shape = val.shape
                    if (len(val.shape) is not 1) and layers_should_be_mutated[index]:
                        if layer_name == 'Conv2D':
                            filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
                            permutation = self.utils.generate_permutation(num_of_output_channels, mutation_ratio)
                            for output_channel_index in permutation:
                                val = self.MMO_utils.WS_on_Conv2D_list(val, output_channel_index)
                        elif layer_name == 'Dense':
                            input_dim, output_dim = val_shape
                            permutation = self.utils.generate_permutation(output_dim, mutation_ratio)
                            for output_dim_index in permutation:
                                val = self.MMO_utils.WS_on_Dense_list(val, output_dim_index)
                        else:
                            pass
                    new_weights.append(val)
                layer.set_weights(new_weights)

        return WS_model

    def NEB_mut(self, model, mutation_ratio, mutated_layer_indices=None):
        self.check.mutation_ratio_range_check(mutation_ratio)

        NEB_model = self.model_utils.model_copy(model, 'NEB')
        layers = [l for l in NEB_model.layers]

        num_of_layers = len(layers)
        self.check.valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        layers_should_be_mutated = self.model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers, mutated_layer_indices)

        for index, layer in enumerate(layers):
            weights = layer.get_weights()
            layer_name = type(layer).__name__
            new_weights = []
            if not (len(weights) == 0):
                for val in weights:
                    val_shape = val.shape
                    if (len(val.shape) is not 1) and layers_should_be_mutated[index]:
                        if layer_name == 'Conv2D':
                            filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
                            permutation = self.utils.generate_permutation(num_of_input_channels, mutation_ratio)
                            for input_channel_index in permutation:
                                val[:, :, input_channel_index, :] = 0
                        elif layer_name == 'Dense':
                            input_dim, output_dim = val_shape
                            permutation = self.utils.generate_permutation(input_dim, mutation_ratio)
                            for input_index in permutation:
                                val[input_index] = 0 
                        else:
                            pass
                    new_weights.append(val)
                layer.set_weights(new_weights)

        return NEB_model

    def NAI_mut(self, model, mutation_ratio, mutated_layer_indices=None):
        self.check.mutation_ratio_range_check(mutation_ratio)

        NAI_model = self.model_utils.model_copy(model, 'NAI')
        layers = [l for l in NAI_model.layers]

        num_of_layers = len(layers)
        self.check.valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        layers_should_be_mutated = self.model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers, mutated_layer_indices)

        for index, layer in enumerate(layers):
            weights = layer.get_weights()
            layer_name = type(layer).__name__
            new_weights = []
            if not (len(weights) == 0):
                for val in weights:
                    val_shape = val.shape
                    if (len(val.shape) is not 1) and layers_should_be_mutated[index]:
                        if layer_name == 'Conv2D':
                            filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
                            permutation = self.utils.generate_permutation(num_of_output_channels, mutation_ratio)
                            for output_channel_index in permutation:
                                val[:, :, :, output_channel_index] *= -1
                        elif layer_name == 'Dense':
                            input_dim, output_dim = val_shape
                            permutation = self.utils.generate_permutation(output_dim, mutation_ratio)
                            for output_dim_index in permutation:
                                val[:, output_dim_index] *= -1
                        else:
                            pass 
                    new_weights.append(val)
                layer.set_weights(new_weights)

        return NAI_model


    def NS_mut(self, model, mutation_ratio, mutated_layer_indices=None):
        self.check.mutation_ratio_range_check(mutation_ratio)

        NS_model = self.model_utils.model_copy(model, 'NS')
        layers = [l for l in NS_model.layers]

        num_of_layers = len(layers)
        self.check.valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        layers_should_be_mutated = self.model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers, mutated_layer_indices)
        for index, layer in enumerate(layers):
            weights = layer.get_weights()
            layer_name = type(layer).__name__
            new_weights = []
            if not (len(weights) == 0):
                for val in weights:
                    val_shape = val.shape
                    if (len(val.shape) is not 1) and layers_should_be_mutated[index]:
                        if layer_name == 'Conv2D':
                            val = self.MMO_utils.NS_on_Conv2D_list(val, mutation_ratio) 
                        elif layer_name == 'Dense':
                            val = self.MMO_utils.NS_on_Dense_list(val, mutation_ratio)
                        else:
                            pass
                    new_weights.append(val)
                layer.set_weights(new_weights)

        return NS_model

    def LD_mut(self, model, mutated_layer_indices=None):
        LD_model = self.model_utils.model_copy(model, 'LD')

        # Randomly select from suitable layers instead of the first one 
        index_of_suitable_layers = self.MMO_utils.LD_model_scan(model)
        number_of_suitable_layers = len(index_of_suitable_layers)
        if number_of_suitable_layers == 0:
            print('None of layers be removed')
            print('LD will only remove the layer with the same input and output')
            print('')
            return LD_model

        layers = [l for l in LD_model.layers]
        new_model = keras.models.Sequential()

        if mutated_layer_indices == None:
            random_picked_layer_index = index_of_suitable_layers[random.randint(0, number_of_suitable_layers-1)]
            print('Selected layer by LD mutation operator', random_picked_layer_index)

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


        return new_model

    def LAm_mut(self, model, mutated_layer_indices=None):
        LAm_model = self.model_utils.model_copy(model, 'LAm')
        copied_LAm_model = self.model_utils.model_copy(model, 'insert')

        # Randomly select from suitable spots instead of the first one 
        index_of_suitable_spots = self.MMO_utils.LAm_model_scan(model)
        number_of_suitable_spots = len(index_of_suitable_spots)
        if number_of_suitable_spots == 0:
            print('No layers be added')
            print('LAm will only add the layer with the same input and output')
            print('There is no suitable spot for the input model')
            print('')
            return LAm_model

        new_model = keras.models.Sequential()
        layers = [l for l in LAm_model.layers]
        copy_layers = [l for l in copied_LAm_model.layers]

        if mutated_layer_indices == None:
            random_picked_spot_index = index_of_suitable_spots[random.randint(0, number_of_suitable_spots-1)]
            print('Selected layer by LRm mutation operator', random_picked_spot_index)

            for index, layer in enumerate(layers):
                if index == random_picked_spot_index:
                    new_model.add(layer)
                    copy_layer = copy_layers[index]
                    new_model.add(copy_layer)
                    continue
                new_model.add(layer)
        else:
            self.check.in_suitable_indices_check(index_of_suitable_spots, mutated_layer_indices)

            for index, layer in enumerate(layers):
                if index in mutated_layer_indices:
                    new_model.add(layer)
                    copy_layer = copy_layers[index]
                    new_model.add(copy_layer)
                    continue
                new_model.add(layer)

        return new_model

    def AFRm_mut(self, model, mutated_layer_indices=None):
        AFRm_model = self.model_utils.model_copy(model, 'AFRm')

        # Randomly select from suitable layers instead of the first one 
        index_of_suitable_layers = self.MMO_utils.AFRm_model_scan(model)
        number_of_suitable_layers = len(index_of_suitable_layers)
        if number_of_suitable_layers == 0:
            print('No activation be removed')
            print('Except the output layer, there is no activation function can be removed')
            print('')
            return AFRm_model

        new_model = keras.models.Sequential()
        layers = [l for l in AFRm_model.layers]

        if mutated_layer_indices == None:
            random_picked_layer_index = index_of_suitable_layers[random.randint(0, number_of_suitable_layers-1)]
            print('Selected layer by AFRm mutation operator', random_picked_layer_index)

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

        return new_model