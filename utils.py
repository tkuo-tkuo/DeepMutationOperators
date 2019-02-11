import numpy as np
import keras

import random 

class GeneralUtils():

    def __init__(self):
        pass

    ''' function decision 
    Return True with prob
    Input: probability within [0, 1]
    Ouput: True or False 
    '''
    def decision(self, prob):
        assert prob >= 0, 'Probability should in the range of [0, 1]'
        assert prob <= 1, 'Probability should in the range of [0, 1]'
        return random.random() < prob

    def shuffle(self, a):
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
        return shuffled_a

    def shuffle_in_uni(self, a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    def shuffle_in_uni_with_permutation(self, a, b, permutation):
        assert len(a) == len(b)
        shuffled_a, shuffled_b = a.copy(), b.copy()
        shuffled_permutation = self.shuffle(permutation)

        for index in range(len(permutation)):
            old_index, new_index = permutation[index], shuffled_permutation[index]
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
            
        return shuffled_a, shuffled_b

    def print_layer_info(self, layer):
        layer_config = layer.get_config()
        print('Print layer configuration information:')
        for key, value in layer_config.items():
            print(key, value)

    '''
    SMM stands for source-level mutated model 
    This function looks quite terrible and messy, should be simplified
    '''
    def print_messages_SMO(self, mode, train_datas=None, train_labels=None, mutated_datas=None, mutated_labels=None, model=None, mutated_model=None, mutation_ratio=0):
        if mode == 'DR' or mode == 'DM':
            print('Before ' + mode)
            print('Train data shape:', train_datas.shape)
            print('Train labels shape:', train_labels.shape)
            print('')

            print('After ' + mode + ', where the mutation ratio is', mutation_ratio)
            print('Train data shape:', mutated_datas.shape)
            print('Train labels shape:', mutated_labels.shape)
            print('')
        elif mode in ['LR', 'LAs', 'AFRs']:
            print('Original untrained model:')
            model.summary()
            print('')

            print('Mutated untrained model:')
            mutated_model.summary()
            print('')
        else:
            pass
    
    '''
    MMM stands for model-level mutated model 
    '''
    def print_messages_MMM_generators(self, mode, network=None, test_datas=None, test_labels=None, model=None, mutated_model=None, STD=0.1, mutation_ratio=0):
        if mode == 'GF':
            print('Before ' + mode)
            network.evaluate_model(model, test_datas, test_labels)
            print('After ' + mode + ', where the mutation ratio is', mutation_ratio, 'and STD is', STD)
            network.evaluate_model(mutated_model, test_datas, test_labels, mode)
        elif mode in ['WS', 'NEB', 'NAI', 'NS']:
            print('Before ' + mode)
            network.evaluate_model(model, test_datas, test_labels)
            print('After ' + mode + ', where the mutation ratio is', mutation_ratio)
            network.evaluate_model(mutated_model, test_datas, test_labels, mode)
        elif mode in ['LD', 'LAm', 'AFRm']:
            print('Before ' + mode)
            model.summary()
            network.evaluate_model(model, test_datas, test_labels)

            print('After ' + mode)
            mutated_model.summary()
            network.evaluate_model(mutated_model, test_datas, test_labels, mode)
        else:
            pass

class ModelUtils():

    def __init__(self):
        pass

    def model_copy(self, model, mode=''):
        original_layers = [l for l in model.layers]
        suffix = '_copy_' + mode 
        new_model = keras.models.clone_model(model)
        for index, layer in enumerate(new_model.layers):
            original_layer = original_layers[index]
            original_weights = original_layer.get_weights()
            layer.name = layer.name + suffix
            layer.set_weights(original_weights)
        new_model.name = new_model.name + suffix
        return new_model

    def get_booleans_of_layers_should_be_mutated(self, num_of_layers, indices):
        if indices == None:
            booleans_for_layers = np.full(num_of_layers, True)
        else:
            booleans_for_layers = np.full(num_of_layers, False)
            for index in indices:
                booleans_for_layers[index] = True
        return booleans_for_layers 

    def weight_of_layers_compare(self, old_model, new_model):
        old_layers = [l for l in old_model.layers]
        new_layers = [l for l in new_model.layers]
        assert len(old_layers) == len(new_layers)
        num_of_layers = len(old_layers)
        booleans_for_layers = np.full(num_of_layers, True)
        for index in range(num_of_layers):
            old_layer, new_layer = old_layers[index], new_layers[index]
            old_layer_weights, new_layer_weights = old_layer.get_weights(), new_layer.get_weights()
            if len(old_layer_weights) == 0:
                print(True)
                continue

            is_equal_connections = np.array_equal(old_layer_weights[0], new_layer_weights[0])
            is_equal_biases = np.array_equal(old_layer_weights[1], new_layer_weights[1])
            is_equal = is_equal_connections and is_equal_biases
            print(is_equal)

class ExaminationalUtils():

    def __init__(self):
        pass


    def mutation_ratio_range_check(self, mutation_ratio):
        assert mutation_ratio >= 0, 'Mutation ratio attribute should in the range [0, 1]'
        assert mutation_ratio <= 1, 'Mutation ratio attribute should in the range [0, 1]'
        pass 

    def training_dataset_consistent_length_check(self, lst_a, lst_b):
        assert len(lst_a) == len(lst_b), 'Training datas and labels should have the same length'
        pass

    def mutated_layer_indices_check(self, num_of_layers, indices):
        if indices is not None:
            for index in indices:
                assert index >= 0, 'Index should be positive'
                assert index < num_of_layers, 'Index should not be out of range, where index should be smaller than ' + str(num_of_layers)
                pass 