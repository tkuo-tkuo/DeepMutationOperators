import tensorflow as tf

import random


def compile_model(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model

class GeneralUtils():

    def shuffle_in_uni(self, a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


class DLProgramMutationUtils():
    def __init__(self):
        self.LR_mut_candidates = ['Dense', 'BatchNormalization']
        self.LA_mut_candidates = [ tf.keras.layers.ReLU(), tf.keras.layers.BatchNormalization()]

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

        new_model = compile_model(new_model)
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

        new_model = compile_model(new_model)
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

        new_model = compile_model(new_model)
        return new_model
