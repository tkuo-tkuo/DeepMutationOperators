import os

import source_mut_operators
import utils, network, model_mut_operators

import keras 


class ModelMutatedModelGenerators():

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.utils = utils.GeneralUtils()
        self.network = network.SimplyNetwork()
        self.model_mut_opts = model_mut_operators.ModelMutationOperators()
        (_, _), (test_datas, test_labels) = self.network.load_data()
        self.test_datas = test_datas
        self.test_labels = test_labels
    
    def integration_test(self, verbose=False):
        # mutation_ratios = [0.01, 0.1, 0.2, 0.5, 1]
        mutation_ratios = [0.2]
        STD = 0.2
        for mutation_ratio in mutation_ratios:
            # self.generate_model_by_GF_mutation('GF_model', mutation_ratio, STD=STD, verbose=verbose)
            # self.generate_model_by_WS_mutation('WS_model', mutation_ratio, verbose=verbose)
            # self.generate_model_by_NEB_mutation('NEB_model', mutation_ratio, verbose=verbose)
            # self.generate_model_by_NAI_mutation('NAI_model', mutation_ratio, verbose=verbose)
            # self.generate_model_by_NS_mutation('NS_model', mutation_ratio, verbose=verbose)
            # self.generate_model_by_LD_mutation('LD_model', mutation_ratio, verbose=verbose)
            # self.generate_model_by_LAm_mutation('LAm_model', mutation_ratio, verbose=verbose)
            self.generate_model_by_AFRm_mutation('AFRm_model', mutation_ratio, verbose=verbose)
            
        # try:
        #     print('Integration test for model-level mutation operators: PASS')
        # except Exception as e:
        #     print('Integration test for model-level mutation operators: FAIL')
        #     print('Error message:', e)

    # This functions is used for debugging
    def print_acc_saved_model(self, name_of_file, mode):
        print('Debugging...', 'the accurancy saved in the same as above')
        prefix = 'training/'
        file_name = prefix + name_of_file 
        model = self.network.load_model(file_name)
        self.network.evaluate_model(model, self.test_datas, self.test_labels, mode=mode)
    

    def generate_model_by_GF_mutation(self, name_of_file, mutation_ratio, STD=0.1, verbose=False):
        '''
        GF: Gaussian Fuzzing
        Level: Weight
        Brief Operator Description: Fuzz weight by Gaussian Distribution

        Implementation:
        i.  Each value of weights is fuzzed or not depends on probability(mutation_ratio).
            If the mutation_ratio is 0.1, it means that each value of weights will be fuzzed with the probability of 0.1. 
        ii. GF operator add noise on selected weight, where the noise follows normal distribution ~N(0, std^2)

        Note that GF_mut manipulates the weights on a copy of original DNN, copy_model = tf.keras.models.clone_model(model) for the purpose. 
        Note that GF_mut works well with Dense, Activation, BatchNormalization. However, not guaranteed for convolutional layer yet. 
        Note that a copy for original DNN(copy_model) is needed since the keras does not supply model deep copy currently
        '''
        mode ='GF'
        # load original DNN
        model = self.network.load_model('normal_model') 
        copy_model = self.network.load_model('normal_model')
        GF_model = self.model_mut_opts.GF_mut(copy_model, mutation_ratio, STD=STD)
        GF_model = self.network.compile_model(GF_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=GF_model, STD=STD, mutation_ratio=mutation_ratio) 

        self.network.save_model(GF_model, name_of_file, mode) 


    def generate_model_by_WS_mutation(self, name_of_file, mutation_ratio, verbose=False):
        '''
        WS: Weight Suffling 
        Level: Neuron
        Brief Operator Description: Shfulle selected weights 

        Implementation: 
        i.  Except the input layer, for each layer, select neurons by probability and 
            shuffle the weights of each neuron's connections to the previous layer. For instance,
            the weights of Dense layer are stored in a matrix (2-dimension list) m * n, if neuron j is selected, 
            then all the weights w_?j connecting to neuron j are extracted, shuffled, and injected 
            back in a matrix. 
        
        Note that biases are excluded for consideration.
        
        Problem: 
        i. What about convolutional layer? How can we cope with that?
        '''
        mode ='WS'
        # load original DNN
        model = self.network.load_model('normal_model') 
        copy_model = self.network.load_model('normal_model')
        WS_model = self.model_mut_opts.WS_mut(copy_model, mutation_ratio)
        WS_model = self.network.compile_model(WS_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=WS_model, mutation_ratio=mutation_ratio) 

        self.network.save_model(WS_model, name_of_file, mode) 

    def generate_model_by_NEB_mutation(self, name_of_file, mutation_ratio, verbose=False):
        '''
        NEB: Neuron Effect Blcok  
        Level: Neuron
        Brief Operator Description: Block a neuron effect on following layers 

        Implementation: 
        i.  Similar to WS, for each layer, select neurons by probability (mutation_ratio) and 
            block the effect of selected neurons by setting all the weights connecting to next layer as 0.
            
        Note that biases are excluded for consideration
        
        Note that when mutation ratio is 0.5, mutated model still has the accurancy above 70%. 
        However, the accurancy drops dramatically when mutation ratio approaches to 1
        
        '''
        mode ='NEB'
        # load original DNN
        model = self.network.load_model('normal_model') 
        copy_model = self.network.load_model('normal_model')
        NEB_model = self.model_mut_opts.NEB_mut(copy_model, mutation_ratio)
        NEB_model = self.network.compile_model(NEB_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=NEB_model, mutation_ratio=mutation_ratio) 

        self.network.save_model(NEB_model, name_of_file, mode) 
    
    def generate_model_by_NAI_mutation(self, name_of_file, mutation_ratio, verbose=False):
        '''
        NAI: Neuron Activation Inverse  
        Level: Neuron
        Brief Operator Description: Invert (the sign) the activation status of a neuron

        According to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, 
        invertion of the activation status of a neuron can be achieved by changing the sign 
        of the output value of a neuron before applying its activation function. 
        =>
        This can be actually achieved by randomly selecting neurons with the weights connecting to 
        its previous layer and multiplying -1 to all the selected weights. 
        (Since the output value of a neuron before applying its activation function is the sum of product
        connecting to a neuron.)

        Implementation: 
        i. Similar as WS mutation operator, multiply weights by -1 instead of shuffling

        Problem:
        i. Same problem as WS, what about layers like convolutional layer?
        '''
        mode ='NAI'
        # load original DNN
        model = self.network.load_model('normal_model') 
        copy_model = self.network.load_model('normal_model')
        NAI_model = self.model_mut_opts.NAI_mut(copy_model, mutation_ratio)
        NAI_model = self.network.compile_model(NAI_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=NAI_model, mutation_ratio=mutation_ratio) 

        self.network.save_model(NAI_model, name_of_file, mode) 

    def generate_model_by_NS_mutation(self, name_of_file, mutation_ratio, verbose=False):
        '''
        NS: Neuron Switch 
        Level: Neuron
        Brief Operator Description: Switch two neurons of the same layer

        If you switch neurons multiple times, it's the effect of shuffle a portion of neurons
        =>
        shuffle selected neurons in a layer according to mutation ratio 

        Implementation: 
        i. Get output_dim 
        ii. select a portion of indices among output_dim according to the mutation ratio by permutation()
        iii. shuffle neurons

        Problem:
        i. For this mutation operator, we calculate the number of selected neurons beforehand. 
        However, other mutation operators use probability to achieve certain mutation ratio. 
        Should this mutation operator based on probability? 
        or should other mutation operator based on calculation(determine specific amount according mutation ratio)?
        '''
        mode ='NS'
        # load original DNN
        model = self.network.load_model('normal_model') 
        copy_model = self.network.load_model('normal_model')
        NS_model = self.model_mut_opts.NS_mut(copy_model, mutation_ratio)
        NS_model = self.network.compile_model(NS_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=NS_model, mutation_ratio=mutation_ratio) 

        self.network.save_model(NS_model, name_of_file, mode) 

    def generate_model_by_LD_mutation(self, name_of_file, mutation_ratio, verbose=False):
        '''
        LD: Layer Deactivation
        Level: Layer
        Brief Operator Description: Deactivate the effects of a layer 

        Note that simply removing a layer from a trained DL model can break the model structure. 
        We restrict the LD operator to layers whose input and output shapes are consistent.

        Note that input and output layer should not be removed since removal of input and ouput 
        mutate the model to which totally different from the original one

        To be honest, LD is quite similar to LR operator, LR removes a layer and train while 
        LD simply removes a layer. However, for layer removal, both mutation operators use the
        same code segments.

        Implementation: 
        i.

        '''
        mode ='LD'
        # load original DNN
        model = self.network.load_model('debug_model') 
        copy_model = self.network.load_model('debug_model')
        LD_model = self.model_mut_opts.LD_mut(copy_model, mutation_ratio)
        LD_model = self.network.compile_model(LD_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=LD_model, mutation_ratio=mutation_ratio) 

        self.network.save_model(LD_model, name_of_file, mode) 

    def generate_model_by_LAm_mutation(self, name_of_file, mutation_ratio, verbose=False):
        '''
        LAm: Layer Addition
        Level: Layer
        Brief Operator Description: Add a layer in neuron network 

        Note that LAm operator duplicate and insert copied layer with the condition that the shape of 
        layer input and ouput should be consistent to avoid breaking the original DNNs. 

        To be honest, LAm is quite similar to LAs operator, both of them add a layer with the constriction 
        that the input and output must be the same. The only difference is that the weights of newly added layer
        in LAm need to be copied from the previous layer. 

        Implementation: 
        i.

        Problem:
        i. The simplest implementation like 
        new_model.add(layer)
        new_model.add(layer)
        since two layer being added have the exact same name, the name needs to unique for each layer.

        But the problem is that name is read-only parameter. (something like copy_layer.name = 'copy_layer' is disabled)
        => 
        The alternative is to craft a new layer for addition
        => 
        However, it would be tedious and troublesome to cope with various type of layer. 
        =>
        Solution: downgrade the version of tensorflow from 1.12 to 1.11 and use 2.2.4 instead of tensorflow backend 2.1.6-tf
        '''
        mode ='LAm'
        # load original DNN
        model = self.network.load_model('debug_model') 
        copy_model = self.network.load_model('debug_model')
        copy_model_2 = self.network.load_model('debug_model')

        LAm_model = self.model_mut_opts.LAm_mut(copy_model, copy_model_2, mutation_ratio)
        LAm_model = self.network.compile_model(LAm_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=LAm_model, mutation_ratio=mutation_ratio) 

        self.network.save_model(LAm_model, name_of_file, mode) 



    def generate_model_by_AFRm_mutation(self, name_of_file, mutation_ratio, verbose=False):
        '''
        AFRm: Activation Function Removal
        Level: Layer
        Brief Operator Description: Remove activation functions

        Implementation: 
        i.

        Problem:
        i. 
        '''
        mode ='AFRm'
        # load original DNN
        model = self.network.load_model('debug_model') 
        copy_model = self.network.load_model('debug_model')

        AFRm_model = self.model_mut_opts.AFRm_mut(copy_model, mutation_ratio)
        AFRm_model = self.network.compile_model(AFRm_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=AFRm_model, mutation_ratio=mutation_ratio) 

        self.network.save_model(AFRm_model, name_of_file, mode) 