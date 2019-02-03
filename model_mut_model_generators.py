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
        mutation_ratios = [0.3]
        for mutation_ratio in mutation_ratios:
            # self.generate_model_by_GF_mutation('GF_model', mutation_ratio, verbose=verbose)
            # self.generate_model_by_WS_mutation('WS_model', mutation_ratio, verbose=verbose)
            # self.generate_model_by_NEB_mutation('NEB_model', mutation_ratio, verbose=verbose)
            # self.generate_model_by_NAI_mutation('NAI_model', mutation_ratio, verbose=verbose)
            # self.generate_model_by_NS_mutation('NS_model', mutation_ratio, verbose=verbose)
            # self.generate_model_by_LD_mutation('LD_model', mutation_ratio, verbose=verbose)
            # self.generate_model_by_LAm_mutation('LAm_model', mutation_ratio, verbose=verbose)
            self.generate_model_by_AFRm_mutation('AFRm_model', mutation_ratio, verbose=verbose)
         
    # This functions is used for debugging
    def print_acc_saved_model(self, name_of_file, mode):
        print('Debugging...', 'the accurancy saved in the same as above')
        prefix = 'training/'
        file_name = prefix + name_of_file 
        model = self.network.load_model(file_name)
        self.network.evaluate_model(model, self.test_datas, self.test_labels, mode=mode)
    

    def generate_model_by_GF_mutation(self, name_of_file, mutation_ratio, STD=0.1, verbose=False):
        mode ='GF'
        model = self.network.load_model('debug_model') 
        GF_model = self.model_mut_opts.GF_mut(model, mutation_ratio, STD=STD)
        GF_model = self.network.compile_model(GF_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=GF_model, STD=STD, mutation_ratio=mutation_ratio) 

        self.network.save_model(GF_model, name_of_file, mode) 


    def generate_model_by_WS_mutation(self, name_of_file, mutation_ratio, verbose=False):
        mode ='WS'
        model = self.network.load_model('debug_model') 
        WS_model = self.model_mut_opts.WS_mut(model, mutation_ratio)
        WS_model = self.network.compile_model(WS_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=WS_model, mutation_ratio=mutation_ratio) 

        self.network.save_model(WS_model, name_of_file, mode) 

    def generate_model_by_NEB_mutation(self, name_of_file, mutation_ratio, verbose=False):
        mode ='NEB'
        model = self.network.load_model('debug_model') 
        NEB_model = self.model_mut_opts.NEB_mut(model, mutation_ratio)
        NEB_model = self.network.compile_model(NEB_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=NEB_model, mutation_ratio=mutation_ratio) 

        self.network.save_model(NEB_model, name_of_file, mode) 
    
    def generate_model_by_NAI_mutation(self, name_of_file, mutation_ratio, verbose=False):
        mode ='NAI'
        model = self.network.load_model('debug_model') 
        NAI_model = self.model_mut_opts.NAI_mut(model, mutation_ratio)
        NAI_model = self.network.compile_model(NAI_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=NAI_model, mutation_ratio=mutation_ratio) 

        self.network.save_model(NAI_model, name_of_file, mode) 

    def generate_model_by_NS_mutation(self, name_of_file, mutation_ratio, verbose=False):
        mode ='NS'
        model = self.network.load_model('debug_model') 
        NS_model = self.model_mut_opts.NS_mut(model, mutation_ratio)
        NS_model = self.network.compile_model(NS_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=NS_model, mutation_ratio=mutation_ratio) 

        self.network.save_model(NS_model, name_of_file, mode) 

    def generate_model_by_LD_mutation(self, name_of_file, mutation_ratio, verbose=False):
        mode ='LD'
        model = self.network.load_model('debug_model') 
        LD_model = self.model_mut_opts.LD_mut(model)
        LD_model = self.network.compile_model(LD_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=LD_model) 

        self.network.save_model(LD_model, name_of_file, mode) 

    def generate_model_by_LAm_mutation(self, name_of_file, mutation_ratio, verbose=False):
        mode ='LAm'
        model = self.network.load_model('debug_model') 
        LAm_model = self.model_mut_opts.LAm_mut(model)
        LAm_model = self.network.compile_model(LAm_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=LAm_model) 

        self.network.save_model(LAm_model, name_of_file, mode) 

    def generate_model_by_AFRm_mutation(self, name_of_file, mutation_ratio, verbose=False):
        mode ='AFRm'
        model = self.network.load_model('debug_model') 
        AFRm_model = self.model_mut_opts.AFRm_mut(model)
        AFRm_model = self.network.compile_model(AFRm_model)

        if verbose:
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_labels, model=model, mutated_model=AFRm_model) 

        self.network.save_model(AFRm_model, name_of_file, mode) 