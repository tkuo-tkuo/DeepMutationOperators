import os

import utils
import source_mut_operators
import network
import tensorflow as tf


class SourceMutatedModelGenerators():

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.utils = utils.GeneralUtils()
        self.network = network.SimplyNetwork()
        self.source_mut_opts = source_mut_operators.SourceMutationOperators()


        # self.data_mut_opts = source_mut_operators.DataMutationOperators()
        

    def integration_test(self, verbose=False, with_checkpoint=False):
        mutation_ratios = [0.1]
        for mutation_ratio in mutation_ratios:
            self.generate_model_by_DR_mutation('DR_model', mutation_ratio, verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_LE_mutation('LE_model', mutation_ratio, verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_DM_mutation('DM_model', mutation_ratio, verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_DF_mutation('DF_model', mutation_ratio, verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_NP_mutation('NP_model', mutation_ratio, verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_LR_mutation('LR_model', verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_LAs_mutation('LAs_model', verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_AFRs_mutation('AFRs_model', verbose=verbose, with_checkpoint=with_checkpoint)

    def generate_model_by_DR_mutation(self, name_of_file, mutation_ratio, verbose=False, with_checkpoint=False):
        mode ='DR'
        # original dataset and model (untrained)
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        model = self.network.create_model()
        # DR mutation on training data
        (DR_train_datas, DR_train_labels), DR_model = self.source_mut_opts.DR_mut((train_datas, train_labels), model, mutation_ratio)
        DR_model = self.network.compile_model(DR_model)
        DR_model = self.network.train_model(DR_model, DR_train_datas, DR_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.utils.print_messages_SMO(mode, train_datas=train_datas, train_labels=train_labels, mutated_datas=DR_train_datas, mutated_labels=DR_train_labels, mutation_ratio=mutation_ratio)            
            self.network.evaluate_model(DR_model, test_datas, test_labels, mode)

        self.network.save_model(DR_model, name_of_file, mode)

    def generate_model_by_LE_mutation(self, name_of_file, mutation_ratio, verbose=False, with_checkpoint=False):
        mode ='LE'
        # original dataset and model (untrained)
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        model = self.network.create_model()
        # randomly select a portion of data and mislabel them
        (LE_train_datas, LE_train_labels), LE_model = self.source_mut_opts.LE_mut((train_datas, train_labels), model, 0, 9, mutation_ratio)
        LE_model = self.network.compile_model(LE_model)
        LE_model = self.network.train_model(LE_model, LE_train_datas, LE_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.network.evaluate_model(LE_model, test_datas, test_labels, mode)

        self.network.save_model(LE_model, name_of_file, mode)

    def generate_model_by_DM_mutation(self, name_of_file, mutation_ratio, verbose=False, with_checkpoint=False):
        mode = 'DM'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        model = self.network.create_model()
        # randomly select a portion of data and delete them
        (DM_train_datas, DM_train_labels), DM_model = self.source_mut_opts.DM_mut((train_datas, train_labels), model, mutation_ratio)
        DM_model = self.network.compile_model(DM_model)
        DM_model = self.network.train_model(DM_model, DM_train_datas, DM_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.utils.print_messages_SMO(mode, train_datas=train_datas, train_labels=train_labels, mutated_datas=DM_train_datas, mutated_labels=DM_train_labels, mutation_ratio=mutation_ratio)            
            self.network.evaluate_model(DM_model, test_datas, test_labels, mode)

        self.network.save_model(DM_model, name_of_file, mode)

    def generate_model_by_DF_mutation(self, name_of_file, mutation_ratio, verbose=False, with_checkpoint=False):
        mode = 'DF'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        model = self.network.create_model()
        # shuffle a portion of original train data
        (DF_train_datas, DF_train_labels), DF_model = self.source_mut_opts.DF_mut((train_datas, train_labels), model, mutation_ratio)
        DF_model = self.network.compile_model(DF_model)
        DF_model = self.network.train_model(DF_model, DF_train_datas, DF_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.network.evaluate_model(DF_model, test_datas, test_labels, mode)

        self.network.save_model(DF_model, name_of_file, mode)

    def generate_model_by_NP_mutation(self, name_of_file, mutation_ratio, verbose=False, with_checkpoint=False):
        mode = 'NP'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        model = self.network.create_model()
        # randomly select a portion of data and add noise perturb on them
        (NP_train_datas, NP_train_labels), NP_model = self.source_mut_opts.NP_mut((train_datas, train_labels), model, mutation_ratio)
        NP_model = self.network.compile_model(NP_model)
        NP_model = self.network.train_model(NP_model, NP_train_datas, NP_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.network.evaluate_model(NP_model, test_datas, test_labels, mode)

        self.network.save_model(NP_model, name_of_file, mode)

    def generate_model_by_LR_mutation(self, name_of_file, verbose=False, with_checkpoint=False):
        mode = 'LR'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        model = self.network.create_debug_model()
        # randomly remove specify layer from model
        (LR_train_datas, LR_train_labels), LR_model = self.source_mut_opts.LR_mut((train_datas, train_labels), model)
        LR_model = self.network.compile_model(LR_model)
        LR_model = self.network.train_model(LR_model, LR_train_datas, LR_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.utils.print_messages_SMO(mode, model=model, mutated_model=LR_model) 
            model = self.network.compile_model(model)
            model = self.network.train_model(model, LR_train_datas, LR_train_labels, name_of_file, with_checkpoint=with_checkpoint)
            self.network.evaluate_model(model, test_datas, test_labels, mode)
            self.network.evaluate_model(LR_model, test_datas, test_labels, mode)

        self.network.save_model(LR_model, name_of_file, mode)

    def generate_model_by_LAs_mutation(self, name_of_file, verbose=False, with_checkpoint=False):
        mode = 'LAs'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        model = self.network.create_debug_model()
        # add a layer to DL structure
        (LAs_train_datas, LAs_train_labels), LAs_model = self.source_mut_opts.LAs_mut( (train_datas, train_labels), model)
        LAs_model = self.network.compile_model(LAs_model)
        LAs_model = self.network.train_model(LAs_model, LAs_train_datas, LAs_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.utils.print_messages_SMO(mode, model=model, mutated_model=LAs_model) 
            model = self.network.compile_model(model)
            model = self.network.train_model(model, LAs_train_datas, LAs_train_labels, name_of_file, with_checkpoint=with_checkpoint)
            self.network.evaluate_model(model, test_datas, test_labels, mode)           
            self.network.evaluate_model(LAs_model, test_datas, test_labels, mode)

        self.network.save_model(LAs_model, name_of_file, mode)

    def generate_model_by_AFRs_mutation(self, name_of_file, verbose=False, with_checkpoint=False):
        mode = 'AFRs'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        model = self.network.create_debug_model()
        # randomly remove an activation function
        (AFRs_train_datas, AFRs_train_labels), AFRs_model = self.source_mut_opts.AFRs_mut((train_datas, train_labels), model)
        AFRs_model = self.network.compile_model(AFRs_model)
        AFRs_model = self.network.train_model(AFRs_model, AFRs_train_datas, AFRs_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.utils.print_messages_SMO(mode, model=model, mutated_model=AFRs_model)   
            model = self.network.compile_model(model)
            model = self.network.train_model(model, AFRs_train_datas, AFRs_train_labels, name_of_file, with_checkpoint=with_checkpoint)
            self.network.evaluate_model(model, test_datas, test_labels, mode)          
            self.network.evaluate_model(AFRs_model, test_datas, test_labels, mode)

        self.network.save_model(AFRs_model, name_of_file, mode)
