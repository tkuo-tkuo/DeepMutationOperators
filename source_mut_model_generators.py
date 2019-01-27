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
        self.program_mut_opts = source_mut_operators.ProgramMutationOperators()
        self.data_mut_opts = source_mut_operators.DataMutationOperators()
        

    def integration_test(self, verbose=False, with_checkpoint=False):
        mutation_ratio = 0.1
        try:
            self.network.train_and_save_normal_model('normal_model', verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_DR_mutation('DR_model', mutation_ratio, verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_LE_mutation('LE_model', mutation_ratio, verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_DM_mutation('DM_model', mutation_ratio, verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_DF_mutation('DF_model', verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_NP_mutation('NP_model', mutation_ratio, verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_LR_mutation('LR_model', verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_LAs_mutation('LAs_model', verbose=verbose, with_checkpoint=with_checkpoint)
            self.generate_model_by_AFRs_mutation('AFRs_model', verbose=verbose, with_checkpoint=with_checkpoint)
            print('Integration test for source-level mutation operators: PASS')
        except Exception as e:
            print('Integration test for source-level mutation operators: FAIL')
            print('Error message:', e)
    

    def generate_model_by_DR_mutation(self, name_of_file, mutation_ratio, verbose=False, with_checkpoint=False):
        mode ='DR'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        # DR mutation on training data
        DR_train_datas, DR_train_labels = self.data_mut_opts.DR_mut(train_datas, train_labels, mutation_ratio)
        model = self.network.create_model()
        model = self.network.train_model(model, DR_train_datas, DR_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.utils.print_messages_SMM_generators(mode, train_datas=train_datas, train_labels=train_labels, mutated_datas=DR_train_datas, mutated_labels=DR_train_labels, mutation_ratio=mutation_ratio)            
            self.network.evaluate_model(model, test_datas, test_labels, mode)

        self.network.save_model(model, name_of_file, mode)

    def generate_model_by_LE_mutation(self, name_of_file, mutation_ratio, verbose=False, with_checkpoint=False):
        mode ='LE'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        # randomly select a portion of data and mislabel them
        LE_train_datas, LE_train_labels = self.data_mut_opts.LE_mut(train_datas, train_labels, 0, 9, mutation_ratio)
        model = self.network.create_model()
        model = self.network.train_model(model, LE_train_datas, LE_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.network.evaluate_model(model, test_datas, test_labels, mode)

        self.network.save_model(model, name_of_file, mode)

    def generate_model_by_DM_mutation(self, name_of_file, mutation_ratio, verbose=False, with_checkpoint=False):
        mode = 'DM'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        # randomly select a portion of data and delete them
        DM_train_datas, DM_train_labels = self.data_mut_opts.DM_mut(train_datas, train_labels, mutation_ratio)
        model = self.network.create_model()
        model = self.network.train_model(model, DM_train_datas, DM_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.utils.print_messages_SMM_generators(mode, train_datas=train_datas, train_labels=train_labels, mutated_datas=DM_train_datas, mutated_labels=DM_train_labels, mutation_ratio=mutation_ratio)            
            self.network.evaluate_model(model, test_datas, test_labels, mode)

        self.network.save_model(model, name_of_file, mode)

    def generate_model_by_DF_mutation(self, name_of_file, verbose=False, with_checkpoint=False):
        mode = 'DF'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        # shuffle the original train data
        DF_train_datas, DF_train_labels = self.data_mut_opts.DF_mut(train_datas, train_labels)
        model = self.network.create_model()
        model = self.network.train_model(model, DF_train_datas, DF_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.network.evaluate_model(model, test_datas, test_labels, mode)

        self.network.save_model(model, name_of_file, mode)

    def generate_model_by_NP_mutation(self, name_of_file, mutation_ratio, verbose=False, with_checkpoint=False):
        mode = 'NP'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        # randomly select a portion of data and add noise perturb on them
        NP_train_datas, NP_train_labels = self.data_mut_opts.NP_mut(train_datas, train_labels, mutation_ratio)
        model = self.network.create_model()
        model = self.network.train_model(model, NP_train_datas, NP_train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.network.evaluate_model(model, test_datas, test_labels, mode)

        self.network.save_model(model, name_of_file, mode)

    def generate_model_by_LR_mutation(self, name_of_file, verbose=False, with_checkpoint=False):
        mode = 'LR'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        # randomly remove specify layer from model
        model = self.network.create_debug_model()
        LR_model = self.program_mut_opts.LR_mut(model)
        LR_model = self.network.compile_model(LR_model)
        LR_model = self.network.train_model(LR_model, train_datas, train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.utils.print_messages_SMM_generators(mode, model=model, mutated_model=LR_model)            
            self.network.evaluate_model(LR_model, test_datas, test_labels, mode)

        self.network.save_model(LR_model, name_of_file, mode)

    def generate_model_by_LAs_mutation(self, name_of_file, verbose=False, with_checkpoint=False):
        mode = 'LAs'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        # add a layer to DL structure
        model = self.network.create_debug_model()
        LA_model = self.program_mut_opts.LA_mut(model)
        LA_model = self.network.compile_model(LA_model)
        LA_model = self.network.train_model(LA_model, train_datas, train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.utils.print_messages_SMM_generators(mode, model=model, mutated_model=LA_model)            
            self.network.evaluate_model(LA_model, test_datas, test_labels, mode)

        self.network.save_model(LA_model, name_of_file, mode)

    def generate_model_by_AFRs_mutation(self, name_of_file, verbose=False, with_checkpoint=False):
        mode = 'AFRs'
        # original data
        (train_datas, train_labels), (test_datas, test_labels) = self.network.load_data()
        # randomly remove an activation function
        model = self.network.create_debug_model()
        AFR_model = self.program_mut_opts.AFR_mut(model)
        AFR_model = self.network.compile_model(AFR_model)
        AFR_model = self.network.train_model(AFR_model, train_datas, train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            self.utils.print_messages_SMM_generators(mode, model=model, mutated_model=AFR_model)            
            self.network.evaluate_model(AFR_model, test_datas, test_labels, mode)

        self.network.save_model(AFR_model, name_of_file, mode)


    # def load_and_train_model_test(self, file_name):
    #     model = self.network.load_model(file_name)
    #     (train_datas, train_labels), (test_datas,
    #                                   test_labels) = self.network.load_data()   
    #     loss, acc = model.evaluate(test_datas, test_labels)
    #     print('model accurancy: {:5.2f}%'.format(100*acc))
    #     print('')

    #     filepath = "model2-{epoch:02d}-{loss:.4f}.h5"
    #     checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #         filepath, monitor='loss', verbose=5, save_best_only=True, mode='min')
    #     callbacks_list = [checkpoint]
    #     (train_datas, train_labels), (test_datas,
    #                                   test_labels) = self.network.load_data()
    #     model.fit(train_datas, train_labels, epochs=40, batch_size=50, callbacks=callbacks_list)