import os

import utils
import source_mut_operators
import network


class SourceMutatedModelGenerators():

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.network = network.SimplyNetwork()
        self.program_mut_opts = source_mut_operators.ProgramMutationOperators()
        self.data_mut_opts = source_mut_operators.DataMutationOperators()

    def integration_test(self, verbose=False):
        mutation_ratio = 0.01
        try:
            self.generate_model_by_LR_mutation('LR_model1', verbose=verbose)
            self.generate_model_by_LAs_mutation('LAs_model1', verbose=verbose)
            self.generate_model_by_AFRs_mutation('AFRs_model1', verbose=verbose)
            self.generate_model_by_DR_mutation('DR_model1', mutation_ratio, verbose=verbose)
            self.generate_model_by_LE_mutation('LE_model1', mutation_ratio, verbose=verbose)
            self.generate_model_by_DM_mutation('DM_model1', mutation_ratio, verbose=verbose)
            self.generate_model_by_DF_mutation('DF_model1', verbose=verbose)
            self.generate_model_by_NP_mutation('NP_model1', mutation_ratio, verbose=verbose)
            print('integration test for source-level mutation operators: pass')
        except Exception as e:
            print('integration test for source-level mutation operators: fail')
            print('Error message:', e)

    def generate_model_by_DR_mutation(self, name_of_file, mutation_ratio, verbose=False):
        '''
        Data Reptition(DR): Duplicates training data
        Source-level mutation testing operator
        Target: training data
        Level: Global/Local

        return M'(a new mutated model based on mutated training data and original model)
        '''
        # original data
        (train_datas, train_labels), (test_datas,
                                       test_labels) = self.network.load_data()
        # DR mutation on training data
        repeated_train_datas, repeated_train_labels = self.data_mut_opts.DR_mut(
            train_datas, train_labels, mutation_ratio)
        model = self.network.create_model()
        model.fit(repeated_train_datas, repeated_train_labels,
                  epochs=20, verbose=False)

        if verbose:
            print('before data repetition')
            print('train data shape:', train_datas.shape)
            print('train labels shape:', train_labels.shape)
            print('')

            print('after data repetition, where the mutation ratio is', mutation_ratio)
            print('train data shape:', repeated_train_datas.shape)
            print('train labels shape:', repeated_train_labels.shape)
            print('')

            loss, acc = model.evaluate(test_datas, test_labels)
            print('DR mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
            print('')

        file_name = name_of_file + '.h5'
        model.save(file_name)
        print('Mutated model by DR(Data Repetition) is successfully created and saved as', file_name)

    def generate_model_by_LE_mutation(self, name_of_file, mutation_ratio, verbose=False):
        '''
        Label Error(LE): Falisify results (e.g., labels) of data
        Source-level mutation testing operator
        Target: training data
        Level: Global/Local

        return M'(a new mutated model based on mutated training data and original model)
        '''
        # original data
        (train_datas, train_labels), (test_datas,
                                       test_labels) = self.network.load_data()
        # randomly select a portion of data and mislabel them
        train_datas, train_labels = self.data_mut_opts.LE_mut(
            train_datas, train_labels, 0, 9, mutation_ratio)
        model = self.network.create_model()
        model.fit(train_datas, train_labels, epochs=20, verbose=False)

        if verbose:
            loss, acc = model.evaluate(test_datas, test_labels)
            print('LE mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
            print('')

        file_name = name_of_file + '.h5'
        model.save(file_name)
        print('Mutated model by LE(Label Error) is successfully created and saved as', file_name)

    def generate_model_by_DM_mutation(self, name_of_file, mutation_ratio, verbose=False):
        '''
        Data Missing(DM): Remove selected data
        Source-level mutation testing operator
        Target: training data
        Level: Global/Local

        return M'(a new mutated model based on mutated training data and original model)
        '''
        # original data
        (train_datas, train_labels), (test_datas,
                                       test_labels) = self.network.load_data()

        if verbose:
            print('before data missing')
            print('train data shape:', train_datas.shape)
            print('train labels shape:', train_labels.shape)
            print('')

        # randomly select a portion of data and delete them
        train_datas, train_labels = self.data_mut_opts.DM_mut(
            train_datas, train_labels, mutation_ratio)

        model = self.network.create_model()
        model.fit(train_datas, train_labels, epochs=20, verbose=False)

        if verbose:
            print('after data missing, where the mutation ratio is', mutation_ratio)
            print('train data shape:', train_datas.shape)
            print('train labels shape:', train_labels.shape)
            print('')

            loss, acc = model.evaluate(test_datas, test_labels)
            print('DM mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
            print('')

        file_name = name_of_file + '.h5'
        model.save(file_name)
        print('Mutated model by DM(Data Missing) is successfully created and saved as', file_name)

    def generate_model_by_DF_mutation(self, name_of_file, verbose=False):
        '''
        Data Shuffle(DF): Shuffle selected data
        Source-level mutation testing operator
        Target: training data
        Level: Global/Local

        return M'(a new mutated model based on mutated training data and original model)
        '''
        # original data
        (train_datas, train_labels), (test_datas,
                                       test_labels) = self.network.load_data()

        # shuffle the original train data
        shuffled_train_datas, shuffled_train_labels = self.data_mut_opts.DF_mut(
            train_datas, train_labels)
        model = self.network.create_model()
        model.fit(shuffled_train_datas, shuffled_train_labels,
                  epochs=20, verbose=False)

        if verbose:
            loss, acc = model.evaluate(test_datas, test_labels)
            print('DF mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
            print('')

        file_name = name_of_file + '.h5'
        model.save(file_name)
        print('Mutated model by DF(Data Shuffle) is successfully created and saved as', file_name)

    def generate_model_by_NP_mutation(self, name_of_file, mutation_ratio, verbose=False):
        '''
        Noise Perturb(NP):
        Source-level mutation testing operator
        Target: training data
        Level: Global/Local

        return M'(a new mutated model based on mutated training data and original model)
        '''
        # original data
        (train_datas, train_labels), (test_datas,
                                       test_labels) = self.network.load_data()

        # randomly select a portion of data and add noise perturb on them
        train_datas, train_labels = self.data_mut_opts.NP_mut(
            train_datas, train_labels, mutation_ratio)
        model = self.network.create_model()
        model.fit(train_datas, train_labels, epochs=20, verbose=False)

        if verbose:
            loss, acc = model.evaluate(test_datas, test_labels)
            print('NP mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
            print('')

        file_name = name_of_file + '.h5'
        model.save(file_name)
        print('Mutated model by NP(Noise Perturb) is successfully created and saved as', file_name)

    def generate_model_by_LR_mutation(self, name_of_file, verbose=False):
        '''
        Layer Removal(LR):
        Source-level mutation testing operator
        Target: training program
        Level: Global

        return M'(a new mutated model based on original training data and mutated model)

        Note that according to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, 
        LR operator mainly focuses on layers (e.g., Dense, BatchNormalization layer), whose deletion doesn't make 
        too much difference on the mutated model. 

        Note that according to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, 
        a condition is imposed on LR operator. (The input and output of the removed layer should be the same)

        Note that after modification on model, you should compile the model again. 
        Use model.summary() to check the total amount of parameters 

        Note that for source-level mutation operator, do NOT use the trained network. 
        LR operator should create and modify the untrained model and further train the model. 

        Note that LR should not remove the input or output layer (my implementation)

        Note that LR will remove the first layer which satisfies the condition. It means it has 
        one of specified types and has the same input and output. (my implementation)
        '''
        # original data
        (train_datas, train_labels), (test_datas,
                                       test_labels) = self.network.load_data()

        # randomly remove specify layer from model
        model = self.network.create_debug_model()
        LR_model = self.program_mut_opts.LR_mut(model)
        LR_model = self.network.compile_model(LR_model)
        LR_model.fit(train_datas, train_labels, epochs=20, verbose=False)

        if verbose:
            print('Original untrained model:')
            model.summary()
            print('')

            print('Mutated untrained model:')
            LR_model.summary()
            print('')

            loss, acc = LR_model.evaluate(test_datas, test_labels)
            print('LR mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
            print('')

        file_name = name_of_file + '.h5'
        LR_model.save(file_name)
        print('Mutated model by LR(Layer Removal) is successfully created and saved as', file_name)

    def generate_model_by_LAs_mutation(self, name_of_file, verbose=False):
        '''
        Layer Addition(LA):
        Source-level mutation testing operator
        Target: training program
        Level: Global

        return M'(a new mutated model based on original training data and mutated model)

        Note that according to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, 
        LA operator mainly focuses on layers (e.g., Activation, BatchNormalization layer). it has 
        one of specified types and has the same input and output. (my implementation)

        Note that LAs operator will add the layer to the first space with satisfied condition from 
        input to output. (my implementation)
        '''

        # original data
        (train_datas, train_labels), (test_datas,
                                       test_labels) = self.network.load_data()

        # add a layer to DL structure
        model = self.network.create_debug_model()
        LA_model = self.program_mut_opts.LA_mut(model)
        LA_model = self.network.compile_model(LA_model)
        LA_model.fit(train_datas, train_labels, epochs=20, verbose=False)

        if verbose:
            print('Original untrained model:')
            model.summary()
            print('')

            print('Mutated untrained model:')
            LA_model.summary()
            print('')

            loss, acc = LA_model.evaluate(test_datas, test_labels)
            print('LA mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
            print('')

        file_name = name_of_file + '.h5'
        LA_model.save(file_name)
        print('Mutated model by LA(Layer Addition) is successfully created and saved as', file_name)

    def generate_model_by_AFRs_mutation(self, name_of_file, verbose=False):
        '''
        Activation Function Removal(AFR):
        Source-level mutation testing operator
        Target: training program
        Level: Global

        return M'(a new mutated model based on original training data and mutated model)

        Note that AFRs operator randomly remove activation frunction 
        from one of the arbitrary layer except the output layer. (my implementation)
        '''

        # original data
        (train_datas, train_labels), (test_datas,
                                       test_labels) = self.network.load_data()

        # randomly remove an activation function
        model = self.network.create_debug_model()
        AFR_model = self.program_mut_opts.AFR_mut(model)
        AFR_model = self.network.compile_model(AFR_model)
        AFR_model.fit(train_datas, train_labels, epochs=20, verbose=False)

        if verbose:
            print('Original untrained model:')
            model.summary()
            print('')

            print('Mutated untrained model:')
            AFR_model.summary()
            print('')

            loss, acc = AFR_model.evaluate(test_datas, test_labels)
            print('LA mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
            print('')

        file_name = name_of_file + '.h5'
        AFR_model.save(file_name)
        print('Mutated model by AFR(Activation Function Removal) is successfully created and saved as', file_name)
