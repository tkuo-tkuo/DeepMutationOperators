import random
import numpy as np

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

    # SMM stands for sourve-level mutated model 
    # This func looks quite terrible, should be simplified
    def print_messages_SMM_generators(self, mode, train_datas=None, train_labels=None, mutated_datas=None, mutated_labels=None, model=None, mutated_model=None, mutation_ratio=0):
        if mode == 'DR':
            print('Before data repetition')
            print('Train data shape:', train_datas.shape)
            print('Train labels shape:', train_labels.shape)
            print('')

            print('After data repetition, where the mutation ratio is', mutation_ratio)
            print('Train data shape:', mutated_datas.shape)
            print('Train labels shape:', mutated_labels.shape)
            print('')
        elif mode == 'DM':
            print('Before data missing')
            print('Train data shape:', train_datas.shape)
            print('Train labels shape:', train_labels.shape)
            print('')

            print('After data missing, where the mutation ratio is', mutation_ratio)
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