import os

import source_mut_operators
import network


class ModelMutatedModelGenerators():

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.network = network.SimplyNetwork()