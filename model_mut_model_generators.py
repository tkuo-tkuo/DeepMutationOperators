import os

import source_mut_operators
import utils, network, model_mut_operators


class ModelMutatedModelGenerators():

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.utils = utils.GeneralUtils()
        self.network = network.SimplyNetwork()
        self.model_mut_opts = model_mut_operators.ModelMutationOperators()
        