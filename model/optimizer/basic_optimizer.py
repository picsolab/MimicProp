import numpy as np
import pickle
import os
from config import BASE_PATH, OUTPUT_PATH


class BaseOptimizer:
    def __init__(self, seed=101, verbose=True):
        self.seed = seed
        self.verbose = verbose

    def optimize(self, graph, name, alpha=0.5, regularize=False):
        raise NotImplementedError

    @staticmethod
    def build_init_emb_mat(shape, init_emb, word2idx):
        init_emb_mat = np.zeros(shape)
        for word in word2idx:
            init_emb_mat[word2idx[word]] = init_emb[word]
        return init_emb_mat

    @staticmethod
    def _check_out():
        out_dir = OUTPUT_PATH
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    @staticmethod
    def _get_out_filename(name):
        return os.path.join(OUTPUT_PATH,
                            "trained_{}.pkl".format(name))

    def _write_out(self, output, name):
        self._check_out()
        out_file = self._get_out_filename(name)
        with open(out_file, 'wb') as pickle_file:
            pickle.dump(output, pickle_file)

    def save_output(self, output, name):
        if name is not None:
            self._write_out(output, name)
        else:
            print('The trained embeddings are not saved! No name is given!')
