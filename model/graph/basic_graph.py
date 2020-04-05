import numpy as np
import os
import pickle
from config import BASE_PATH


class BasicGraph():
    def __init__(self, data_container,
                 name, seed=101, verbose = True):

        self.seed = seed
        self.data_container = data_container
        self.name = name
        self.binary = data_container.binary
        self.graph = self.load_graph()
        if verbose:
            self.graph_stats()

    def build_graph(self):
        raise NotImplementedError

    def graph_stats(self):
        raise NotImplementedError

    @staticmethod
    def _check_cache():
        cache_dir = os.path.join(BASE_PATH, "_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    @staticmethod
    def calculate_score(s1, s2):
        if s1 * s2 > 0:
            return 1 - np.abs(s1 - s2)
        elif s1 * s2 < 0:
            return - np.abs(s1 - s2)
        elif s1 * s2 == 0:
            raise ValueError

    def _get_cache_filename(self):
        return os.path.join(BASE_PATH, "_cache/",
                            "_cachedGraph_{}.pkl".format(self.name))

    def _write_cache(self, graph):
        self._check_cache()

        cache_file = self._get_cache_filename()

        with open(cache_file, 'wb') as pickle_file:
            pickle.dump(graph, pickle_file)

    def load_graph(self):

        # NOT using cache
        if self.name is None:
            print("cache deactivated for graph!")
            return self.build_graph()

        # using cache
        cache_file = self._get_cache_filename()

        if os.path.exists(cache_file):
            print("Loading Graph {} from cache!".format(self.name))
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("No cache file for {} ... ".format(self.name))
            print("Building the Graph from scratch ...")
            graph = self.build_graph()
            self._write_cache(graph)
            return graph


    @staticmethod
    def build_adjacency(edges, word2idx):
        print('Creating the Adjacency Matrix from edges ...')
        _dim = len(edges)
        adjacency = np.zeros(shape=(_dim, _dim))
        for word in edges:
            w_idx = word2idx[word]
            for s_word in edges[word]:
                s_w_idx = word2idx[s_word]
                adjacency[w_idx][s_w_idx] = edges[word][s_word]
        assert np.allclose(adjacency, adjacency.T, rtol=1e-05, atol=1e-08)
        return adjacency


