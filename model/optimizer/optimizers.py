import numpy as np
from model.optimizer.basic_optimizer import BaseOptimizer
from utils.evaluate import evaluate


class LabelSpreadingOptimizer(BaseOptimizer):
    def __init__(self, seed=101, verbose=True):
        BaseOptimizer.__init__(self, seed=seed, verbose=verbose)

    def optimize(self, graph, name, alpha=0.5, regularize=False):
        print('--------------Start Optimizing:', name, '-------------------')
        if isinstance(graph.graph, list):
            trained_emb = {}

            for idx, _batch_graph in enumerate(graph.graph):
                print('Optimizing Sub-Graph:', idx+1)
                _trained_emb = self._batch_optimize(batch_graph=_batch_graph,
                                                    data_container=graph.data_container,
                                                    alpha=alpha, regularize=regularize)
                trained_emb.update(_trained_emb)
            _lex_emb = {}
            for word in graph.data_container.lexicon_words:
                _lex_emb[word] = graph.data_container.init_emb[word]
            trained_emb.update(_lex_emb)
            self.save_output(trained_emb, name=name)
            print('----------------------Optimization Done----------------------------\n\n')
            return trained_emb

        else:
            trained_emb = self._single_optimize(graph=graph, alpha=alpha, regularize=regularize)
            self.save_output(trained_emb, name=name)
            print('----------------------Optimization Done----------------------------\n\n')
            return trained_emb

    def _batch_optimize(self, batch_graph, data_container, alpha=0.5, regularize=False):
        _word2idx, _idx2word, _adjacency, _ = batch_graph
        _emb_shape = data_container.emb_shape
        _vocab_size = len(_word2idx)
        _init_emb = data_container.init_emb
        _init_emb_mat = self.build_init_emb_mat(shape=(_vocab_size, _emb_shape[0]),
                                                word2idx=_word2idx, init_emb=_init_emb)

        if self.verbose:
            print('--Before Optimization:')
            evaluate(data_container.init_emb,
                     data_container.lexicon)

        print('Building Degree Matrix ...')
        _degree = np.zeros(shape=(_vocab_size, _vocab_size))
        for i in range(_vocab_size):
            _degree[i][i] = np.sum(np.absolute(_adjacency[i]))
            assert _degree[i][i] > 0

        print('Calculating Graph Laplacian ...')
        _reg_degree = np.linalg.matrix_power(np.sqrt(_degree), -1)
        _Laplacian = np.matmul(np.matmul(_reg_degree, _adjacency), _reg_degree)

        print('Starting Label Spreading optimization with **alpha =', alpha, '** ...')

        _optimized = np.matmul(
            np.linalg.matrix_power(
                np.identity(_vocab_size) - alpha * _Laplacian, -1
            ),
            (1 - alpha) * _init_emb_mat
        )

        if regularize:
            optimized = np.copy(_init_emb_mat)
            for i in range(len(_optimized)):
                optimized[i] = _optimized[i] / np.linalg.norm(_optimized[i])
        else:
            optimized = _optimized

        trained_emb = {}
        for word in _word2idx:
            trained_emb[word] = optimized[_word2idx[word]]

        if self.verbose:
            print('--After Optimization:')
            evaluate(trained_emb,
                     data_container.lexicon)

        return trained_emb

    def _single_optimize(self, graph, alpha=0.5, regularize=False):
        _word2idx, _idx2word, _adjacency, _ = graph.graph
        _emb_shape = graph.data_container.emb_shape
        _vocab_size = len(_word2idx)
        _init_emb = graph.data_container.init_emb
        _init_emb_mat = self.build_init_emb_mat(shape=(_vocab_size, _emb_shape[0]),
                                                word2idx=_word2idx, init_emb=_init_emb)
        if self.verbose:
            print('--Before Optimization:')
            evaluate(graph.data_container.init_emb,
                     graph.data_container.lexicon)

        print('Building Degree Matrix ...')
        _degree = np.zeros(shape=(_vocab_size, _vocab_size))
        for i in range(_vocab_size):
            _degree[i][i] = np.sum(np.absolute(_adjacency[i]))
            assert _degree[i][i] > 0

        print('Calculating Graph Laplacian ...')
        _reg_degree = np.linalg.matrix_power(np.sqrt(_degree), -1)
        _Laplacian = np.matmul(np.matmul(_reg_degree, _adjacency), _reg_degree)

        print('Starting Label Spreading optimization with **alpha =', alpha, '** ...')

        _optimized = np.matmul(
            np.linalg.matrix_power(
                np.identity(_vocab_size) - alpha * _Laplacian, -1
            ),
            (1 - alpha) * _init_emb_mat
        )

        if regularize:
            optimized = np.copy(_init_emb_mat)
            for i in range(len(_optimized)):
                optimized[i] = _optimized[i] / np.linalg.norm(_optimized[i])
        else:
            optimized = _optimized

        trained_emb = {}
        for word in _word2idx:
            trained_emb[word] = optimized[_word2idx[word]]

        if self.verbose:
            print('--After Optimization:')
            evaluate(trained_emb,
                     graph.data_container.lexicon)

        return trained_emb


class RetrofitOptimizer(LabelSpreadingOptimizer):

    def _batch_optimize(self, batch_graph, data_container, alpha=0.5, regularize=False):
        _word2idx, _idx2word, _adjacency, _ = batch_graph
        _emb_shape = data_container.emb_shape
        _vocab_size = len(_word2idx)
        _init_emb = data_container.init_emb
        _init_emb_mat = self.build_init_emb_mat(shape=(_vocab_size, _emb_shape[0]),
                                                word2idx=_word2idx, init_emb=_init_emb)

        if self.verbose:
            print('--Before Optimization:')
            evaluate(data_container.init_emb,
                     data_container.lexicon)

        print('Starting Retrofit optimization with **alpha =', alpha, '** ...')

        _num_iters = 15
        _optimized = np.copy(_init_emb_mat)
        _update = np.copy(_init_emb_mat)

        for it in range(_num_iters):
            for i in range(len(_optimized)):
                _temp = alpha * np.dot(_adjacency[i], _optimized) \
                        + (1 - alpha) * _init_emb_mat[i]

                _update[i] = _temp / ((alpha * np.sum(_adjacency[i])) + (1 - alpha))

            _optimized = _update

        if regularize:
            optimized = np.copy(_init_emb_mat)
            for i in range(len(_optimized)):
                optimized[i] = _optimized[i] / np.linalg.norm(_optimized[i])
        else:
            optimized = _optimized

        trained_emb = {}
        for word in _word2idx:
            trained_emb[word] = optimized[_word2idx[word]]

        if self.verbose:
            print('--After Optimization:')
            evaluate(trained_emb,
                     data_container.lexicon)

        return trained_emb

    def _single_optimize(self, graph, alpha=0.5, regularize=False):
        _word2idx, _idx2word, _adjacency, _ = graph.graph
        _emb_shape = graph.data_container.emb_shape
        _vocab_size = len(_word2idx)
        _init_emb = graph.data_container.init_emb
        _init_emb_mat = self.build_init_emb_mat(shape=(_vocab_size, _emb_shape[0]),
                                                word2idx=_word2idx, init_emb=_init_emb)
        if self.verbose:
            print('--Before Optimization:')
            evaluate(graph.data_container.init_emb,
                     graph.data_container.lexicon)

        print('Starting Retrofit optimization with **alpha =', alpha, '** ...')

        _num_iters = 15
        _optimized = np.copy(_init_emb_mat)
        _update = np.copy(_init_emb_mat)

        for it in range(_num_iters):
            for i in range(len(_optimized)):
                _temp = alpha * np.dot(_adjacency[i], _optimized) \
                       + (1 - alpha) * _init_emb_mat[i]

                _update[i] = _temp / ((alpha * np.sum(_adjacency[i])) + (1 - alpha))

            _optimized = _update

        if regularize:
            optimized = np.copy(_init_emb_mat)
            for i in range(len(_optimized)):
                optimized[i] = _optimized[i] / np.linalg.norm(_optimized[i])
        else:
            optimized = _optimized

        trained_emb = {}
        for word in _word2idx:
            trained_emb[word] = optimized[_word2idx[word]]

        if self.verbose:
            print('--After Optimization:')
            evaluate(trained_emb,
                     graph.data_container.lexicon)

        return trained_emb
