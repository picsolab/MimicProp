from model.graph.basic_graph import BasicGraph
import numpy as np
import random
import copy
from sklearn.neighbors import NearestNeighbors


class LexiconGraph(BasicGraph):
    def __init__(self, data_container,
                 name, m, signed=True,
                 seed=101, verbose=True):
        self.signed = signed
        self.m = m
        BasicGraph.__init__(self, data_container=data_container, name=name,
                            seed=seed, verbose=verbose)

    def build_edges_binary(self):
        '''
        build edges by random sampling
        :return:
        '''

        random.seed(self.seed)
        edges = {}
        if self.signed:
            print('Building **signed** edges from **binary** lexicon ...')
            m = self.m

            for word in self.data_container.lexicon_words:
                edges[word] = {}

            #create edges for positive words
            for word in self.data_container.pos_words:
                _sampled_words = random.sample(set(self.data_container.pos_words) - {word}, m) \
                                 + random.sample(set(self.data_container.neg_words) - {word}, m)
                for s_word in _sampled_words:
                    _score_word = self.data_container.lexicon[word]
                    _score_s_word = self.data_container.lexicon[s_word]
                    _edge_weight = self.calculate_score(_score_word, _score_s_word)
                    edges[word][s_word] = _edge_weight
                    edges[s_word][word] = _edge_weight
            #create edges for negative words
            for word in self.data_container.neg_words:
                _sampled_words = random.sample(set(self.data_container.pos_words) - {word}, m) \
                                 + random.sample(set(self.data_container.neg_words) - {word}, m)
                for s_word in _sampled_words:
                    _score_word = self.data_container.lexicon[word]
                    _score_s_word = self.data_container.lexicon[s_word]
                    _edge_weight = self.calculate_score(_score_word, _score_s_word)
                    edges[word][s_word] = _edge_weight
                    edges[s_word][word] = _edge_weight

        else:
            print('Building **un-signed** from **binary** lexicon ...')
            m = self.m * 2 #double the number of m since we are not creating negative edges

            for word in self.data_container.lexicon_words:
                edges[word] = {}
            #for positive words
            for word in self.data_container.pos_words:
                _sampled_words = random.sample(set(self.data_container.pos_words) - {word}, m)
                for s_word in _sampled_words:
                    _score_word = self.data_container.lexicon[word]
                    _score_s_word = self.data_container.lexicon[s_word]
                    _edge_weight = self.calculate_score(_score_word, _score_s_word)
                    edges[word][s_word] = _edge_weight
                    edges[s_word][word] = _edge_weight
            #for negative words
            for word in self.data_container.neg_words:
                _sampled_words = random.sample(set(self.data_container.neg_words) - {word}, m)
                for s_word in _sampled_words:
                    _score_word = self.data_container.lexicon[word]
                    _score_s_word = self.data_container.lexicon[s_word]
                    _edge_weight = self.calculate_score(_score_word, _score_s_word)
                    edges[word][s_word] = _edge_weight
                    edges[s_word][word] = _edge_weight
        return edges

    def build_edges(self):
        '''
        build edges by score constraints and opposite random sampling
        :return:
        '''
        random.seed(self.seed)
        np.random.seed(self.seed)
        edges = {}
        if self.signed:
            print('Building **signed** edges from **non-binary** lexicon ...')
            m = self.m
            for word in self.data_container.lexicon_words:
                edges[word] = {}

            for word in self.data_container.pos_words:
                _sampled_words = random.sample(set(self.data_container.neg_words) - {word}, m)
                for s_word in _sampled_words:
                    _score_word = self.data_container.lexicon[word]
                    _score_s_word = self.data_container.lexicon[s_word]
                    _edge_weight = self.calculate_score(_score_word, _score_s_word)
                    edges[word][s_word] = _edge_weight
                    edges[s_word][word] = _edge_weight

            for word in self.data_container.neg_words:
                _sampled_words = random.sample(set(self.data_container.pos_words) - {word}, m)
                for s_word in _sampled_words:
                    _score_word = self.data_container.lexicon[word]
                    _score_s_word = self.data_container.lexicon[s_word]
                    _edge_weight = self.calculate_score(_score_word, _score_s_word)
                    edges[word][s_word] = _edge_weight
                    edges[s_word][word] = _edge_weight

            sorted_words = sorted(self.data_container.lexicon, key=self.data_container.lexicon.get)

            for i in range(len(sorted_words)):
                word = sorted_words[i]
                _score_word = self.data_container.lexicon[word]
                for j in range(i + 1, min(i + 1 + m, len(sorted_words))):
                    s_word = sorted_words[j]
                    _score_s_word = self.data_container.lexicon[s_word]
                    _edge_weight = self.calculate_score(_score_word, _score_s_word)
                    edges[word][s_word] = _edge_weight
                    edges[s_word][word] = _edge_weight

        else:
            print('Building **un-signed** edges from **non-binary** lexicon ...')
            m = 2 * self.m
            for word in self.data_container.lexicon_words:
                edges[word] = {}

            _pos_lex = {}
            for word in self.data_container.pos_words:
                _pos_lex[word] = self.data_container.lexicon[word]

            _neg_lex = {}
            for word in self.data_container.neg_words:
                _neg_lex[word] = self.data_container.lexicon[word]

            sorted_words_pos = sorted(_pos_lex, key=_pos_lex.get)
            sorted_words_neg = sorted(_neg_lex, key=_neg_lex.get)

            for i in range(len(sorted_words_pos)):
                word = sorted_words_pos[i]
                _score_word = self.data_container.lexicon[word]
                for j in range(i + 1, min(i + 1 + m, len(sorted_words_pos))):
                    s_word = sorted_words_pos[j]
                    _score_s_word = self.data_container.lexicon[s_word]
                    _edge_weight = self.calculate_score(_score_word, _score_s_word)
                    edges[word][s_word] = _edge_weight
                    edges[s_word][word] = _edge_weight

            for i in range(len(sorted_words_neg)):
                word = sorted_words_neg[i]
                _score_word = self.data_container.lexicon[word]
                for j in range(i + 1, min(i + 1 + m, len(sorted_words_neg))):
                    s_word = sorted_words_neg[j]
                    _score_s_word = self.data_container.lexicon[s_word]
                    _edge_weight = self.calculate_score(_score_word, _score_s_word)
                    edges[word][s_word] = _edge_weight
                    edges[s_word][word] = _edge_weight
        return edges

    def build_graph(self):
        '''
        build the graph
        :return:
        graoh: a tuple of (word2idx, idx2word, adjacency matrix, set of edges)
        '''
        word2idx = {}
        idx2word = {}
        for idx, word in enumerate(self.data_container.lexicon_words):
            word2idx[word] = idx
            idx2word[idx] = word

        if self.data_container.binary:
            edges = self.build_edges_binary()
        else:
            edges = self.build_edges()

        adjacency = self.build_adjacency(edges=edges, word2idx=word2idx)

        graph = (word2idx, idx2word, adjacency, edges)
        return graph

    def graph_stats(self):
        _num_nodes = self.graph[2].shape[0]
        _degree = []
        for row in self.graph[2]:
            _degree.append(np.count_nonzero(row))
        _mean_degree = np.mean(_degree)
        _std_degree = np.std(_degree)
        _num_edges = np.sum(_degree)
        _graph_density = 2 * _num_edges / _num_nodes / (_num_nodes - 1)
        print('\n-------------------Graph Stats-------------------')
        print('- Number of Nodes:', _num_nodes)
        print('- Number of Edges:', _num_edges)
        print('- Average Degree of Nodes: ', _mean_degree)
        print('- STD Degree of Nodes: ', _std_degree)
        print('- Graph Density: ', _graph_density)
        print('-------------------------------------------------\n')


class PropagationGraphs(BasicGraph):
    def __init__(self, data_container,
                 name, init_edges, k,
                 seed=101, verbose=True):

        self.init_edges = copy.deepcopy(init_edges)
        self.edge_distribution = []
        for word in self.init_edges:
            self.edge_distribution.append(len(self.init_edges[word]))

        self.k = k
        BasicGraph.__init__(self, data_container=data_container, name=name,
                            seed=seed, verbose=verbose)

    def find_neighbors(self, unseen_words):
        print('Finding', self.k, 'neighbors...')
        _exclude = ['POSPOLE', 'NEGPOLE']

        _train_sample = [self.data_container.init_1st_emb[word]
                         for word in self.data_container.lexicon_words
                         if word not in _exclude]
        _test_sample = [self.data_container.init_1st_emb[word] for word in unseen_words]

        _nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(_train_sample)
        distances, indices = _nbrs.kneighbors(_test_sample)

        neighbors = {}
        for word in unseen_words:
            neighbors[word] = {}
            _distances = distances[unseen_words.index(word)]
            _indices = indices[unseen_words.index(word)]
            for i in range(len(_distances)):
                _distance = _distances[i]
                _nbr_word = self.data_container.lexicon_words[_indices[i]]
                neighbors[word][_nbr_word] = _distance
        return neighbors

    def draw_degree(self):
        self.seed += 1
        random.seed(self.seed)
        degree = random.sample(self.edge_distribution, 1)[0]
        return degree

    def mimic_edges(self, unseen_words):
        extended_edges = copy.deepcopy(self.init_edges)
        _all_neighbors = self.find_neighbors(unseen_words)

        import pickle
        pickle.dump(_all_neighbors, open('/home/muheng/tolocal/neighbors.pkl', 'wb'), protocol=2)

        for word in unseen_words:
            _neighbors = _all_neighbors[word]

            _outward_edges = {}
            _degree = self.draw_degree()

            for neighbor in _neighbors:
                _penalty = np.exp(-_neighbors[neighbor])
                _candidate_edges = copy.deepcopy(self.init_edges[neighbor])
                for candidate in _candidate_edges:
                    try:
                        _outward_edges[candidate] += _candidate_edges[candidate] * _penalty
                    except Exception:
                        _outward_edges[candidate] = _candidate_edges[candidate] * _penalty

            _sorted_outward_word = sorted(_outward_edges,
                                          key=_outward_edges.get,
                                          reverse=True)
            extended_edges[word] = {}

            for outward_word in _sorted_outward_word[:_degree]:
                extended_edges[word][outward_word] = _outward_edges[outward_word] / self.k
                extended_edges[outward_word][word] = _outward_edges[outward_word] / self.k
            for outward_word in _sorted_outward_word[-_degree:]:
                extended_edges[word][outward_word] = _outward_edges[outward_word] / self.k
                extended_edges[outward_word][word] = _outward_edges[outward_word] / self.k

        return extended_edges

    def build_graph(self):
        graph = []
        for bthNo, non_lexicon_batch in enumerate(self.data_container.non_lexicon_words):
            print('---------Building Sub-Graph:', bthNo, '------------')
            _word2idx = {}
            _idx2word = {}
            _edges = None
            _adjacency = None

            _full_set = self.data_container.lexicon_words + non_lexicon_batch
            for idx, word in enumerate(_full_set):
                _word2idx[word] = idx
                _idx2word[idx] = word
            try:
                print('Generating edges for Sub-Graph', bthNo)
                _edges = self.mimic_edges(non_lexicon_batch)

                print('Creating adjacency matrix for Sub-Graph', bthNo)
                _adjacency = self.build_adjacency(edges=_edges, word2idx=_word2idx)

                _graph = (_word2idx, _idx2word, _adjacency, _edges)

                graph.append(_graph)
            except Exception:
                pass
            print('----------------------------------------\n')

        return graph

    def graph_stats(self):
        for idx, sub_graph in enumerate(self.graph):
            _num_nodes = sub_graph[2].shape[0]
            _degree = []
            for row in sub_graph[2]:
                _degree.append(np.count_nonzero(row))
            _mean_degree = np.mean(_degree)
            _std_degree = np.std(_degree)
            _num_edges = np.sum(_degree)
            _graph_density = 2 * _num_edges / _num_nodes / (_num_nodes - 1)
            print('\n-------------------Graph Stats: Sub-Graph', idx+1, '-------------------')
            print('- Number of Nodes:', _num_nodes)
            print('- Number of Edges:', _num_edges)
            print('- Average Degree of Nodes: ', _mean_degree)
            print('- STD Degree of Nodes: ', _std_degree)
            print('- Graph Density: ', _graph_density)
            print('-------------------------------------------------\n')
        return


class NaivePropagationGraphs(PropagationGraphs):

    def find_300_neighbors(self, unseen_words):
        print('Finding', 300, 'neighbors...')
        _exclude = ['POSPOLE', 'NEGPOLE']

        _train_sample = [self.data_container.init_1st_emb[word]
                         for word in self.data_container.lexicon_words
                         if word not in _exclude]
        _test_sample = [self.data_container.init_1st_emb[word] for word in unseen_words]

        _nbrs = NearestNeighbors(n_neighbors=300, algorithm='ball_tree').fit(_train_sample)
        distances, indices = _nbrs.kneighbors(_test_sample)

        neighbors = {}
        for word in unseen_words:
            neighbors[word] = {}
            _distances = distances[unseen_words.index(word)]
            _indices = indices[unseen_words.index(word)]
            for i in range(len(_distances)):
                _distance = _distances[i]
                _nbr_word = self.data_container.lexicon_words[_indices[i]]
                neighbors[word][_nbr_word] = _distance
        return neighbors

    def naive_edges(self, unseen_words):
        extended_edges = copy.deepcopy(self.init_edges)
        _all_neighbors = self.find_300_neighbors(unseen_words)
        for word in unseen_words:
            _neighbors = _all_neighbors[word]
            # _neighbors: dict{'word': distance}
            _sorted_neighbors = sorted(_neighbors,
                                       key=_neighbors.get,
                                       reverse=False)

            _degree = self.draw_degree()
            extended_edges[word] = {}
            for neighbor in _sorted_neighbors[:_degree]:
                _penalty = np.exp(-_neighbors[neighbor])
                extended_edges[word][neighbor] = 1 * _penalty
                extended_edges[neighbor][word] = 1 * _penalty
        return extended_edges

    def build_graph(self):
        graph = []
        for bthNo, non_lexicon_batch in enumerate(self.data_container.non_lexicon_words):
            print('---------Building Sub-Graph:', bthNo, '------------')
            _word2idx = {}
            _idx2word = {}
            _edges = None
            _adjacency = None

            _full_set = self.data_container.lexicon_words + non_lexicon_batch
            for idx, word in enumerate(_full_set):
                _word2idx[word] = idx
                _idx2word[idx] = word
            try:
                print('Generating edges for Sub-Graph', bthNo)
                _edges = self.naive_edges(non_lexicon_batch)

                print('Creating adjacency matrix for Sub-Graph', bthNo)
                _adjacency = self.build_adjacency(edges=_edges, word2idx=_word2idx)

                _graph = (_word2idx, _idx2word, _adjacency, _edges)

                graph.append(_graph)
            except Exception:
                pass
            print('----------------------------------------\n')
        return graph
