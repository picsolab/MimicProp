import pickle
import os
import random
import numpy as np
from config import BASE_PATH


class DataContainer:
    def __init__(self, pretrained_emb_path,
                 lexicon_path, corpus_vocab_path, down_sampling=False,
                 seed=101, name=None, verbose=True):
        if down_sampling:
            name = '{}_balanced'.format(name)
        self.name = name
        self.seed = seed
        self.verbose = verbose
        self.down_sampling = down_sampling

        data = self.load_from_cache(name=name, pretrained_emb_path=pretrained_emb_path,
                                    lexicon_path=lexicon_path, corpus_vocab_path=corpus_vocab_path)

        self.lexicon_words, self.non_lexicon_words, self.lexicon, self.init_emb = data
        _vec = next(iter(self.init_emb.values()))
        self.emb_shape = _vec.shape

        self.pos_words, self.neg_words = self.init_poles()

        _scores = [self.lexicon[key] for key in self.lexicon]
        if len(set(_scores)) == 2:
            self.binary = True
        else:
            self.binary = False

        if self.verbose:
            print('Num of lexicon words:', len(self.lexicon_words))
            print('Class Distribution of the lexicon words:')
            print('Positive:', len(self.pos_words), 'Negative:', len(self.neg_words),
                  'Neutral:', len(self.lexicon_words) - len(self.pos_words) - len(self.neg_words))

            _len_nl = 0
            for _bth in self.non_lexicon_words:
                _len_nl += len(_bth)
            print('Num of non-lexicon words:', _len_nl)

            print('The shape of the word representations is:', self.emb_shape, '!')

            if self.binary:
                print('***The lexicon is binary!***')
            else:
                print('***The lexicon is not binary!***')

        # self.init_2nd_emb = None
        self.init_1st_emb = dict(self.init_emb)

    def update_emb(self, embeddings):
        for word in embeddings:
            self.init_emb[word] = embeddings[word]
        # self.init_2nd_emb = dict(self.init_emb)

    def rescale(self, num, _min, _max):
        random.seed(self.seed)
        res = (num - _min) * 2 / (_max - _min) - 1
        if res == 0:
            res += random.choice([0.000001, -0.000001])
            self.seed += 1
        return res

    @staticmethod
    def batchify(words, batch_size):
        assert batch_size > 1
        batches = []
        batch = []
        count = 0
        for word in words:
            batch.append(word)
            count += 1
            if count > batch_size:
                batches.append(batch)
                batch = []
                count = 0
        batches.append(batch)
        return batches

    def preprocess(self, pretrained_emb_path, lexicon_path, corpus_vocab_path):
        '''
        cross compare vocabularies and initialize lexicon words, non lexicon words;
        rescale the lexicon scores to [-1, 1]
        cache the preprocessed data if wanted

        :param pretrained_emb_path:
        :param lexicon_path:
        :param corpus_vocab_path:
        :return:
        '''
        pretrained_emb = pickle.load(open(pretrained_emb_path, 'rb'))
        lexicon = pickle.load(open(lexicon_path, 'rb'))
        corpus_vocab = pickle.load(open(corpus_vocab_path, 'rb'))

        inter_corpus_vocab = list(set(pretrained_emb.keys()) & set(corpus_vocab.keys()))

        lexicon_words = list(set(pretrained_emb.keys()) & set(lexicon.keys()))
        non_lexicon_words = list(set(inter_corpus_vocab) - set(lexicon_words))

        random.seed(self.seed)
        random.shuffle(lexicon_words)
        random.shuffle(non_lexicon_words)

        _scores = [lexicon[key] for key in lexicon]
        _min = np.floor(min(_scores))
        _max = np.ceil(max(_scores))

        scaled_lexicon = {}
        for word in lexicon_words:
            scaled_lexicon[word] = self.rescale(lexicon[word], _min, _max)

        # down sampling
        if self.down_sampling:
            print('Down Sampling...')
            _pos_word = []
            _neg_word = []
            _neutral_word = []
            for word in lexicon_words:
                if scaled_lexicon[word] > 0.001:
                    _pos_word.append(word)
                elif scaled_lexicon[word] < -0.001:
                    _neg_word.append(word)
                else:
                    _neutral_word.append(word)
            _maxLen = min(len(_pos_word), len(_neg_word))

            _down_sampled = _pos_word[:_maxLen] + _neg_word[:_maxLen] + _neutral_word
            lexicon_words = _down_sampled

        full_vocab = list(set(lexicon_words) | set(non_lexicon_words))
        init_emb = {}
        for word in full_vocab:
            init_emb[word] = pretrained_emb[word]

        batch_non_lexicon_words = self.batchify(non_lexicon_words, len(lexicon_words) / 9)

        data = (lexicon_words, batch_non_lexicon_words, scaled_lexicon, init_emb)
        return data

    def init_poles(self):
        '''
        initialize the poles and save to the data container

        :return:
        '''
        pos_words = []
        neg_words = []
        _agg_pos_emb = np.zeros(shape=self.emb_shape)
        _agg_neg_emb = np.zeros(shape=self.emb_shape)
        _num_pos = 0
        _num_neg = 0

        for word in self.lexicon_words:
            if self.lexicon[word] > 0.001:
                pos_words.append(word)
                _agg_pos_emb += self.init_emb[word]
                _num_pos += 1
            elif self.lexicon[word] < -0.001:
                neg_words.append(word)
                _agg_neg_emb += self.init_emb[word]
                _num_neg += 1

        assert _num_pos > 0
        assert _num_neg > 0
        _pos_pole = _agg_pos_emb / _num_pos
        _neg_pole = _agg_neg_emb / _num_neg
        self.init_emb['POSPOLE'] = _pos_pole
        self.init_emb['NEGPOLE'] = _neg_pole
        self.lexicon['POSPOLE'] = 1
        self.lexicon['NEGPOLE'] = -1
        self.lexicon_words.append('POSPOLE')
        self.lexicon_words.append('NEGPOLE')
        pos_words.append('POSPOLE')
        neg_words.append('NEGPOLE')

        return pos_words, neg_words

    @staticmethod
    def _check_cache():
        cache_dir = os.path.join(BASE_PATH, "_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_filename(self):
        return os.path.join(BASE_PATH, "_cache",
                            "_cachedData_{}.pkl".format(self.name))

    def _write_cache(self, data):
        self._check_cache()

        cache_file = self._get_cache_filename()

        with open(cache_file, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    def load_from_cache(self, name, pretrained_emb_path,
                        lexicon_path, corpus_vocab_path):

        # NOT using cache
        if name is None:
            print("cache deactivated for data loading!")
            return self.preprocess(pretrained_emb_path=pretrained_emb_path,
                                   lexicon_path=lexicon_path,
                                   corpus_vocab_path=corpus_vocab_path)

        # using cache
        cache_file = self._get_cache_filename()

        if os.path.exists(cache_file):
            print("Loading {} from cache ...".format(self.name))
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("No cache file for {} ...".format(self.name))
            data = self.preprocess(pretrained_emb_path=pretrained_emb_path,
                                   lexicon_path=lexicon_path,
                                   corpus_vocab_path=corpus_vocab_path)
            self._write_cache(data)
            return data
