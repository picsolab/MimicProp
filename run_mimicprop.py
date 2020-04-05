from utils.data_loading import DataContainer
from model.graph.graphs import LexiconGraph, PropagationGraphs, NaivePropagationGraphs
from model.optimizer.optimizers import LabelSpreadingOptimizer, RetrofitOptimizer
from utils.evaluate import evaluate, outputScore
import pickle
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='Inputs for MimicProp.')

parser.add_argument('--m', help='the number of connected nodes for lexicon words', default='20')
parser.add_argument('--alpha', help='the hyper-parameter controlling the balancing between lexicon/semantic in learning', default='0.55')
parser.add_argument('--k', help='the number of nodes to mimic', default='5')
parser.add_argument('--pre', help='location of the pretrained embedding', default='data/pretrained/bert_pretrained.pkl')
parser.add_argument('--lex', help='location of the lexicon', default='data/lexicon/bert_liwc.pkl')
parser.add_argument('--voc', help='location of the vocabulary desired', default='data/vocabulary/bert_vocab.pkl')
parser.add_argument('--name', help='name of the run', default='Untitled')


seed = 42


args = parser.parse_args()
m = int(args.m)
alpha = float(args.alpha)
k = int(args.k)

_name = args.name
pre_emb_path = args.pre
lex_path = args.lex
corpus_vocab_path = args.voc


mimic = True #MimicProp
signed = True #signed graph
regularize = False #regularize at the end
down_sampling = False #down sampling lexicon words
verbose = True

if mimic:
    _mmc = 'mimic'
else:
    _mmc = 'non_mimic'
if signed:
    _sgned = 'signed'
else:
    _sgned = 'unsigned'
exp_name = '{}-{}-{}'.format(_name, _mmc, _sgned)


if __name__ == "__main__":
    ''' 
    'data/pretrained.pkl': a dict, {word:embedding(np array)}
    'data/liwc_twt_lex.pkl': a dict, {word:score}
    'data/toy_corpus.pkl': a dict, {word:frequency(or 1, frequency not used for now)}
    '''

    #load data
    data_container = DataContainer(pretrained_emb_path=pre_emb_path, lexicon_path=lex_path,
                                   corpus_vocab_path=corpus_vocab_path, down_sampling=down_sampling,
                                   name='{}-Data'.format(exp_name), seed=seed, verbose=verbose)
    #build lexicon graph
    lexicon_graph = LexiconGraph(data_container=data_container,
                                 name='{}-LexiconGraph'.format(exp_name),
                                 m=m, signed=signed,
                                 verbose=verbose, seed=seed)
    #initialize optimizer
    if signed:
        ls_optimizer = LabelSpreadingOptimizer(verbose=verbose, seed=seed)
    else:
        ls_optimizer = RetrofitOptimizer(verbose=verbose, seed=seed)

    #optimize lexicon graph
    opt_lex_emb = ls_optimizer.optimize(graph=lexicon_graph,
                                        name='{}-lex_emb'.format(exp_name),
                                        alpha=alpha, regularize=regularize)

    #update embeddings after 1st stage
    data_container.update_emb(opt_lex_emb)

    #build propagation graph
    init_edges = dict(lexicon_graph.graph[3]) #cgopy existing graph for lexicon words

    if mimic:
        propagation_graph = PropagationGraphs(data_container=data_container,
                                              name='{}-MimicPropGraph'.format(exp_name),
                                              init_edges=init_edges, k=k,
                                              seed=seed, verbose=verbose)
    else:
        propagation_graph = NaivePropagationGraphs(data_container=data_container,
                                              name='{}-NaivePropGraph'.format(exp_name),
                                              init_edges=init_edges, k=k,
                                              seed=seed, verbose=verbose)
    
    print('\nScores among all nodes:')
    final_emb = ls_optimizer.optimize(graph=propagation_graph,
                                      name='{}-final_emb'.format(exp_name),
                                      alpha=alpha, regularize=regularize)
    
    print('finished!')