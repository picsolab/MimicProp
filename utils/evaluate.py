import numpy as np
from scipy import stats
from scipy import spatial
from sklearn import metrics
import pickle

def findeBasis(embeddings):
    return (embeddings['POSPOLE'] - embeddings['NEGPOLE']) / 2

def findOrigin(embeddings):
    return (embeddings['POSPOLE'] + embeddings['NEGPOLE']) / 2

def calScore(embedding, basis, origin):
    _vec = embedding - origin
    _projection = (np.dot(_vec, basis) * basis) / \
                  (np.linalg.norm(basis) * np.linalg.norm(basis))

    if np.dot(_projection, basis) < 0:
        score = -np.linalg.norm(_projection) / np.linalg.norm(basis)
    else:
        score = np.linalg.norm(_projection) / np.linalg.norm(basis)

    return score

def calDDR(embedding, pospole, negpole):
    _pos_score = 1 - spatial.distance.cosine(embedding, pospole)
    _neg_score = 1 - spatial.distance.cosine(embedding, negpole)
    score = _pos_score / (_pos_score + _neg_score)

    return score

def calDDR1(embedding, basis, origin):
    score = 1 - spatial.distance.cosine(embedding-origin, basis)
    return score

def calDDR2(embedding, pospole):
    _pos_score = 1 - spatial.distance.cosine(embedding, pospole)
    # _neg_score = 1 - spatial.distance.cosine(embedding, negpole)
    return _pos_score


def outputScore(embeddings):
    basis = findeBasis(embeddings)
    origin = findOrigin(embeddings)
    exclude = ['POSPOLE', 'NEGPOLE']

    #original
    s_pred = [[word, calScore(embeddings[word], basis, origin)]
              for word in embeddings if word not in exclude]


    return s_pred


def outputScore_ddr(embeddings):
    basis = findeBasis(embeddings)
    origin = findOrigin(embeddings)
    exclude = ['POSPOLE', 'NEGPOLE']

    #original
    s_pred = [[word, calDDR(embeddings[word], embeddings['POSPOLE'], embeddings['NEGPOLE'])]
              for word in embeddings if word not in exclude]


    return s_pred


def evaluate(embeddings, lexicon):
    basis = findeBasis(embeddings)
    origin = findOrigin(embeddings)
    exclude = ['POSPOLE', 'NEGPOLE']

    #original
    s_pred = [calScore(embeddings[word], basis, origin)
              for word in lexicon if word not in exclude]

    #ddr
    # s_pred = [calDDR(embeddings[word], embeddings['POSPOLE'], embeddings['NEGPOLE'])
    #           for word in lexicon if word not in exclude]

    #ddr1
    # s_pred = [calDDR1(embeddings[word], basis, origin)
    #           for word in lexicon if word not in exclude]

    #ddr2
    # s_pred = [calDDR2(embeddings[word], embeddings['NEGPOLE'])
    #           for word in lexicon if word not in exclude]

    s_true = [lexicon[word] for word in lexicon if word not in exclude]


    tau, p_value = stats.kendalltau(s_true, s_pred)
    corr = np.corrcoef(s_true, s_pred)[0][1]

    pickle.dump((s_true, s_pred), open('_cache/checkpoint.pkl', 'wb'), protocol=2)

    zeroed_true = []
    for s in s_true:
        if s >= 0:
            zeroed_true.append(1)
        else:
            zeroed_true.append(0)

    max_F1 = 0
    best_THR = 0

    for thrshd in np.arange(min(s_pred), max(s_pred), 0.01):
        rounded = []
        for s in s_pred:
            if s >= thrshd:
                rounded.append(1)
            else:
                rounded.append(0)

        f1 = metrics.f1_score(zeroed_true, rounded)

        if f1 > max_F1:
            max_F1 = f1
            best_THR = thrshd
    print('\n-------------------Evaluation Results: -------------------')
    print('- Best threshold:', best_THR)

    rounded = []
    for s in s_pred:
        if s >= best_THR:
            rounded.append(1)
        else:
            rounded.append(0)


    fpr, tpr, thresholds = metrics.roc_curve(zeroed_true, rounded)
    auc = metrics.auc(fpr, tpr)
    confusion = metrics.confusion_matrix(zeroed_true, rounded)
    precision = metrics.precision_score(zeroed_true, rounded)
    recall = metrics.recall_score(zeroed_true, rounded)
    f1 = metrics.f1_score(zeroed_true, rounded, average='macro')

    # print("- corr_coef: %2f, tau: %2F" % (corr, tau))
    # print('pred positives:', np.sum(rounded), '/', len(rounded), 'true 1s:', np.sum(zeroed_true), '/', len(zeroed_true))
    # print('- auc:', auc, '\n- p:', precision, '\n- r', recall, '\n- f1', f1)
    print(corr, '&', f1)
    print('---------------------------------------------------------\n')
    # return np.sum(rounded), len(rounded), np.sum(zeroed), len(zeroed), corr, tau, precision, recall, f1, auc
    return s_true, s_pred, zeroed_true, rounded


def cal_wordScore(embeddings):
    basis = findeBasis(embeddings)
    origin = findOrigin(embeddings)
    exclude = ['POSPOLE', 'NEGPOLE']

    # original
    s_pred = {}
    for word in embeddings:
        if word not in exclude:
            s_pred[word] = calScore(embeddings[word], basis, origin)
    return s_pred