import numpy as np
# Get the interactive Tools for Matplotlib
#%matplotlib notebook
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
from numpy.linalg import norm
#from sklearn.decomposition import PCA
#from gensim.test.utils import datapath, get_tmpfile
#from gensim.models import KeyedVectors
#from gensim.scripts.glove2word2vec import glove2word2vec
#from gensim.models.word2vec import Word2Vec

def cos_sim(a, b):
    """
    compute the cosine similarity between a and b
    :params[in]: a, b, vectors
    
    :params[out]: res, real number
    """
    res = np.dot(a, b)/(norm(a)*norm(b))
    return res


def coherence(keywords, emb_model):
    """
    compute the coherence score of a list of keywords
    
    :params[in]: keywords, a list of keywords
    :params[in]: emb_model, a word embedding model
    
    :params[out]: coherence score, real
    """
    sim = []     ## cosine similarity
    ## check whether a keyword is in the dictionary
    clean = []  ## keywords in dictionary
    for it in keywords:
        if emb_model.__contains__(it):
            clean.append(it)
    lens = len(clean)  ## total length
    for i in range(lens):
        cur_vec = emb_model[clean[i]]
        for sec in range(i, lens):
            tmp_vec = emb_model[clean[sec]]
            sim.append(cos_sim(cur_vec, tmp_vec))
    ## compute the average
    avg = np.mean(sim)
    return avg #, sim
