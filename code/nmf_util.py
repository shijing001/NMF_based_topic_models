import numpy as np
#import torch
import time
from numpy.linalg import norm

def random_initialization(A, rank):
    """
    use uniform distribution to initialize factor matrices
    
    :params[in]: A, token-document matrix
    :params[in]: rank, factorization rank
    
    :params[out]: W, token topic matrix
    :params[out]: H, topic document matrix
    """
    number_of_documents = A.shape[0]
    number_of_terms = A.shape[1]
    W = np.random.uniform(1,2,(number_of_documents,rank))
    H = np.random.uniform(1,2,(rank,number_of_terms))
    return W,H
                          

def nndsvd_initialization(A, rank):
    """
    use singular value decomposition method to initialize factor matrices
    
    :params[in]: A, token-document matrix
    :params[in]: rank, factorization rank
    
    :params[out]: W, token topic matrix
    :params[out]: H, topic document matrix
    """
    u,s,v=np.linalg.svd(A,full_matrices=False)
    v=v.T
    w=np.zeros((A.shape[0],rank))
    h=np.zeros((rank,A.shape[1]))

    w[:,0]=np.sqrt(s[0])*np.abs(u[:,0])
    h[0,:]=np.sqrt(s[0])*np.abs(v[:,0].T)

    for i in range(1,rank):
        
        ui=u[:,i]
        vi=v[:,i]
        ui_pos=(ui>=0)*ui
        ui_neg=(ui<0)*-ui
        vi_pos=(vi>=0)*vi
        vi_neg=(vi<0)*-vi
        
        ui_pos_norm=np.linalg.norm(ui_pos,2)
        ui_neg_norm=np.linalg.norm(ui_neg,2)
        vi_pos_norm=np.linalg.norm(vi_pos,2)
        vi_neg_norm=np.linalg.norm(vi_neg,2)
        
        norm_pos=ui_pos_norm*vi_pos_norm
        norm_neg=ui_neg_norm*vi_neg_norm
        
        if norm_pos>=norm_neg:
            w[:,i]=np.sqrt(s[i]*norm_pos)/ui_pos_norm*ui_pos
            h[i,:]=np.sqrt(s[i]*norm_pos)/vi_pos_norm*vi_pos.T
        else:
            w[:,i]=np.sqrt(s[i]*norm_neg)/ui_neg_norm*ui_neg
            h[i,:]=np.sqrt(s[i]*norm_neg)/vi_neg_norm*vi_neg.T

    return w,h

def gaussian_method(A, k, max_iter, init_mode='nndsvd', W_init=None, H_init=None):
    """
    multiplicative approach for Gaussian NMF (squared L2 loss) 
    
    :params[in]: A, token-document matrix
    :params[in]: k, factorization rank
    :params[in]: max_iter, maximum iteration
    :params[in]: init_mode, how to initialize
    :params[in]: W_init, H_init, initial matrices
    
    :params[out]: W, token topic matrix
    :params[out]: H, topic document matrix
    """
    
    if init_mode == 'random':
        W, H = random_initialization(A, k)
    elif init_mode == 'nndsvd':
        W, H = nndsvd_initialization(A,k) 
    else:  ## other case --- initialized matrices must be provided
        W, H = W_init, H_init
    
    ## errors
    norms = []
    e = 1.0e-10
    for n in range(max_iter):
        # Update H
        W_TA = W.T@A
        W_TWH = W.T@W@H+e
        H *= W_TA / W_TWH
        
        # Update W
        AH_T = A@H.T
        WHH_T =  W@H@H.T+ e
        W *= AH_T / WHH_T

        norm = np.linalg.norm(A - W@H, 'fro')/np.sqrt(np.prod(A.shape))
        norms.append(norm)
    return W, H, norms[-1]

## split an iterable of items into batches
def chunks(ls, batch_size):
    """
    Yield successive n-sized chunks from l.

    :params[in]: ls, an iterable of items
    :params[in]: batch_size, an integer, batch size

    returns a generator
    """
    for i in range(0, len(ls), batch_size):
        yield ls[i:i + batch_size]

def fed_gaussian_nmf(ls_matrices, epochs, perc=0., batch_size=4, num_topics=10, max_iter=10):
    """
    federated Gaussian NMF
    
    :params[in]: ls_matrices, list of matrices for factorization, where
          each element is a matrix of size: number_tokens * number_documents
    :params[in]: epochs, the number of epochs
    :params[in]: perc, the percent of previous weight to keep
    :params[in]: batch_size, integer, the batch size
    
    """
    ls_nums = [it.shape[1] for it in ls_matrices] ## list of number of documents on each client
    train_indexes = list(range(len(ls_nums)))    # indices of datasets
    token_num = ls_matrices[0].shape[0]          ## number of tokens
    W_mats = {i:None for i in range(len(ls_nums))} ## token topic weights
    H_mats = {i:None for i in range(len(ls_nums))} ## topic document weights
    Ab_errors = {i:0. for i in range(len(ls_nums))} ## absolute errors
    ## we update federated weight once over each batch
    fed_W = np.zeros((token_num, num_topics)) ## federated average weight
    
    ## loop over epochs
    for ep in range(epochs):
        batches = chunks(train_indexes, batch_size)  ## split data into batches
        ## loop over batches
        for batch in batches:
            ## we update federated weight once over each batch
            tmp_W = np.zeros((token_num, num_topics)) ## set Federated weight to zero
            ## current matrices/datasets
            cur_data = [ls_matrices[i] for i in batch]
            ## data size of matrices on clients
            cur_nums = [ls_nums[i] for i in batch]
            ## percents
            pts = np.array(cur_nums)/np.sum(cur_nums)
            ## pdb.set_trace()
            ## loop over each client in a batch
            for i,cl in enumerate(batch):
                W_mats[cl] = fed_W     ## copy latest weight matrix from the server
                if H_mats[cl] is None:
                    cl_res = gaussian_method(ls_matrices[cl], k=num_topics, max_iter=max_iter, 
                                   init_mode='nndsvd', W_init=W_mats[cl], H_init=H_mats[cl])
                else:
                    cl_res = gaussian_method(ls_matrices[cl], k=num_topics, max_iter=max_iter, 
                                   init_mode='', W_init=W_mats[cl], H_init=H_mats[cl])
                ## update factor matrices for each client    
                W_mats[cl], H_mats[cl], Ab_errors[cl] = cl_res
                ## federated token-topic weight
                tmp_W += pts[i]*W_mats[cl]
            ## update overall weight matrix
            fed_W =perc*fed_W +(1. - perc)*tmp_W
        ## compute average loss
        print('Absolute errors at Epoch ', ep, Ab_errors)
        print('Absolute error at Epoch ',ep, ' is:',sum(Ab_errors.values()))
        np.random.shuffle(train_indexes)     # shuffle indices of examples for next epoch
    ## finished training
    return fed_W, W_mats, H_mats

def poisson_method(A, k, max_iter, init_mode='nndsvd', W_init=None, H_init=None):
    """
    multiplicative approach for Poisson NMF 

    :params[in]: A, token-document matrix
    :params[in]: k, factorization rank
    :params[in]: max_iter, maximum iteration
    :params[in]: init_mode, how to initialize
    :params[in]: W_init, H_init, initial matrices

    :params[out]: W, token topic matrix
    :params[out]: H, topic document matrix
    """

    if init_mode == 'random':
        W, H = random_initialization(A, k)
    elif init_mode == 'nndsvd':
        W, H = nndsvd_initialization(A,k)
    else:  ## other case --- initialized matrices must be provided
        W, H = W_init, H_init

    ## errors
    norms = []
    e = 1.0e-10
    for n in range(max_iter):
        # Update H
        WH= W@H
        R=A/(WH + e)
        num=np.array(A)
        m,n = R.shape
        E=np.ones((m,n))
        W_TE = W.T@E+e
        W_TR=W.T@R
        H *= W_TR / W_TE

        # Update W
        RH_T = R@H.T
        EH_T =  E@H.T+ e
        W *= RH_T / EH_T

        norm = np.linalg.norm(A - W@H, 'fro')/np.sqrt(np.prod(A.shape))
        norms.append(norm)
    return W, H, norms[-1]


def fed_poisson_nmf(ls_matrices, epochs, perc=0., batch_size=4, num_topics=10, max_iter=10):
    """
    federated Poisson NMF
    
    :params[in]: ls_matrices, list of matrices for factorization, where
          each element is a matrix of size: number_tokens * number_documents
    :params[in]: epochs, the number of epochs
    :params[in]: perc, the percent of previous weight to keep
    :params[in]: batch_size, integer, the batch size
    
    """
    ls_nums = [it.shape[1] for it in ls_matrices] ## list of number of documents on each client
    train_indexes = list(range(len(ls_nums)))    # indices of datasets
    token_num = ls_matrices[0].shape[0]          ## number of tokens
    W_mats = {i:None for i in range(len(ls_nums))} ## token topic weights
    H_mats = {i:None for i in range(len(ls_nums))} ## topic document weights
    Ab_errors = {i:0. for i in range(len(ls_nums))} ## absolute errors
    ## we update federated weight once over each batch
    fed_W = np.zeros((token_num, num_topics)) ## federated average weight
    
    ## loop over epochs
    for ep in range(epochs):
        batches = chunks(train_indexes, batch_size)  ## split data into batches
        ## loop over batches
        for batch in batches:
            ## we update federated weight once over each batch
            tmp_W = np.zeros((token_num, num_topics)) ## set Federated weight to zero
            ## current matrices/datasets
            cur_data = [ls_matrices[i] for i in batch]
            ## data size of matrices on clients
            cur_nums = [ls_nums[i] for i in batch]
            ## percents
            pts = np.array(cur_nums)/np.sum(cur_nums)
            ## pdb.set_trace()
            ## loop over each client in a batch
            for i,cl in enumerate(batch):
                W_mats[cl] = fed_W     ## copy latest weight matrix from the server
                if H_mats[cl] is None:
                    cl_res = poisson_method(ls_matrices[cl], k=num_topics, max_iter=max_iter, 
                                   init_mode='nndsvd', W_init=W_mats[cl], H_init=H_mats[cl])
                else:
                    cl_res = poisson_method(ls_matrices[cl], k=num_topics, max_iter=max_iter, 
                                   init_mode='', W_init=W_mats[cl], H_init=H_mats[cl])
                ## update factor matrices for each client    
                W_mats[cl], H_mats[cl], Ab_errors[cl] = cl_res
                ## federated token-topic weight
                tmp_W += pts[i]*W_mats[cl]
            ## update overall weight matrix
            fed_W =perc*fed_W +(1. - perc)*tmp_W
        ## compute average loss
        print('Absolute errors at Epoch ', ep, Ab_errors)
        print('Absolute error at Epoch ',ep, ' is:',sum(Ab_errors.values()))
        np.random.shuffle(train_indexes)     # shuffle indices of examples for next epoch
    ## finished training
    return fed_W, W_mats, H_mats

def exponential_method(A, k, max_iter, init_mode='nndsvd', W_init=None, H_init=None):
    """
    multiplicative approach for Exponential NMF 

    :params[in]: A, token-document matrix
    :params[in]: k, factorization rank
    :params[in]: max_iter, maximum iteration
    :params[in]: init_mode, how to initialize
    :params[in]: W_init, H_init, initial matrices

    :params[out]: W, token topic matrix
    :params[out]: H, topic document matrix
    """

    if init_mode == 'random':
        W, H = random_initialization(A, k)
    elif init_mode == 'nndsvd':
        W, H = nndsvd_initialization(A,k)
    else:  ## other case --- initialized matrices must be provided
        W, H = W_init, H_init

    ## errors
    norms = []
    e = 1.0e-10
    for n in range(max_iter):
        # Update H
        D=W@H
        F=np.power(D, 2)
        B=A/(F+e)
        W_TB= W.T@B
        C=1/(D+e)
        W_TC = W.T@C +e
        H *= W_TB / W_TC

        # Update W
        BH_T = B@H.T
        CH_T =  C@H.T+ e
        W *= BH_T / CH_T

        norm = np.linalg.norm(A - W@H, 'fro')/np.sqrt(np.prod(A.shape))/100
        norms.append(norm)
    return W, H, norms[-1]


def fed_exponential_nmf(ls_matrices, epochs, perc=0., batch_size=4, num_topics=10, max_iter=10):
    """
    federated Gaussian NMF
    
    :params[in]: ls_matrices, list of matrices for factorization, where
          each element is a matrix of size: number_tokens * number_documents
    :params[in]: epochs, the number of epochs
    :params[in]: perc, the percent of previous weight to keep
    :params[in]: batch_size, integer, the batch size
    
    """
    ls_nums = [it.shape[1] for it in ls_matrices] ## list of number of documents on each client
    train_indexes = list(range(len(ls_nums)))    # indices of datasets
    token_num = ls_matrices[0].shape[0]          ## number of tokens
    W_mats = {i:None for i in range(len(ls_nums))} ## token topic weights
    H_mats = {i:None for i in range(len(ls_nums))} ## topic document weights
    Ab_errors = {i:0. for i in range(len(ls_nums))} ## absolute errors
    ## we update federated weight once over each batch
    fed_W = np.zeros((token_num, num_topics)) ## federated average weight
    
    ## loop over epochs
    for ep in range(epochs):
        batches = chunks(train_indexes, batch_size)  ## split data into batches
        ## loop over batches
        for batch in batches:
            ## we update federated weight once over each batch
            tmp_W = np.zeros((token_num, num_topics)) ## set Federated weight to zero
            ## current matrices/datasets
            cur_data = [ls_matrices[i] for i in batch]
            ## data size of matrices on clients
            cur_nums = [ls_nums[i] for i in batch]
            ## percents
            pts = np.array(cur_nums)/np.sum(cur_nums)
            ## pdb.set_trace()
            ## loop over each client in a batch
            for i,cl in enumerate(batch):
                W_mats[cl] = fed_W     ## copy latest weight matrix from the server
                if H_mats[cl] is None:
                    cl_res = exponential_method(ls_matrices[cl], k=num_topics, max_iter=max_iter, 
                                   init_mode='nndsvd', W_init=W_mats[cl], H_init=H_mats[cl])
                else:
                    cl_res = exponential_method(ls_matrices[cl], k=num_topics, max_iter=max_iter, 
                                   init_mode='', W_init=W_mats[cl], H_init=H_mats[cl])
                ## update factor matrices for each client    
                W_mats[cl], H_mats[cl], Ab_errors[cl] = cl_res
                ## federated token-topic weight
                tmp_W += pts[i]*W_mats[cl]
            ## update overall weight matrix
            fed_W =perc*fed_W +(1. - perc)*tmp_W
        ## compute average loss
        print('Absolute errors at Epoch ', ep, Ab_errors)
        print('Absolute error at Epoch ',ep, ' is:',sum(Ab_errors.values()))
        np.random.shuffle(train_indexes)     # shuffle indices of examples for next epoch
    ## finished training
    return fed_W, W_mats, H_mats

def top_k(arr, n):
    """
    find the indices of top K entries of a numoy array
    
    :params[in]: arr, numpy array
    :params[in]: n, integer, top n elements
    
    :params[out]: loc, numpy array
    """
    indices = (-arr).argsort()[:n]
    return indices

def top_keywords(W, features, num=20):
    """
    return top keywords for all topics
    
    :params[in]: W, the term-topic matrix, a factor matrix from NMF
    :params[in]: features, a list, all features for the corpora
    :params[in]: num, integer, default 20, the number of top keywords
    
    :params[out]: res, a dictionary
    """
    num_topics = W.shape[1]   ## total number of topics
    if W.shape[0]!=len(features):
        raise Exception("Error: The number of features Should be equal to the number of rows of input matrix")
        return
    else:
        res = {}  ## to store result
        for i in range(num_topics):
            indices = top_k(W[:,i], num) ## array of indices
            res[i] = [features[it] for it in indices]  ## top keywords for topic i
        ## return dictionary
        return res