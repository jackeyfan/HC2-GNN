import numpy as np
import pickle as pkl

import scipy.sparse
import scipy.sparse as sp
# from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse.linalg import eigsh

#from New_graphs import *

import sys
import random
import re
from tqdm import tqdm  # 显示进度的


# import sparse


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str, names):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors and adjacency matrix of the training instances as list;
    ind.dataset_str.tx => the feature vectors and adjacency matrix of the test instances as list;
    ind.dataset_str.allx => the feature vectors and adjacency matrix of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as list;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    # names = ['x_adj', 'x_embed', 'y', 'tx_adj', 'tx_embed', 'ty', 'allx_adj', 'allx_embed', 'ally']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x_adj, x_embed, y, valx_adj, valx_embed, valy, tx_adj, tx_embed, ty = tuple(objects)
    names3 = ['token_embed', 'token_adj', 'token_M', 'val_token_embed', 'val_token_adj', 'val_token_M', 't_token_embed',
              't_token_adj', 't_token_M']

    train_adj = []
    train_embed = []
    val_adj = []
    val_embed = []
    test_adj = []
    test_embed = []

    for i in range(len(y)):
        adj = x_adj[i].toarray()
        embed = np.array(x_embed[i])
        train_adj.append(adj)
        train_embed.append(embed)

    for i in range( len(valy)):  # train_size):
        adj = valx_adj[i].toarray()
        embed = np.array(valx_embed[i])
        val_adj.append(adj)
        val_embed.append(embed)

    for i in range(len(ty)):
        adj = tx_adj[i].toarray()
        embed = np.array(tx_embed[i])
        test_adj.append(adj)
        test_embed.append(embed)

    train_adj = np.array(train_adj, dtype=object)
    val_adj = np.array(val_adj, dtype=object)
    test_adj = np.array(test_adj, dtype=object)
    train_embed = np.array(train_embed, dtype=object)
    val_embed = np.array(val_embed, dtype=object)
    test_embed = np.array(test_embed, dtype=object)
    train_y = np.array(y)
    val_y = np.array(valy)  # get the validation part
    test_y = np.array(ty)

    return train_adj, train_embed, train_y, val_adj, val_embed, val_y, test_adj, test_embed, test_y


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            # mx = mx.tocoo()
            mx = scipy.sparse.coo_matrix(mx)
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def coo_to_tuple(sparse_coo):
    return (sparse_coo.coords.T, sparse_coo.data, sparse_coo.shape)



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def preprocess_features_oringnal(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # max_length = max([len(f) for f in features])
    max_length = 197
    #res_rll = []
    # max_length = 300
    for i in tqdm(range(features.shape[0])):

        feature = np.array(features[i])
        pad = max_length - feature.shape[0]  # padding for each epoch
        feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
        features[i] = feature

    return np.array(list(features))
# 这里默认的max_length 是针对于各个不同的集合（test, training, validation, 会不同长度。为了方便，这里测试的时候把mr 的 max_length固定成44

def count_max_length(subgraph_index_list):
    max_length = 0  # 先得到最长的subgraph的长度。因为会发生 test, train, val 不一致的情况，所以这里放外面去
    for subgraph_index in subgraph_index_list:
        max_length_temp = max([len(f) for f in subgraph_index])
        if max_length_temp > max_length:
            max_length = max_length_temp
    return max_length

def preprocess_cluster_featrures(features, adj_list, adj_subgraph_index_list, max_length):

    # 开始padding 成最长的
    new_features = []
    new_adj = []
    mask = []
    for i in range(features.shape[0]):
        feature = np.array(features[i])
        sub_adj_list = adj_list[i]
        sub_adj_index = adj_subgraph_index_list[i]
        sub_mask = []
        temp_feature_list = []
        counter = 0

        for index in sub_adj_index:
            temp_mask = np.ones(max_length)
            pad = max_length - len(index)
            feature_temp = feature[index] # 先抽取出来index 个 feature
            feature_temp = np.pad(feature_temp,((0, pad),(0,0)),mode='constant')
            temp_mask[len(index):] = 0
            temp_feature_list.append(feature_temp)
            sub_adj_list[counter] = np.pad(sub_adj_list[counter],((0, pad),(0,pad)),mode='constant')
            counter = counter +1 # 因为也要 Padding sub_adj 因此用一个counter 来循环
            sub_mask.append(temp_mask)

        mask.append(sub_mask)
        new_features.append(temp_feature_list)
        new_adj.append(sub_adj_list)

    return np.array(list(new_features)), np.array(list(adj_list)), mask


def preprocess_adj(cluster_adj_list, max_len):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    batch = len(cluster_adj_list)
    adj_list = []
    if isinstance(cluster_adj_list, list):
        #S = max(A.shape[0] for A in cluster_adj_list)
        for i in range(batch):
            adj = cluster_adj_list[i].toarray()
            adj_list.append(adj)
        cluster_adj_list = adj_list

    padded = np.zeros((batch, max_len, max_len), dtype=np.float32)

    for i, A in enumerate(cluster_adj_list):
        n = A.shape[0]
        A = A + np.eye(n)
        A = normalize_adj(A)
        padded[i, :n, :n] = A

    return padded

# padded_A : np.ndarray  (Batch, K, S, S)
def preprocess_cluster_adj(cluster_A_list, max_len):
    """
    cluster_A_list: list[list[np.ndarray]]
        cluster_A_list[b][k] 是一个 (n × n) 的 adjacency matrix

    return:
        padded_A : np.ndarray  (B, K, S, S)
        S        : 最大 cluster size
    """
    B = len(cluster_A_list)
    K = max(len(clusters) for clusters in cluster_A_list)
    # 初始化 padded tensor
    padded = np.zeros((B, K, max_len, max_len), dtype=np.float32)
    mask = np.zeros((B, K, max_len), dtype=np.float32)
    # 写入真实 adjacency
    for b in range(B):
        for k, A in enumerate(cluster_A_list[b]):
            n = A.shape[0]
            A = A + np.eye(n)
            A = normalize_adj(A)
            padded[b, k, :n, :n] = A
            mask[b, k, :n] = 1.0
    return padded,mask







def chebyshev_polynomials(adj, k):  # 这个是用来预处理adj的，因为如果不是kipf的理论的话。adj是要转成cheb(adj)的形式的
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    # return sparse_to_tuple(t_k), #scaled_laplacian
    return t_k[k]


# p1 = np.random.random([3,3])
# p1 = np.matmul(p1,p1.T)
# laplacian = chebyshev_polynomials(p1,2)
#
# print('test the chebyshev recurrence')

def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if (len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# def pad_idx(idx, max_len, ratio):
#     len1 = int(max_len * ratio)
#     len2 = len1 - len(idx)
#     len3 = max_len - len2
#     data = np.arange(max_len)
#     y  = np.concatenate((idx,data[len3:]))
#     return y
#
from Gennerate_coarsening_adj import *
from V_Cluster import *
import tensorflow as tf
from C2GC_cluster import *
from Preper_Data import *
#
# def preprocessing_fuc( test_feature, test_adj):
#
#     clusters =[]
#     A_coar =[]
#     vc =[]
#     V_clusters = []
#     V_Masks =[]
#     V_Adj = []
#     # A = []
#     c2gc = C2GC(beta=1.2)
#
#     for i in range(len(test_adj)):
#         clusters_i, mappings, adj_subgraphs = c2gc.run(test_adj[i], 8) # generate cluster
#         clusters.append(clusters_i)
#         vc_i = VCluster(test_adj[i], n_hop=2) # define vc
#         V_clusters_i, V_Masks_i, V_Adj_i = vc_i.extend_cluster(clusters_i) # generate V_cluster
#         V_Adj.append(V_Adj_i)
#         V_Masks.append(V_Masks_i)
#         V_clusters.append(V_clusters_i)
#         A_coar_i, A_int_out, A_ext_out, S, C_list, A_k_list =compute_coarsened_adjacency( clusters_i, test_adj[i], False)
#         A_coar_i = prune_and_normalize_Acoar(A_coar_i,alpha=0.2)
#         A_coar.append(A_coar_i)
#         vc.append(vc_i)
#
#     test_feature, node_mask, pad_index = pad_node_features(test_feature)
#     test_adj = simple_pad_adjs(test_adj)
#     A = [test_adj, A_coar]   # A_coar 不需要 padding
#     V_Adj, _ = pad_cluster_adjs(V_Adj)
#     V_Masks, _,_ = simple_pad_cluster_list(V_Masks, pad_value= 0)
#     V_Masks = np.array(V_Masks, dtype=np.float32)
#     V_clusters = simple_pad_cluster_list(V_clusters)
#
#     return test_adj, test_feature, A ,V_Adj, V_Masks, V_clusters
#

def new_load_data(dataset_str, names):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors and adjacency matrix of the training instances as list;
    ind.dataset_str.tx => the feature vectors and adjacency matrix of the test instances as list;
    ind.dataset_str.allx => the feature vectors and adjacency matrix of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as list;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names3 = ['token_embed', 'token_adj', 'token_M', 'val_token_embed', 'val_token_adj', 'val_token_M', 't_token_embed',
              't_token_adj', 't_token_M']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    # token_embed, token_adj, token_M, val_token_embed, val_token_adj, val_token_M, t_token_embed, t_token_adj, t_token_M = tuple(objects)
    # return token_embed, token_adj, token_M, val_token_embed, val_token_adj, val_token_M, t_token_embed, t_token_adj, t_token_M

    return objects
