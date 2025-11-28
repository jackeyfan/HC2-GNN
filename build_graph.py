import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm

from V_Cluster import *
from C2GC_cluster import *
from Preper_Data import *
from Gennerate_coarsening_adj import *


# if len(sys.argv) < 2:
# 	sys.exit("Use: python build_graph.py <dataset>")

# settings
datasets = ['mr', 'ohsumed', 'R8', 'R52', 'TREC', 'ag_news', 'WebKB', 'SST1', 'SST2']

#dataset = sys.argv[1]
dataset = 'ohsumed'
if dataset not in datasets:
    sys.exit("wrong dataset name")

try:
    window_size = 3
    #window_size = int(sys.argv[2])
except:
    window_size = 3
    print('using default window size = 3')

try:
    weighted_graph = bool(sys.argv[3])
except:
    weighted_graph = False
    print('using default unweighted graph')

truncate = False # whether to truncate long document
MAX_TRUNC_LEN = 1000


print('loading raw data')

# load pre-trained word embeddings
word_embeddings_dim = 300
word_embeddings = {}

with open('glove.6B.' + str(word_embeddings_dim) + 'd.txt', 'r') as f:
    for line in f.readlines():
        data = line.split()
        word_embeddings[str(data[0])] = list(map(float,data[1:]))


# load document list
doc_name_list = []
doc_train_list = []
doc_test_list = []

with open('data/' + dataset + '.txt', 'r') as f:
    for line in f.readlines():
        doc_name_list.append(line.strip())
        temp = line.split("\t")

        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())


# load raw text
doc_content_list = []

with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip())


# map and shuffle
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

ids = train_ids + test_ids


shuffle_doc_name_list = []
shuffle_doc_words_list = []
for i in ids:
    shuffle_doc_name_list.append(doc_name_list[int(i)])
    shuffle_doc_words_list.append(doc_content_list[int(i)])


# build corpus vocabulary
word_set = set()

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    word_set.update(words)

vocab = list(word_set)
vocab_size = len(vocab)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i


# initialize out-of-vocabulary word embeddings
oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)


# build label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)


# select 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size
test_size = len(test_ids)



# build graph function
def build_graph(start, end):
    x_adj = []
    x_feature = []
    y = []
    doc_len_list = []
    token_len_list = []
    vocab_set = set()
 ####### for token graph asn the Virture cluster
    token_adj_list = []
    token_embed_list = []
    token_M_list = []

    clusters = []
    A_coar = []
    vc = []
    V_clusters = []
    V_Masks = []
    V_Adj = []
    c2gc = C2GC(beta=1.2)
#######################

    #for i in tqdm(range(3)):  # 非测试阶段，请把 end 的赋值去掉。
    for i in tqdm(range(start, end)):  # 非测试阶段，请把 end 的赋值去掉。
        doc_words = shuffle_doc_words_list[i].split()
        if truncate:
            doc_words = doc_words[:MAX_TRUNC_LEN]
        doc_len = len(doc_words)

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
    
        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[vocab[p]])
            col.append(doc_word_id_map[vocab[q]])
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
    
        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            features.append(word_embeddings[k] if k in word_embeddings else oov[k])

        x_adj.append(adj)
        x_feature.append(features)

        #######################################
        # 构建 token graph
        #######################################
        A_word_dense = adj.toarray()
        A_token, X_token, M_token2word, features2 = build_token_graph(
            doc_words=doc_words,
            doc_word_id_map=doc_word_id_map,
            A_word=A_word_dense,
            word_embeddings=word_embeddings
        )
        # 保存 token graph
        token_adj_list.append(sp.csr_matrix(A_token))
        token_embed_list.append(X_token)
        token_M_list.append(M_token2word)
        token_len_list.append(X_token.shape[0])
        # 以上为C2GC  准备的 token graph

        # 切分c2gc 切分的过程请放在这里 Build subgraphs for the token graph、
        # 这里基本上是挨个的生成的，所以速度会比较慢， 生成 A_coar 还是有点儿问题，要确认一下
        #for i in range(len(test_adj)):
        clusters_i, mappings, adj_subgraphs = c2gc.run(A_token, 8)  # generate cluster
        clusters.append(clusters_i)
        vc_i = VCluster(A_token, n_hop=2)  # define vc
        V_clusters_i, V_Masks_i, V_Adj_i = vc_i.extend_cluster(clusters_i)  # generate V_cluster
        V_Adj.append(V_Adj_i)
        V_Masks.append(V_Masks_i)
        V_clusters.append(V_clusters_i)
        A_coar_i, A_int_out, A_ext_out, S, C_list, A_k_list = compute_coarsened_adjacency(clusters_i, A_token,False)
        A_coar_i = prune_and_normalize_Acoar(A_coar_i, alpha=0.2)
        A_coar.append(A_coar_i)
        vc.append(vc_i)
        # 后面三个的返回为 token graph 准备的

    # one-hot labels
    for i in range(start, end):
    #for i in range(3):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)

    # acturely, the token_embed_list can be obtained from x_adj multiply token_M_list
    return (x_adj, x_feature, y, doc_len_list, vocab, token_embed_list, token_adj_list, token_M_list, token_len_list, V_clusters, V_Adj, V_Masks, A_coar)


def build_token_graph(doc_words, doc_word_id_map, A_word, word_embeddings):
    """
    doc_words: token sequence (list of str)
    doc_word_id_map: mapping unique-word → word-graph-node index
    A_word: np.array (word graph, W×W)
    word_embeddings: dict word → embedding (300维)

    return:
        A_token: token-level adjacency (T×T)
        X_token: token-level embedding (T×300)
        M: mapping matrix (T×W)
    """
    T = len(doc_words)
    W = A_word.shape[0]

    # 1. 构建 token→unique-word 映射矩阵 M
    M = np.zeros((T, W), dtype=np.float32)
    for i, w in enumerate(doc_words):
        if w in doc_word_id_map:
            wid = doc_word_id_map[w]
            M[i, wid] = 1.0

    # 2. token adjacency  A_token = M * A_word * M^T
    A_token = M @ A_word @ M.T

    # 3. token embedding  X_token = M * X_word

    X_word = np.zeros((W, 300), dtype=np.float32)
    for w, wid in doc_word_id_map.items():
        vec = word_embeddings[w] if w in word_embeddings else oov[w]
        X_word[wid] = np.array(vec)

    X_token = M @ X_word

    return A_token, X_token, M, X_word



print('building graphs for training')
x_adj, x_feature, y, doc_len_list_train, vocab_train, token_embed_list, token_adj_list, token_M_list, train_token_len_list, V_clusters, V_Adj, V_Masks, A_coar = build_graph(start = 0, end = real_train_size)

# print('building graphs for training + validation')
# allx_adj, allx_feature, ally, doc_len_list_train, vocab_train,token_adj_list2, token_embed_list2, token_M_list2 = build_graph(start=0, end=train_size)

print('building graphs for validation')
# Where the validation set is setting down here, we recommend to use the training + validation and random it when training.
val_x_adj, val_x_feature, val_y, doc_len_list_val,vocab_val, val_token_embed_list, val_token_adj_list, val_token_M_list, val_token_len_list, val_V_clusters, val_V_Adj, val_V_Masks,val_A_coar = build_graph(start=real_train_size, end= train_size)

print('building graphs for test')
t_x_adj, t_x_feature, t_y, doc_len_list_test, vocab_test, t_token_embed_list,t_token_adj_list, t_token_M_list, t_token_len_list, t_V_clusters, t_V_Adj, t_V_Masks,t_A_coar = build_graph(start=train_size, end=train_size +test_size)

doc_len_list = doc_len_list_train +doc_len_list_val + doc_len_list_test
token_len_list = train_token_len_list + val_token_len_list + t_token_len_list


# save for padding
max_train_cluster_len = max(len(sub2) for sub1 in V_clusters for sub2 in sub1 if isinstance(sub1, list) and isinstance(sub2, list))
max_val_cluster_len =  max(len(sub2) for sub1 in val_V_clusters for sub2 in sub1 if isinstance(sub1, list) and isinstance(sub2, list))
max_text_cluster_len = max(len(sub2) for sub1 in t_V_clusters for sub2 in sub1 if isinstance(sub1, list) and isinstance(sub2, list))
max_cluster_len = max([max_train_cluster_len, max_val_cluster_len, max_text_cluster_len])
max_lens = [max(doc_len_list), max_cluster_len, max(token_len_list)]

# statistics
print('max_doc_length',max(doc_len_list),'min_doc_length',min(doc_len_list),
      'average {:.2f}'.format(np.mean(doc_len_list)))
# print('training_vocab',len(vocab_train),'test_vocab',len(vocab_test),
#       'intersection',len(vocab_train & vocab_test) )


# dump objects
with open("data/ind.{}.x_adj".format(dataset), 'wb') as f:
    pkl.dump(x_adj, f)

with open("data/ind.{}.x_embed".format(dataset), 'wb') as f:
    pkl.dump(x_feature, f)

with open("data/ind.{}.y".format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open("data/ind.{}.tx_adj".format(dataset), 'wb') as f:
    pkl.dump(t_x_adj, f)

with open("data/ind.{}.tx_embed".format(dataset), 'wb') as f:
    pkl.dump(t_x_feature, f)

with open("data/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(t_y, f)

with open("data/ind.{}.valx_adj".format(dataset), 'wb') as f:
    pkl.dump(val_x_adj, f)

with open("data/ind.{}.valx_embed".format(dataset), 'wb') as f:
    pkl.dump(val_x_feature, f)

with open("data/ind.{}.valy".format(dataset), 'wb') as f:
    pkl.dump(val_y, f)




### 保存和 token graph 相关的数据结构
with open("data/ind.{}.token_adj".format(dataset), 'wb') as f:
    pkl.dump(token_adj_list, f)

with open("data/ind.{}.token_embed".format(dataset), 'wb') as f:
    pkl.dump(token_embed_list, f)

with open("data/ind.{}.token_M".format(dataset), 'wb') as f:
    pkl.dump(token_M_list, f)

with open("data/ind.{}.val_token_adj".format(dataset), 'wb') as f:
    pkl.dump(val_token_adj_list, f)

with open("data/ind.{}.val_token_embed".format(dataset), 'wb') as f:
    pkl.dump(val_token_embed_list, f)

with open("data/ind.{}.val_token_M".format(dataset), 'wb') as f:
    pkl.dump(val_token_M_list, f)

with open("data/ind.{}.t_token_adj".format(dataset), 'wb') as f:
    pkl.dump(t_token_adj_list, f)

with open("data/ind.{}.t_token_embed".format(dataset), 'wb') as f:
    pkl.dump(t_token_embed_list, f)

with open("data/ind.{}.t_token_M".format(dataset), 'wb') as f:
    pkl.dump(t_token_M_list, f)

############ Save the clusters and the related parameters

with open("data/ind.{}.V_clusters".format(dataset), 'wb') as f:
    pkl.dump(V_clusters, f)

with open("data/ind.{}.V_Adj".format(dataset), 'wb') as f:
    pkl.dump(V_Adj, f)

with open("data/ind.{}.V_Masks".format(dataset), 'wb') as f:
    pkl.dump(V_Masks, f)

with open("data/ind.{}.val_V_clusters".format(dataset), 'wb') as f:
    pkl.dump(val_V_clusters, f)

with open("data/ind.{}.val_V_Adj".format(dataset), 'wb') as f:
    pkl.dump(val_V_Adj, f)

with open("data/ind.{}.val_V_Masks".format(dataset), 'wb') as f:
    pkl.dump(val_V_Masks, f)

with open("data/ind.{}.t_V_clusters".format(dataset), 'wb') as f:
    pkl.dump(t_V_clusters, f)

with open("data/ind.{}.t_V_Adj".format(dataset), 'wb') as f:
    pkl.dump(t_V_Adj, f)

with open("data/ind.{}.t_V_Masks".format(dataset), 'wb') as f:
    pkl.dump(t_V_Masks, f)

### 保存和 token graph 相关的数据结构
with open("data/ind.{}.max_length".format(dataset), 'wb') as f:
    pkl.dump(max_lens, f)

with open("data/ind.{}.A_coar".format(dataset), 'wb') as f:
    pkl.dump(A_coar, f)
with open("data/ind.{}.val_A_coar".format(dataset), 'wb') as f:
    pkl.dump(val_A_coar, f)
with open("data/ind.{}.t_A_coar".format(dataset), 'wb') as f:
    pkl.dump(t_A_coar, f)

