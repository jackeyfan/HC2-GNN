

# from Gennerate_coarsening_adj import *
# from V_Cluster import *

from C2GC_cluster import *




def pad_node_features(list_X, max_len):
    """
    list_X: list of np.arrays, each shape (Ni, d)
    Returns:
      X_batch: np.array (B, N_max, d)
      node_mask: np.array (B, N_max)  (1 for real nodes, 0 for padding)
      pad_index: list length B, pad_index[b] == original_N[b] (first padding pos)
    """
    B = len(list_X)
    d = list_X[0].shape[1]
    # ns = [x.shape[0] for x in list_X]
    # N_max = max(ns)
    X_batch = np.zeros((B, max_len, d), dtype=np.float32)
    node_mask = np.zeros((B, max_len), dtype=np.float32)
    pad_index = []
    for b, x in enumerate(list_X):
        n = x.shape[0]
        X_batch[b, :n, :] = x
        node_mask[b, :n] = 1.0
        pad_index.append(n if n < max_len else max_len-1)  # if full, pad_index points to last row
    return X_batch, node_mask, pad_index



# padded_A : np.ndarray  (Batch, K, S, S)
def pad_cluster_adjs(cluster_A_list, max_len):
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

    # 写入真实 adjacency
    for b in range(B):
        for k, A in enumerate(cluster_A_list[b]):
            n = A.shape[0]
            padded[b, k, :n, :n] = A


    return padded


# the data sturture is [k * N * N]
def simple_pad_adjs(cluster_adj_list, max_len):
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
        padded[i, :n, :n] = A

    return padded


# the data sturture is [Batch * cluster_number * nodes_number]
def simple_pad_cluster_list(batch_V_clusters, max_len, pad_value=0):
    """
    batch_V_clusters: list of list of list
        batch, clusters, node_ids (variable length)

    Returns:
        padded: list of list of list of int
            shape = (B, K, S)
        K: max number of clusters
        S: max nodes per cluster
    """
    B = len(batch_V_clusters)
    K = max(len(v_list) for v_list in batch_V_clusters)
    S = max(len(cluster) for v_list in batch_V_clusters for cluster in v_list)

    # 初始化
    padded = [[[pad_value for _ in range(max_len)] for _ in range(K)] for _ in range(B)]

    # 填充
    for i, v_list in enumerate(batch_V_clusters):
        for j, cluster in enumerate(v_list):
            for k, node_id in enumerate(cluster):
                padded[i][j][k] = node_id
    return padded

def pad_M(M, T, W):
    """
    M: list of numpy arrays, shape may vary per sample
    T, W: padding 后目标大小
    返回: numpy array, shape = (B, T, W)
    """
    B = len(M)
    M_pad = np.zeros((B, T, W), dtype=np.float32)

    for i in range(B):
        m = np.asarray(M[i])
        t_old, w_old = m.shape
        t_copy = min(t_old, T)
        w_copy = min(w_old, W)
        M_pad[i, :t_copy, :w_copy] = m[:t_copy, :w_copy]

    return M_pad
