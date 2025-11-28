import numpy as np
class C2GC:
    def __init__(self, beta=1.5):
        self.beta = beta

    def cal_conductance(self, adj, nodes_subset):
        # ensure numpy
        A = np.asarray(adj)
        n = len(nodes_subset)

        if n <= 1:
            return [np.inf]

        degrees = A.sum(axis=1)  # degree vector
        conductances = []

        # sweep cut
        for k in range(1, n):
            S = nodes_subset[:k]
            rest = nodes_subset[k:]

            # create boolean mask
            S_mask = np.zeros(A.shape[0], dtype=bool)
            S_mask[S] = True
            rest_mask = ~S_mask

            # cut = sum of edges crossing S and rest
            cut = A[np.ix_(S, rest)].sum()

            vol_S = degrees[S].sum()
            vol_rest = degrees[rest].sum()

            phi = cut / max(min(vol_S, vol_rest), 1e-12)
            conductances.append(phi)

        return conductances

    def run(self, adj, K):
        n = adj.shape[0]
        subgraph_list = [list(range(n))]
        mapping_list = []  # 保存每次切分的映射矩阵
        adj_subgraphs = [] # 保存每个子图的邻接矩阵

        while len(subgraph_list) < K:
            avg_len = np.mean([len(c) for c in subgraph_list])
            selected_idx = None
            for idx, cluster in enumerate(subgraph_list):
                if len(cluster) > self.beta * avg_len:
                    selected_idx = idx
                    break
            if selected_idx is None:
                best_conductance = np.inf
                best_idx = None
                best_k = None
                for idx, cluster in enumerate(subgraph_list):
                    temp_c = self.cal_conductance(adj, cluster)
                    if len(temp_c) == 0:
                        continue
                    min_c = min(temp_c)
                    k = temp_c.index(min_c) + 1
                    if min_c < best_conductance:
                        best_conductance = min_c
                        best_idx = idx
                        best_k = k
                if best_idx is None:
                    print("Warning: no valid cluster to split!")
                    break
                selected_idx = best_idx
                k = best_k
            else:
                cluster = subgraph_list[selected_idx]
                temp_c = self.cal_conductance(adj, cluster)
                if len(temp_c) == 0:
                    k = len(cluster) // 2
                else:
                    min_c = min(temp_c)
                    k = temp_c.index(min_c) + 1

            cluster = subgraph_list.pop(selected_idx)
            V1 = cluster[:k]
            V2 = cluster[k:]

            if len(V1) == 0 or len(V2) == 0:
                k = len(cluster) // 2
                V1 = cluster[:k]
                V2 = cluster[k:]

            # 保存映射矩阵和子图邻接矩阵
            # 映射矩阵 C1, C2: len(cluster) x len(Vi)
            C1 = np.zeros((len(cluster), len(V1)))
            C2 = np.zeros((len(cluster), len(V2)))
            for idx, node in enumerate(V1):
                C1[cluster.index(node), idx] = 1
            for idx, node in enumerate(V2):
                C2[cluster.index(node), idx] = 1
            mapping_list.append((C1, C2))

            A1 = adj[np.ix_(V1, V1)]
            A2 = adj[np.ix_(V2, V2)]
            adj_subgraphs.append((A1, A2))

            subgraph_list.append(V1)
            subgraph_list.append(V2)

            # print(f"Split cluster {selected_idx} len={len(cluster)} -> {len(V1)} + {len(V2)}, K_now={len(subgraph_list)}")
            # print(f"Temp conductance: {temp_c if 'temp_c' in locals() else 'N/A'}")

        return subgraph_list, mapping_list, adj_subgraphs



















# #### Below  here are all demo
#
#
# from Gennerate_coarsening_adj import *
# from V_Cluster import *
# import tensorflow as tf
# from utils import *
#
# if __name__ == "__main__":
#     n = 48
#     p = 0.2
#     # 2. 一键生成对称临街矩阵
#     adj = np.zeros((n, n), dtype=np.int8)
#     # 上三角随机填充
#     for i in range(n):
#         for j in range(i + 1, n):
#             if np.random.rand() < p:
#                 adj[i, j] = 1
#     # 对称镜像
#
#     datasetname = 'ohsumed'
#     tf.keras.backend.set_floatx('float32')  # 默认在tf 2.x的版本中数据会被转成float32来运行节省空间，但是会导致mutmal出问题，所以这里
#     # 直接就把后台所有的类型都禁止广播成32位了。 https://www.mianshigee.com/question/32592ygv/
#     test_adj, test_feature, test_M = load_data_test(
#         datasetname)
#
#     adj += adj.T
#     c2gc = C2GC(beta=1.2)
#     clusters, mappings, adj_subgraphs = c2gc.run(test_adj[2], 6)
#     A_coar, A_int_out, A_ext_out, S, C_list, A_k_list =compute_coarsened_adjacency( clusters, test_adj[2], False)
#     A_coar = prune_and_normalize_Acoar(A_coar,alpha=0.2)
#     vc = VCluster(test_adj[2], n_hop=2)
#
#     V_clusters, V_Masks, V_Adj = vc.extend_cluster(clusters)
#     print("Final clusters:")
#     for idx, c in enumerate(clusters):
#         print(f"Cluster {idx}: {c}",)
#     for i, (c, m) in enumerate(zip(V_clusters, V_Masks)):
#         print(f"V-cluster {i}: {c}")
#         print(f"mask: {m.astype(int).tolist()}")