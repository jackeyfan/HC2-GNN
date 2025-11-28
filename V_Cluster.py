import numpy as np

class VCluster:
    def __init__(self, adj: np.ndarray, n_hop: int = 1):
        """
        adj: (N x N) adjacency matrix (numpy array), undirected (can be weighted)
        n_hop: number of hops to collect outside neighbors
        """
        assert isinstance(adj, np.ndarray), "adj must be numpy ndarray"
        self.adj = adj
        self.N = adj.shape[0]
        self.n_hop = n_hop
        self.degrees = np.array(adj.sum(axis=1)).flatten()
        self.m = float(adj.sum()) / 2.0  # total edges for undirected

    def cal_conductance(self, nodes):
        """
        Strict conductance:
            phi(C) = |E(C, Cbar)| / min(vol(C), 2m - vol(C))
        nodes: list of node indices
        """
        if nodes is None or len(nodes) == 0:
            return float('inf')
        if len(nodes) == self.N:
            return float('inf')

        mask = np.zeros(self.N, dtype=bool)
        mask[nodes] = True
        cut = float(self.adj[np.ix_(nodes, ~mask)].sum())
        vol_C = float(self.degrees[nodes].sum())
        denom = min(vol_C, 2.0 * self.m - vol_C)
        if denom <= 0:
            return float('inf')
        return cut / denom

    def get_nhop_neighbors(self, V1):
        """
        Return ordered_neighbors and hop_level mapping.
        ordered_neighbors: list where all 1-hop nodes come first, then 2-hop, ...
        hop_level: dict node -> hop
        Ensures no node from V1 is included.
        """
        V1_set = set(V1)
        visited = set(V1)   # never include original cluster nodes
        frontier = set(V1)
        neighbors_total = []
        hop_level = {}

        for hop in range(1, self.n_hop + 1):
            next_front = set()
            for u in frontier:
                nbrs = set(np.where(self.adj[u] > 0)[0])
                next_front.update(nbrs)
            # remove already visited (includes original V1 and previously found nodes)
            next_front -= visited
            if not next_front:
                frontier = set()
                continue
            # record hop level and append sorted for determinism
            for v in sorted(next_front):
                hop_level[v] = hop
                neighbors_total.append(v)
            visited.update(next_front)
            frontier = next_front

        return neighbors_total, hop_level

    def _score_and_sort_within_hop(self, candidates, V1):
        """
        For a list of candidate nodes (all same hop), compute direct connectivity score
        (sum of adj[v, u] for u in V1) and sort descending by that score.
        Return sorted list.
        """
        scores = []
        V1_arr = np.array(V1, dtype=int) if len(V1) > 0 else np.array([], dtype=int)
        for v in candidates:
            direct = float(self.adj[v, V1_arr].sum()) if V1_arr.size > 0 else 0.0
            scores.append((v, direct))
        # sort by score desc, tie-break by node id
        scores.sort(key=lambda x: (-x[1], x[0]))
        return [v for v, s in scores], scores

    def extend_cluster(self, subgraph_list, stop_on_two_increases: bool = True):
        """
        Extend each cluster in subgraph_list according to the hop-by-hop rule.
        Input:
            subgraph_list: list of clusters, each cluster is a list of node indices
            stop_on_two_increases: whether to stop when conductance increases twice consecutively
        Returns:
            V_cluster_list: list of expanded clusters (list of node lists)
            Mask_list: list of boolean masks aligned with each expanded cluster (True indicates original V1 members)
        """
        V_cluster_list = []
        Mask_list = []
        Adj_list =[]

        # iterate over provided clusters (don't mutate caller's list)
        for orig_cluster in list(subgraph_list):
            original_V1 = list(orig_cluster)  # preserve original for mask
            V1_set = set(orig_cluster)

            # compute base conductance
            base_phi = self.cal_conductance(list(V1_set))

            # get ordered neighbors and hop levels
            ordered_neighbors, hop_level = self.get_nhop_neighbors(list(V1_set))

            # group by hop
            hop_buckets = {}
            for v in ordered_neighbors:
                h = hop_level[v]
                hop_buckets.setdefault(h, []).append(v)

            # hop-by-hop expansion
            V_temp = list(V1_set)  # maintain insertion order: original nodes first
            phi_curr = base_phi

            for hop in sorted(hop_buckets.keys()):
                # within this hop, sort candidates by direct connectivity to current original V1 (not to V_temp)
                # as you required: prefer candidates that are most connected to the ORIGINAL V1
                # (if you prefer connectivity to the currently expanded V_temp, change V1_arr accordingly)
                candidates = hop_buckets[hop]
                sorted_candidates, score_debug = self._score_and_sort_within_hop(candidates, original_V1)

                # debug printing of scores (optional - can comment out)
                # print(f"[DEBUG] hop={hop}, candidates (score desc) = {score_debug}")

                consecutive_increases = 0
                for v in sorted_candidates:
                    if v in V_temp:
                        continue  # safety: skip duplicates

                    cand_nodes = V_temp + [v]
                    phi_new = self.cal_conductance(cand_nodes)

                    if phi_new > phi_curr:
                        consecutive_increases += 1
                    else:
                        consecutive_increases = 0

                    if stop_on_two_increases and consecutive_increases >= 2:
                        # stop processing further nodes in this hop
                        break

                    # accept node
                    V_temp.append(v)
                    phi_curr = phi_new

                # proceed to next hop

            # generate mask relative to original V1
            mask = np.array([n in set(original_V1) for n in V_temp], dtype=bool)

            # ---- NEW: construct adjacency sub-matrix for V_temp ----
            idx = np.array(V_temp, dtype=int)
            A_v = self.adj[np.ix_(idx, idx)]   # (k × k)
            Adj_list.append(A_v)  # <----- 新增

            V_cluster_list.append(V_temp)
            Mask_list.append(mask)

        return V_cluster_list, Mask_list, Adj_list


# # ---------------- Demo usage ----------------
# if __name__ == "__main__":
#
#     adj = np.zeros((25, 25), dtype=float)
#
#     # 4-neighbor grid connectivity
#     def add_edge(i, j):
#         adj[i, j] = 1
#         adj[j, i] = 1
#
#     for r in range(5):
#         for c in range(5):
#             u = r*5 + c
#             if c < 4:  # right
#                 add_edge(u, u+1)
#             if r < 4:  # down
#                 add_edge(u, u+5)
#
#     # ----------- 构造 4 个初始 cluster（象限）-----------
#     # 左上 0~12、右上 2~14、左下 10~22、右下 12~24
#     # 这里用严格 4 个不重叠 cluster
#
#     cluster1 = [0,1,2,5,6,7]         # 左上角 2×3
#     cluster2 = [3,4,8,9]             # 右上角 2×2
#     cluster3 = [10,11,15,16]         # 左下角 2×2
#     cluster4 = [17,18,22,23,24]      # 右下角 2×3
#
#     subgraph_list = [cluster1, cluster2, cluster3, cluster4]
#
#     vc = VCluster(adj, n_hop=2)
#
#     V_clusters, Masks, Adj = vc.extend_cluster(subgraph_list)
#
#     print("\n=== Final Result ===")
#     for i, (c, m) in enumerate(zip(V_clusters, Masks)):
#         print(f"V-cluster {i}: {c}")
#         print(f"mask: {m.astype(int).tolist()}")