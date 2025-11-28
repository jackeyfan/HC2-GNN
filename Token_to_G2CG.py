import numpy as np
import scipy.sparse as sp
from C2GC_cluster import *

# ===== 1. 示例文本和 token 映射 =====
documents = [
    "this is a sample document, and this is a good question",
    "another sample text with words, and good  simple"
]

# 构建简单的 word-level vocab
vocab = list(set(" ".join(documents).split()))
word_id_map = {w: i for i, w in enumerate(vocab)}

# 模拟每个 word 对应的 token ids
# 这里假设每个 word 对应 1 个 token
word2token_ids = {i: [i] for i in range(len(vocab))}

# ===== 2. 构建 word-level adjacency matrix A_word =====
# 简单按共现窗口构建，窗口 = 2
window_size = 2
vocab_size = len(vocab)
A_word = np.zeros((vocab_size, vocab_size), dtype=float)

for doc in documents:
    words = doc.split()
    ids = [word_id_map[w] for w in words]
    for i in range(len(ids)):
        for j in range(i+1, min(i+window_size, len(ids))):
            A_word[ids[i], ids[j]] = 1
            A_word[ids[j], ids[i]] = 1  # 无向

# ===== 4. TokenC2GC =====
class TokenC2GC:
    def __init__(self, word2token_ids, beta=1.5):
        self.word2token_ids = word2token_ids
        self.beta = beta
        self.c2gc = C2GC(beta=self.beta)

    def run_on_tokens(self, A_word, word_ids, K):
        subgraph_list, mapping_list, adj_subgraphs = self.c2gc.run(A_word, K)

        clusters_token = []
        for cluster in subgraph_list:
            token_cluster = []
            for wid_group in cluster:
                if isinstance(wid_group, (list, np.ndarray)):
                    for wid in wid_group:
                        token_cluster.extend(self.word2token_ids[wid])
                else:
                    token_cluster.extend(self.word2token_ids[wid_group])
            clusters_token.append(token_cluster)

        return clusters_token, mapping_list, adj_subgraphs

# ===== 5. 运行示例 =====
word_ids = list(range(len(vocab)))
token_c2gc = TokenC2GC(word2token_ids)
token_clusters, mapping_list, adj_subgraphs = token_c2gc.run_on_tokens(A_word, word_ids, K=2)

# ===== 6. 打印结果 =====
print("Word vocab:", vocab)
for cid, cluster in enumerate(token_clusters):
    token_words = [vocab[tok] for tok in cluster]
    print(f"Token Cluster {cid}: {token_words}")
