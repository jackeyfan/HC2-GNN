import numpy as np
import scipy.sparse as sp
from typing import List, Tuple

def compute_coarsened_adjacency(subgraph_list: List[List[int]],
                                A,
                                make_sparse: bool = True
                                ) -> Tuple:
    """
    Compute coarsened adjacency following your math:
      - C^(k) (N x Nk) binary mapping matrices (sparse)
      - A^(k) = Ck^T @ A @ Ck  (Nk x Nk)
      - A_int = sum_k Ck @ A^(k) @ Ck^T
      - A_ext = A - A_int
      - S (N x K) assignment matrix (dense np.array)
      - A_coar = S^T @ A_ext @ S

    Args:
        subgraph_list: list of clusters, each is a list of node indices (0-based)
        A: adjacency; can be numpy.ndarray, nested list, or scipy.sparse matrix
        make_sparse: if True returns sparse matrices for A_coar/A_int/A_ext; else returns dense np arrays

    Returns:
        A_coar, A_int, A_ext, S, C_list, A_k_list
          - A_coar: K x K (sparse csr if make_sparse else dense ndarray)
          - A_int: N x N (sparse csr if make_sparse else dense ndarray)
          - A_ext: N x N (sparse csr if make_sparse else dense ndarray)
          - S: N x K dense numpy array (0/1)
          - C_list: list of Ck (each scipy.sparse.csr_matrix of shape N x Nk)
          - A_k_list: list of Ak (each scipy.sparse.csr_matrix of shape Nk x Nk)
    """
    # --- normalize A to sparse csr ---
    if sp.issparse(A):
        A_sp = A.tocsr()
        N = A_sp.shape[0]
    else:
        A_arr = np.asarray(A, dtype=float)
        if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
            raise ValueError("Adjacency A must be a square 2D array/matrix.")
        N = A_arr.shape[0]
        A_sp = sp.csr_matrix(A_arr)

    K = len(subgraph_list)

    # --- build C_list and A_k_list ---
    C_list = []
    A_k_list = []

    for cluster in subgraph_list:
        Nk = len(cluster)
        if Nk == 0:
            # empty cluster case
            Ck = sp.csr_matrix((N, 0), dtype=float)
            Ak = sp.csr_matrix((0, 0), dtype=float)
            C_list.append(Ck)
            A_k_list.append(Ak)
            continue

        # Build Ck (N x Nk) with ones at (node_index, local_pos)
        rows = np.array(cluster, dtype=int)
        cols = np.arange(Nk, dtype=int)
        data = np.ones(Nk, dtype=float)
        # coo with (row_i, col_i) pairs
        Ck = sp.coo_matrix((data, (rows, cols)), shape=(N, Nk)).tocsr()
        C_list.append(Ck)

        # Ak = Ck^T * A * Ck  (Nk x Nk)
        Ak = (Ck.T).dot(A_sp).dot(Ck)
        Ak = sp.csr_matrix(Ak)
        A_k_list.append(Ak)

    # --- compute A_int = sum_k Ck @ Ak @ Ck^T ---
    A_int = sp.csr_matrix((N, N), dtype=float)
    for Ck, Ak in zip(C_list, A_k_list):
        if Ak.shape[0] == 0:
            continue
        A_int += Ck.dot(Ak).dot(Ck.T)

    # --- A_ext = A - A_int ---
    A_ext = A_sp - A_int

    # --- build assignment matrix S (dense) N x K ---
    S = np.zeros((N, K), dtype=int)
    for j, cluster in enumerate(subgraph_list):
        if len(cluster) == 0:
            continue
        S[cluster, j] = 1

    # --- compute A_coar = S^T * A_ext * S  (use sparse multiply) ---
    S_sp = sp.csr_matrix(S)  # convert to sparse for multiplication
    A_coar_sp = (S_sp.T).dot(A_ext).dot(S_sp)  # result is K x K sparse

    # --- return in requested format ---
    if make_sparse:
        return A_coar_sp.tocsr(), A_int.tocsr(), A_ext.tocsr(), S, C_list, A_k_list
    else:
        return A_coar_sp.toarray(), A_int.toarray(), A_ext.toarray(), S, C_list, A_k_list


def prune_and_normalize_Acoar( A_coar, alpha: float = 0.0, row_normalize: bool = True, keep_symmetric: bool = True):
    """
    Apply threshold pruning and row-normalization to A_coar.
    Args:
        A_coar: sparse (csr) or dense numpy matrix, shape (K x K)
        alpha: prune threshold. Entries < alpha are set to zero.
        row_normalize: if True, normalize each row to sum to 1.
        keep_symmetric: if True, enforce symmetry by (A + A.T)/2 after pruning.

    Returns:
        A_out: matrix with same type (sparse/dense) as input
    """
    A = A_coar.copy().astype(float)

    # ---- (1) Row-normalize ----
    if row_normalize:
        row_sum = A.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        A = A / row_sum

    # ---- (2) Prune using alpha ----
    # Keep entries >= alpha
    mask = A >= alpha
    A_pruned = A * mask

    # ---- (3) Ensure no isolated nodes ----
    # For any row with all-zero => keep the largest original normalized edge
    for i in range(A.shape[0]):
        if A_pruned[i].sum() == 0:
            # find largest element in normalized A
            j = np.argmax(A[i])
            if A[i, j] > 0:
                A_pruned[i, j] = A[i, j]
                if keep_symmetric:
                    A_pruned[j, i] = A[i, j]

    # ---- (4) Symmetrize ----
    if keep_symmetric:
        A_pruned = np.maximum(A_pruned, A_pruned.T)

    return A_pruned
