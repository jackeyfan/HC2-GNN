HC2-GNN (Hierarchical Clustering and Coarsening Graph Neural Network) is a graph-based text classification framework designed to improve structural modeling efficiency and preserve both local and global contextual information.

Overview

HC2-GNN introduces a two-stage hierarchical structure construction mechanism:

C2GC clustering
A compromise conductance graph clustering algorithm that efficiently groups semantically related nodes while maintaining word order.

Virtue Cluster Extension (V-Cluster)
A hop-based cluster expansion strategy that adaptively enlarges clusters under conductance tolerance.

These components produce a multi-granularity graph structure, enabling the model to better capture information flow across clusters with reduced computation overhead.

The Framework of this paper is:
<img width="1872" height="1186" alt="archit" src="https://github.com/user-attachments/assets/bdc6287d-ab6e-46e5-afc8-199eb7c9081b" />

All the code will come soon after we transfer it from our server.
