

import numpy as np
import pygsp as gsp
from pygsp import graphs, filters, reduction
#from utils import *
import tensorflow as tf
from matplotlib import pyplot as plt
from New_Coarsening import Graphs
from New_cost import *
from New_graph import *

#  对Aint 进行lift_back 操作，进入到整体的A的纬度。然后反向回来可以生成Aext,再利用Aext 来生成 Acoar

# def liftback(adj, C, full_adj):
#     affin = np.zeros([full_adj.shape[0], len(C)])
#     for i in range(full_adj.shape[0]):
#         for j in range(len(C)):
#             if C[j] == i:
#                 affin[i, j] = 1
#     Adj_int = affin @ adj @ np.transpose(affin)
#
#     return Adj_int

# 构建一个带线性顺序的adj,这个adj 能反应这个文档的词汇顺序关系
# def build_new_graph(adj):
#     adj[:,0]=1
#     adj[0,:]=1
#
#     return adj

# def coarsen(G,pattrition):
#     graph = Graphs.coarsening_pooling(G,normalize=False)
#     return graph

# 对图进行cluster， 再patrition的时候，可以计算出 C
def partrtition(adj, numbers):
    #adj = G.adj
    index = np.arange(adj.shape[0])
    index_list =[]
    adj_list = []
    cluser_len_list = []
    adj_list.append(adj)
    index_list.append(index) #保存每个cluster对应的编号
    cluser_len_list.append(adj.shape[0]) # 记录每个cluster的长度，弹出来最长的进行分解

    while len(adj_list)< numbers :
        # 最后生成对应的这个adj_list
        idx = np.argmax(cluser_len_list)
        adj = adj_list[idx]
        ind = index_list[idx]
        #adj_list.remove(adj_list[idx])
        adj_list.pop(idx)
        cluser_len_list.pop(idx)
        #cluser_len_list.remove(cluser_len_list[idx])  # 先弹出来一个，进行分解
        #index_list.remove(index_list[idx])
        index_list.pop(idx)

        conduct = Conductance_Cost(adj, ind)
        frontier, rear,frontier_index,rear_index= conduct.calcurate_condcutance(adj,ind)
        adj_list.append(frontier)
        adj_list.append(rear)
        cluser_len_list.append(len(frontier))
        cluser_len_list.append(len(rear))

        index_list.append(frontier_index)
        index_list.append(rear_index)

    return adj_list, index_list

#  对Aint 进行lift_back 操作，进入到整体的A的纬度。然后反向回来可以生成Aext,再利用Aext 来生成 Acoar
def liftback(adj, C, full_adj):
    affin = np.zeros([full_adj.shape[0], len(C)])
    for i in range(full_adj.shape[0]):
        for j in range(len(C)):
            if C[j] == i:
                affin[i, j] = 1
    Adj_int = affin @ adj @ np.transpose(affin)

    return Adj_int

# 这个代码是生成 sigma(A_int), A_ext = A - sigma(A_int)
def partrtion_Adjaceny(full_adj, adj_list, index_list):
    A_int  = np.zeros([full_adj.shape[0],full_adj.shape[1]])
    for i in range(len(index_list)):
        adj = adj_list[i]
        C = index_list[i]
        Adj_int = liftback(adj, C, full_adj)
        A_int = Adj_int + A_int
    A_ext = full_adj-A_int

    return A_int, A_ext

# 这个映射函数和上面的C是不一样的。这个映射的每一列都是代表一个cluster, 每个列的每一位元素都代表对应的一个节点。
# 所以这个代码就是论文中的P+（也是另外一篇中的S）， 总的1的个数加起来就是整个节点数，比如说6个节点，3个cluster, 1列4个元素，2列为1个元素，3列为1个元素
def Assignment_matrix(index_list, Cluster_number, full_adj):
    affin = np.zeros([full_adj.shape[0], Cluster_number])
    for j in range(Cluster_number):
        for k in index_list[j]:
                affin[k,j] =1
    return affin

# 生成A_coarsen 的函数， A_coar = St * A_ext * S ，其实如果有顺序的话，是可以直接用画图的方式，去掉矩阵中的对角线上的矩阵块来得到的。
def generate_A_coarsen(A_ext,adj_sub_index_list,Cluster_number):
    affin = Assignment_matrix(adj_sub_index_list,Cluster_number,A_ext)
    A_coar = np.transpose(affin) @ A_ext @ affin

    return  A_coar

def Eigen_pooling(sub_graph_adj):
    U_pooling = []
    for adj in sub_graph_adj:
        adj = sp.csr_matrix(adj)
        Li = laplacian(adj, normalized= True)
        lamda_i, U_i= fourier(Li)
        U_pooling.append(U_i)

    return U_pooling




# k = np.random.randint(0,2,size=[20,20])
# k1 = k + np.transpose(k)
# for i in range(k1.shape[0]):
#     k1[i,i] = 0
# adj_list = partrtition(k1,numbers=4)
#
#
# print('debuging...')

#使用三种方式来计算cost, 1, min_conductanse  2, min_varisation_cost 3, spectrum_cluster
def min_conductance_loss():
    cost =0
    return cost
def min_variation_loss():
    cost =0
    return cost
def min_spectrum_loss():
    cost =0
    return cost
def cluster_cost():
    if 'min_conductance':
        conductance_loss = min_conductance_loss()
        return conductance_loss
    elif 'min_variation_cost':
        variation_cost = min_variation_loss()
        return variation_cost
    else:
        spectrum_cost = min_spectrum_loss()
        return spectrum_cost


    # cost function for the edge   这部分就是Lcaol variation的对应的 cost
    # def subgraph_cost(G, A, edge):
    #     edge, w = edge[:2].astype(np.int32), edge[2]
    #     deg_new = 2 * deg[edge] - w
    #     L = np.array([[deg_new[0], -w], [-w, deg_new[1]]])
    #     B = Pibot @ A[edge, :]
    #     return np.linalg.norm(B.T @ L @ B)


def eigen_pooling(G):
    pass

# 如果使用eigen_pooling 可以不使用readout函数，也可以使用GGNN
def readout():
    pass


# datasetname ='mr'
# tf.keras.backend.set_floatx('float32') # 默认在tf 2.x的版本中数据会被转成float32来运行节省空间，但是会导致mutmal出问题，所以这里
#                                        # 直接就把后台所有的类型都禁止广播成32位了。 https://www.mianshigee.com/question/32592ygv/
# train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y = load_data(datasetname)
# adj1 = train_adj[0]
#
# new_adj1 = build_new_graph(adj1)
# s_adj1 = sp.coo_matrix(adj1)
# row = s_adj1.row
# column = s_adj1.col
# # x = []
# # y = []
# # i, j =0,0
# # for i in range(adj1.shape[0]):
# #     for j in range(adj1.shape[1]):
# #         if adj1[i,j]>0:
# #             x.append(i)
# #             y.append(j)
# # plt.scatter(x,y)
# plt.scatter(row,column)
# print("just for test")
