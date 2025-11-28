

import tensorflow as tf
tf.keras.backend.floatx()

import layers as new_layer

class GNN_model(tf.keras.models.Model):
    def __init__(self,  input_dim, output_dim, clusters_per_level, **kwargs):
        super(GNN_model, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim =300
        #self.classes =23    # change it based on the datasets
        self.classes = output_dim
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dropout1 = tf.keras.layers.Dropout(0.5)

        self.num_levels = len(clusters_per_level)
        self.clusters_num_per_level = clusters_per_level

        self.GCN_layers = tuple(
            new_layer.GCN(units=300, hidden=300, outs=300)
            for _ in clusters_per_level
        )
        self.GCN_layers_final = new_layer.GCN(units=300, hidden=300, outs = 300)

        self.GGNN_GRU_blocks = tuple(
            new_layer.GGNNGRUBlock(input_dim=input_dim, hidden_dim=300)
            for _ in clusters_per_level
        )

        self.Att_Merge_layers = tuple(
            new_layer.Att_Merge(input_dim=600, output_dim=self.hidden_dim)
            for _ in clusters_per_level
        )
        # norm and dense for overfit and unfit.
        self.Dense = tf.keras.layers.Dense(self.hidden_dim)
        self.L_normal = tf.keras.layers.LayerNormalization()
        self.L_normal1 = tf.keras.layers.LayerNormalization()
        self.L_normal2 = tf.keras.layers.LayerNormalization()


        self.B_normal = tf.keras.layers.BatchNormalization()
        # self.B_normal1 = tf.keras.layers.BatchNormalization()

        self.readout_final = new_layer.ReadoutLayer(input_dim=self.input_dim, output_dim= self.classes, flag = 'Final')
        self.iterator_levels = 1


    def call(self, inputs, support, A_coar, M, V_clusters, V_Adj, V_mask, training= True ):
        output = self.dropout(inputs)
        A = tf.convert_to_tensor(support)
        A_coar = tf.convert_to_tensor(A_coar)
        #for iterator_times in range(self.iterator_levels):
        for level in range(self.num_levels):
            cluster_nodes = V_clusters[level]

            # The whole graph was handled by GCN
            output1 = self.GCN_layers[level](output,A)
            output1 =  self.L_normal(output1)

            output1 = tf.matmul(M, output1) # transfor to token graph

            output1 = self.Dense(output1)  # add Dense for overfit
            output1 = self.L_normal1(output1)  # for overfit LN

            # After assignment, the clusters are handled with the GGCN-GRU block
            output1 = tf.gather(output1, cluster_nodes , axis=1, batch_dims=-2)
            output2, _ = self.GGNN_GRU_blocks[level](output1, cluster_nodes, V_Adj, V_mask, training = training)

            # Graph merge with the Graph attention merge module
            #output2 = tf.gather(output1, cluster_nodes, axis=1, batch_dims=-2)
            output2 = tf.concat([output1, output2], axis=-1)
            output = self.Att_Merge_layers[level](output2, V_mask) # 这里是论文中的 Merging 部分

        output = self.GCN_layers_final(output, A_coar)
        output = self.B_normal(output)
        output = self.readout_final(output) # 注意，这里是 Readout 函数了。

        return output



