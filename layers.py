
import tensorflow as tf
import  numpy as np



class GCN(tf.keras.layers.Layer):
    def __init__(self,units,hidden,outs):
        super(GCN, self).__init__()
        self.units = units
        self.hidden = hidden
        self.outs = outs
        self.act = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(0.5,)

    def build(self,input_shape):
        self.weight = self.add_weight(name='gcn',shape=(input_shape[-1],self.hidden),regularizer=tf.keras.regularizers.L2(0.01),initializer=tf.initializers.glorot_normal )

        self.bias = self.add_weight(shape=(self.hidden,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        name='bias_value' )

    def call(self,inputs, support, training= False):
        adj = tf.cast(support, dtype=tf.float32)
        inputs = self.dropout(inputs,training= False)  ###从训练的角度，dropout还是起作用的，但是要想办法用training参数来控制住。
        out = tf.matmul(adj, tf.matmul(inputs,self.weight)) ## 乘法的顺序没有什么意义。只要是按照公式，结果都一样
        out = out + self.bias
        out = self.act(out)
        return out

class ReadoutLayer(tf.keras.layers.Layer):
    """Graph Readout Layer."""

    def __init__(self,  input_dim, output_dim, flag = 'Final', bias=False, **kwargs):
        super(ReadoutLayer, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.act = tf.keras.layers.ReLU()
        self.Flag = flag
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.softmax = tf.keras.layers.Softmax()
        self.hidden = output_dim
        self.layernorm1 = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        self.final = self.add_weight(shape=(self.input_dim , self.hidden),
                                     initializer=tf.keras.initializers.glorot_uniform(),
                                     name='kernal', )
        self.bias_final = self.add_weight(shape=(self.hidden),
                                          initializer=tf.keras.initializers.Zeros(),
                                          name='bias_value')

    def call(self, inputs):  # mask here is also a input
        inputs = self.dropout(inputs)
        # ###从输出的角度来看，这个纬度变换，必须要走下面这一步，变成了 batch * output_dim
        output = tf.reduce_mean(inputs, axis=-2) + tf.reduce_max(inputs, axis=-2)
        output = tf.matmul(output, self.final)  + self.bias_final
        output = self.act(output)
       # output = self.layernorm1(output)
        if self.Flag == 'Final':
            output = self.softmax(output)
        return output


# #

# the gru for the nodes update
class GGCN_unit(tf.keras.layers.Layer):
    def __init__(self,input_dim,output_dim):
        super(GGCN_unit, self).__init__()
        self.input_dim = input_dim # the input_dim not be used.
        self.output_dim = output_dim
        self.act = tf.keras.layers.ReLU()
        self.batch_normal = tf.keras.layers.BatchNormalization()
        #self.act1 = tf.math.tanh()
        #self.dropout = tf.keras.layers.Dropout(rate=0.2)

    def build(self, input_shape,):
        self.weight_z0 = self.add_weight(shape=[self.output_dim,self.output_dim],name = 'weight_z0 ',
                                         initializer=tf.keras.initializers.glorot_uniform)
        self.weight_z1 = self.add_weight(shape=[self.output_dim, self.output_dim],name = 'weight_z1 ',
                                         initializer=tf.keras.initializers.glorot_uniform)
        self.weight_r0 = self.add_weight(shape=[self.output_dim,self.output_dim],name = 'weight_r0 ',
                                         initializer=tf.keras.initializers.glorot_uniform)
        self.weight_r1 = self.add_weight(shape=[self.output_dim, self.output_dim],name = 'weight_r1 ',
                                         initializer=tf.keras.initializers.glorot_uniform)
        self.weight_h0 = self.add_weight(shape=[self.output_dim,self.output_dim],name = 'weight_h0 ',
                                         initializer=tf.keras.initializers.glorot_uniform)
        self.weight_h1 = self.add_weight(shape=[self.output_dim, self.output_dim],name = 'weight_h1 ',
                                         initializer=tf.keras.initializers.glorot_uniform)

        self.bais_z0 = self.add_weight(shape=self.output_dim, name='bais_z0 ',initializer=tf.keras.initializers.Zeros)
        self.bais_z1 = self.add_weight(shape=self.output_dim, name='bais_z1 ',initializer=tf.keras.initializers.Zeros)
        self.bais_r0 = self.add_weight(shape=self.output_dim, name='bais_r0 ',initializer=tf.keras.initializers.Zeros)
        self.bais_r1 = self.add_weight(shape=self.output_dim, name='bais_r1 ',initializer=tf.keras.initializers.Zeros)
        self.bais_h0 = self.add_weight(shape=self.output_dim, name='bais_h0 ',initializer=tf.keras.initializers.Zeros)
        self.bais_h1 = self.add_weight(shape=self.output_dim, name='bais_h1 ',initializer=tf.keras.initializers.Zeros)
    def call(self, x,support,sparse_inputs=False):
        support = tf.cast(support, dtype=tf.float32)
        #x = self.dropout(x) # optional
        a = tf.matmul(support, x)
        # update gate
        z0 = tf.matmul(a, self.weight_z0) + self.bais_z0
        z1 = tf.matmul(x, self.weight_z1) + self.bais_z1
        z = tf.sigmoid(z0 + z1)
        # reset gate
        r0 = tf.matmul(a, self.weight_r0) + self.bais_r0
        r1 = tf.matmul(x, self.weight_r1) + self.bais_r1
        r = tf.sigmoid(r0 + r1)
        # update embeddings
        h0 = tf.matmul(a, self.weight_h0) + self.bais_h0
        h1 = tf.matmul(r * x, self.weight_h1) + self.bais_h1
        #h = self.act(mask * (h0 + h1))
        h = self.act(h0 + h1)
        #h = tf.math.tanh(h0 + h1)
        gru_output = h * z + x * (1 - z)
        #gru_output = self.batch_normal(gru_output)

        return gru_output


class Att_Merge(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(Att_Merge, self).__init__()
        self.Flag = 'Max'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = tf.keras.layers.ReLU()


    def build(self, input_shape):

        self.weights_att = self.add_weight(shape=[self.input_dim, 1],name = 'weights_att ',
                                         initializer=tf.keras.initializers.glorot_uniform)
        self.weights_emb = self.add_weight(shape=[self.input_dim, self.input_dim],name = 'weights_emb ',
                                         initializer=tf.keras.initializers.glorot_uniform)
        # self.weights_mlp = self.add_weight(shape=[self.input_dim, self.output_dim],name = 'weights_mlp ',
        #                                  initializer=tf.keras.initializers.glorot_uniform)

        self.bias_att = self.add_weight(shape=[1], initializer=tf.keras.initializers.Zeros(), name='bias_att')
        self.bias_emb = self.add_weight(shape=[self.input_dim], initializer=tf.keras.initializers.Zeros(), name='bias_emb')


    def call(self, inputs, mask):

        # mask padding nodes
        mask = tf.cast(mask,dtype=tf.float32)

        # soft attention
        att = tf.sigmoid(tf.matmul(inputs, self.weights_att) + self.bias_att)
        emb = self.act(tf.matmul(inputs, self.weights_emb) + self.bias_emb)
        if len(mask.get_shape()) < len(emb.get_shape()):
            mask = tf.expand_dims(mask, -1)
        N = tf.reduce_sum(mask, axis=-2)
        N = tf.maximum(N, 1.0)
        M = (mask - 1) * 1e9
        # graph summation
        g = mask * att * emb
        g = tf.reduce_sum(g, axis=-2)/ N + tf.reduce_max(g + M, axis=-2)  #该代码是取sum 和 max 的融合了。简单的attention的话，就可以先不这么操作。

        return g


class GGNNGRUBlock(tf.keras.layers.Layer):
    """
    A single GGNN → GRU processing block.
    Used for one cluster. Parameters are NOT shared across clusters.
    """

    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)

        # GGCN (GGNN) message passing layer
        self.ggnn = GGCN_unit(
            input_dim=input_dim,
            output_dim=hidden_dim
        )
        # GRU for sequential/state modeling

        self.gru = tf.keras.layers.GRU(
            hidden_dim,
            return_sequences=True,
            return_state=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            #recurrent_clipnorm=1.0
        )
    def call(self, inputs,cluster_nodes, adj_v, mask, training = True):

        # 1. GGNN propagation
        #output = tf.gather(inputs, cluster_nodes, axis=1, batch_dims=1)

        output = self.ggnn(inputs, adj_v, training= training)
        B, N, M, D = output.shape
        output = tf.reshape(output, (B * N, M, D))
        mask = tf.reshape(mask, [B * N, M])
        mask = tf.cast(mask, tf.bool)

        # 2. GRU encoding (treat node dimension as sequence if needed)
        seq_output, last_state = self.gru(output, mask =mask, training= training)
        seq_output = tf.reshape(seq_output, (B , N, M, D))
        return seq_output, last_state


class ReadoutLayer_test(tf.keras.layers.Layer):
    """Graph Readout Layer."""

    def __init__(self,  input_dim, output_dim, flag = 'Final', bias=False, **kwargs):
        super(ReadoutLayer, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.act = tf.keras.layers.ReLU()
        self.Flag = flag
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.softmax = tf.keras.layers.Softmax()
        self.hidden = output_dim
        self.layernorm1 = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        self.final = self.add_weight(shape=(self.input_dim , self.hidden),
                                     initializer=tf.keras.initializers.glorot_uniform(),
                                     name='kernal', )
        self.bias_final = self.add_weight(shape=(self.hidden),
                                          initializer=tf.keras.initializers.Zeros(),
                                          name='bias_value')

    def call(self, inputs):  # mask here is also a input
        #inputs = tf.keras.layers.Dropout(0.5)(inputs)
        # ###从输出的角度来看，这个纬度变换，必须要走下面这一步，变成了 batch * output_dim
        output = tf.reduce_mean(inputs, axis=-2) + tf.reduce_max(inputs, axis=-2)
        output = tf.matmul(output, self.final)  + self.bias_final
        output = self.act(output)
       # output = self.layernorm1(output)
        if self.Flag == 'Final':
            output = tf.keras.layers.Softmax()(output)
        return output






class Att_Merge_GAT(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(Att_Merge_GAT, self).__init__()
        self.Flag = 'Max'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = tf.keras.layers.ReLU()


    def build(self, input_shape):

        self.weights_att = self.add_weight(shape=[input_shape[-2], 1],name = 'weights_att ',
                                         initializer=tf.keras.initializers.glorot_uniform)
        self.weights_emb = self.add_weight(shape=[self.input_dim, self.input_dim],name = 'weights_emb ',
                                         initializer=tf.keras.initializers.glorot_uniform)

        # self.bias_att = self.add_weight(shape=[1], initializer=tf.keras.initializers.Zeros(), name='bias_att')
        # self.bias_emb = self.add_weight(shape=[self.input_dim], initializer=tf.keras.initializers.Zeros(), name='bias_emb')

    def call(self, inputs, mask):
        # inputs: [B, K, D]
        # mask:   [B, K]

        mask = tf.cast(mask, tf.float32)  # [B, K]
        mask_exp = tf.expand_dims(mask, -1)  # [B, K, 1]

        # ----- 1. linear transform -----
        Wh = tf.matmul(inputs, self.weights_emb)  # [B, K, F]

        # ----- 2. attention score e_i -----
        # a: [F], but use as [1,1,F]
        e = tf.nn.leaky_relu(tf.reduce_sum( self.weights_att * Wh, axis=-1))  # [B, K]

        # ----- 3. mask padding to -inf -----
        minus_inf = (1.0 - mask) * (-1e9)  # [B, K]
        e = e + minus_inf  # padding → -1e9

        # ----- 4. softmax attention α_i -----
        alpha = tf.nn.softmax(e, axis=-1)  # [B, K]
        alpha = tf.expand_dims(alpha, -1)  # [B, K, 1]

        # ----- 5. weighted sum -----
        gat_pool = tf.reduce_sum(alpha * Wh, axis=-2)  # [B, F]

        # ----- 6. max pooling (masked) -----
        max_pool = tf.reduce_max(Wh + minus_inf[:, :, :, None], axis=-2)  # [B, F]

        # ----- 7. fusion -----
        g = gat_pool + max_pool  # [B, F]

        return g

