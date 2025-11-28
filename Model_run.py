import time
# 代码运行时需要设置的参数：dataset = mr,
# preprocess_adj(adj)： cluster number
#

import tensorflow as tf

from model import *
import pickle as pkl
from utilis import *

# ablation conclusion
# 1, 必须使用adj, 会比不使用adj提升5%左右（纯readout,74%,带 adj 79%)
# 2, 使用的adj 是 A*A。最好测试一下 A + A*A。
# 3，pooling 默认并不会改变太多精度。但是可以缩短运算时间。且 A*A 的 效果要比A的效果好

##### 训练的一些技巧，大batch 其收敛慢，稳定性好点。大batch 配大学习率（batch_size是更新梯度时候的分母，学习率是分子）
#
######



datasetname ='ohsumed'
tf.keras.backend.set_floatx('float32') # 默认在tf 2.x的版本中数据会被转成float32来运行节省空间，但是会导致mutmal出问题，所以这里
                                       # 直接就把后台所有的类型都禁止广播成32位了。 https://www.mianshigee.com/question/32592ygv/
names1 = ['x_adj', 'x_embed', 'y', 'valx_adj', 'valx_embed', 'valy', 'tx_adj', 'tx_embed', 'ty']

# # 处理 cluster padding
# test_adj1, test_mask = preprocess_adj(test_adj)
# val_adj, val_mask = preprocess_adj(val_adj)
# train_adj, train_mask = preprocess_adj(train_adj)

names2 = ['V_clusters', 'V_Adj', 'V_Masks', 'val_V_clusters', 'val_V_Adj', 'val_V_Masks', 't_V_clusters', 't_V_Adj', 't_V_Masks']
names3 = ['token_embed', 'token_adj', 'token_M', 'val_token_embed', 'val_token_adj', 'val_token_M', 't_token_embed', 't_token_adj', 't_token_M']

#train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y = tuple(new_load_data(datasetname, names1))
train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y = load_data(datasetname, names1)

token_embed_list, token_adj_list, token_M_list, val_token_embed_list, val_token_adj_list, val_token_M_list, t_token_embed_list, t_token_adj_list, t_token_M_list = tuple( new_load_data(datasetname, names3))

V_clusters, V_Adj, V_Masks, val_V_clusters, val_V_Adj, val_V_Masks, t_V_clusters, t_V_Adj, t_V_Masks = tuple(new_load_data(datasetname, names2))

A_coar,val_A_coar, t_A_coar = tuple(new_load_data(datasetname, ['A_coar', 'val_A_coar', 't_A_coar']))
print('all the data for the model are loaded......')





# max_len 1 for text, max_len 2 for cluster
max_length, = tuple(new_load_data(datasetname, ['max_length']))
train_feature, node_mask, pad_index = pad_node_features(train_feature, max_length[0])
#train_adj = simple_pad_adjs(train_adj, max_length[0])
train_adj = preprocess_adj(train_adj, max_length[0])
# token_embed, node_mask, pad_index = pad_node_features(token_embed_list, max_length[2])
# token_adj = simple_pad_adjs(token_adj_list, max_length[2])

token_M = pad_M(token_M_list,max_length[2],max_length[0]) # transfer word to token
V_Adj, V_Masks = preprocess_cluster_adj(V_Adj, max_length[1]) #
# V_Masks = simple_pad_cluster_list(V_Masks, max_length[1])  # 如果使用Cluster 做GRU，则使用这个V_masks
# V_Masks = np.array(V_Masks, dtype=np.float32)   # 请把这个留在这个地方，因为下面的 V_clusters 也使用到了这个函数，这个是float32, 下面的是 int
V_clusters = simple_pad_cluster_list(V_clusters,max_length[1])
A_coar = np.array(A_coar, dtype=np.float32)
A_coar = preprocess_adj(A_coar, A_coar.shape[-1]) # 其参与了GCN，因此也需要进行处理。也可以在生成的时候进行处理。但是建议这里处理比较合适


# preprocess data for val
val_feature, _, _ = pad_node_features(val_feature, max_length[0])
val_adj =  preprocess_adj(val_adj, max_length[0])
val_A_coar = np.array(val_A_coar, dtype=np.float32)
val_A_coar = preprocess_adj(val_A_coar, A_coar.shape[-1])
val_token_M = pad_M(val_token_M_list,max_length[2],max_length[0]) # transfer word to token
val_V_clusters = simple_pad_cluster_list(val_V_clusters,max_length[1])
val_V_Adj, val_V_Masks =  preprocess_cluster_adj(val_V_Adj, max_length[1])
# val_V_Masks =  simple_pad_cluster_list(val_V_Masks, max_length[1])
# val_V_Masks = np.array(val_V_Masks, dtype=np.float32)

# preprocess data for test

test_feature, _, _ = pad_node_features(test_feature, max_length[0])
test_adj =  preprocess_adj(test_adj, max_length[0])
t_A_coar = np.array(t_A_coar, dtype=np.float32)
t_A_coar = preprocess_adj(t_A_coar, A_coar.shape[-1])
t_token_M = pad_M(t_token_M_list,max_length[2],max_length[0]) # transfer word to token
t_V_clusters = simple_pad_cluster_list(t_V_clusters,max_length[1])
t_V_Adj,t_V_Masks  =  preprocess_cluster_adj(t_V_Adj, max_length[1])
# t_V_Masks =  simple_pad_cluster_list(t_V_Masks, max_length[1])
# t_V_Masks = np.array(t_V_Masks, dtype=np.float32)

# impact the data to data_set, prepare training.

data_set = tf.data.Dataset.from_tensor_slices((train_feature, train_adj , A_coar, token_M, V_clusters, V_Adj, V_Masks, train_y)).batch(batch_size=64,drop_remainder=True)
model1 = GNN_model(input_dim=300, output_dim=23, clusters_per_level=[8])


def accuracy_function(real, pred):
    accuracies = tf.equal(tf.argmax(real, axis=-1), tf.argmax(pred, axis=-1))
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    return accuracies

def loss_function(real, pred):
    loss_ = tf.keras.losses.categorical_crossentropy(real,pred)
    return tf.reduce_sum(loss_)/real.shape[0]

class dynamic_learning_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,):
        super(dynamic_learning_schedule, self).__init__()
      #self.epochs = epoch
    def __call__(self, step):
        lr =0.0003
        warmup_steps = 1000
        if step > 999 :
            if step % 20 == 0 :
                lr = warmup_steps/step * lr
                print('now, the step is:', step,'   and the learning rate is',lr)
        return lr

l_rate = dynamic_learning_schedule()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')
loss_object = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=True, reduction='none')

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate= 0.06 )
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')

def train_step(inp, support, A_coar, token_M, V_clusters, V_adj, mask, tar ):
    with tf.GradientTape() as tape:
        predictions = model1(inp, support, A_coar, token_M, V_clusters, V_adj, mask, training= True)
        #model1.summary()
        loss = loss_function(tar, predictions)
    gradients = tape.gradient(loss, model1.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model1.trainable_variables))
    train_loss(loss)
    train_accuracy(accuracy_function(tar, predictions))
    print([tf.reduce_sum(tf.abs(g)) for g in gradients])
    print("logits mean/std:",
          tf.reduce_mean(predictions).numpy(),
          tf.math.reduce_std(predictions).numpy())


EPOCHS =41
for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    print('runing in epoch:', epoch)
    batch = 0
    for data_slice in data_set:
        batch = batch + 1
        inp = data_slice[0]
        support = data_slice[1]
        A_coar = data_slice[2]
        token_M = data_slice[3]
        V_clusters = data_slice[4]
        V_adj = data_slice[5]
        mask = data_slice[6]
        tar = data_slice[7]
        train_step(inp, support, A_coar, token_M, V_clusters, V_adj, mask, tar)
        if batch % 1 == 0:
            print(
                f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    Flag = True

    # Val 部分还是要的
    val = model1(val_feature, val_adj, val_A_coar, val_token_M, val_V_clusters, val_V_Adj, val_V_Masks, training= False )
    val_loss = loss_function(val_y, val)
    val_acc = val_accuracy(accuracy_function(val_y, val))
    print('............the validation loss is...........', val_loss, 'and the val acc is', val_acc)

    if (epoch + 1) % 8 == 0:
        test = model1(test_feature, test_adj, t_A_coar, t_token_M, t_V_clusters, t_V_Adj, t_V_Masks, training= False )
        print('_________________the test accuracy is____________', test_accuracy(accuracy_function(test_y, test)))
    #

    if (epoch + 1) % 6 == 0:
      ckpt_save_path = ckpt_manager.save()
      print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

