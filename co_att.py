# import tensorflow as tf
# from tensorflow.python.keras import layers

import torch
import torch.nn as nn
import torch.nn.functional as F

# def mask_softmax(inputs, mask, dim):
#
#     e_x = tf.exp(inputs-tf.reduce_max(inputs, axis=dim, keepdims=True))
#     mol = e_x * mask
#     dem = tf.reduce_sum(mol, axis=dim, keepdims=True)
#     return mol / dem
#
#
# def self_attention(inputs, channel, sn, activation, name=None):
#
#     with tf.name_scope(name, 'self_attention', [inputs]):
#         f = layers.dense(inputs, channel//2, activation, name='f')
#         g = layers.dense(inputs, channel//2, activation, name='g')
#         h = layers.dense(inputs, channel, activation, name='h')
#
#         s = tf.matmul(g, tf.transpose(f, [0, 2, 1]))
#         beta = tf.nn.softmax(s, -2)
#
#         gamma = tf.get_variable("gamma", [1], tf.float64, initializer=tf.constant_initializer(0.0))
#
#         o = tf.matmul(beta, h,)
#
#         x = gamma * o + inputs
#
#     return x


# def co_attention(inputs, num_q, num_v, units, activation=tf.nn.tanh):
#
#     """
#     @inproceedings{lu2016hierarchical,
#         title={Hierarchical question-image co-attention for visual question answering},
#         author={Lu, Jiasen and Yang, Jianwei and Batra, Dhruv and Parikh, Devi},
#         booktitle={Advances In Neural Information Processing Systems},
#         pages={289--297},
#         year={2016}
#     }
#     """
#
#     # shape = inputs.get_shape().as_list()
#     shape = list(inputs.shape)
#     dim = len(shape)
#     # dim = list(range(len(shape)))
#
#     modality_q = tf.slice(inputs, [0]*dim, [-1]*(dim-2)+[num_q, -1])
#     modality_v = tf.slice(inputs, [0]*(dim-2)+[num_q, 0], [-1]*dim)
#
#     correlation_matrix = tf.matmul(layers.dense(modality_q, shape[-1], use_bias=False), modality_v, transpose_b=True)
#     if activation:
#         correlation_matrix = activation(correlation_matrix)
#
#     modality_q_transform = layers.dense(modality_q, units, use_bias=False)
#     modality_v_transform = layers.dense(modality_v, units, use_bias=False)
#
#     attention_q = modality_q_transform + tf.matmul(correlation_matrix, modality_v_transform)
#     attention_v = modality_v_transform + tf.matmul(correlation_matrix, modality_q_transform, transpose_a=True)
#     if activation:
#         attention_q = activation(attention_q)
#         attention_v = activation(attention_v)
#
#     modality_q = tf.nn.softmax(layers.dense(attention_q, 1, use_bias=False), -2) * modality_q * num_q
#     modality_v = tf.nn.softmax(layers.dense(attention_v, 1, use_bias=False), -2) * modality_v * num_v
#
#     return tf.concat([modality_q, modality_v], -2)

class CoAttention(nn.Module):
    def __init__(self, input_feature_size, hidden_dim):
        super(CoAttention, self).__init__()
        self.line1 = nn.Linear(in_features=input_feature_size, out_features=input_feature_size, bias=False)
        self.line2_v = nn.Linear(input_feature_size, hidden_dim, bias=False)
        self.line2_s = nn.Linear(input_feature_size, hidden_dim, bias=False)
        self.line3_v = nn.Linear(hidden_dim, 1, bias=False)
        self.line3_s = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, modality_s, modality_v, num_s, num_v):
        correlation_matrix = torch.matmul(self.line1(modality_s), modality_v.permute(0,1,3,2))
        correlation_matrix = torch.tanh(correlation_matrix)

        modality_s_transform = self.line2_s(modality_s)
        modality_v_transform = self.line2_v(modality_v)

        attention_s = modality_s_transform + torch.matmul(correlation_matrix, modality_v_transform)
        attention_v = modality_v_transform + torch.matmul(correlation_matrix.permute(0,1,3,2), modality_s_transform)


        attention_s = torch.tanh(attention_s)
        attention_v = torch.tanh(attention_v)
        attention_s = torch.softmax(self.line3_s(attention_s), dim=-2)
        attention_v = torch.softmax(self.line3_v(attention_v), dim=-2)
        modality_s = attention_s*modality_s*num_s
        modality_v = attention_v*modality_v*num_v
        return modality_s, modality_v

# def co_attention_2(input_q, input_v, num_q, num_v, units, activation=tf.nn.tanh):
#
#     """
#     @inproceedings{lu2016hierarchical,
#         title={Hierarchical question-image co-attention for visual question answering},
#         author={Lu, Jiasen and Yang, Jianwei and Batra, Dhruv and Parikh, Devi},
#         booktitle={Advances In Neural Information Processing Systems},
#         pages={289--297},
#         year={2016}
#     }
#     """
#
#     shape = input_v.get_shape().as_list()
#
#     correlation_matrix = tf.matmul(layers.dense(input_q, shape[-1], use_bias=False), input_v, transpose_b=True)
#     if activation:
#         correlation_matrix = activation(correlation_matrix)
#
#     modality_q_transform = layers.dense(input_q, units, use_bias=False)
#     modality_v_transform = layers.dense(input_v, units, use_bias=False)
#
#     attention_q = modality_q_transform + tf.matmul(correlation_matrix, modality_v_transform)
#     attention_v = modality_v_transform + tf.matmul(correlation_matrix, modality_q_transform, transpose_a=True)
#     if activation:
#         attention_q = activation(attention_q)
#         attention_v = activation(attention_v)
#
#     modality_q = tf.nn.softmax(layers.dense(attention_q, 1, use_bias=False), -2) * input_q * num_q
#     modality_v = tf.nn.softmax(layers.dense(attention_v, 1, use_bias=False), -2) * input_v * num_v
#
#     return tf.concat([modality_q, modality_v], -2)

# if __name__ == "__main__":
#     x1 = torch.randn([20,1,1,1024])
#     x2 = torch.randn([20,1,1,1025])
#     x1 = x1.permute(0, 1, 3, 2)
#     x2 = x2.permute(0, 1, 3, 2)
#     co_att = CoAttention(1, 256)
#     output1, output2 = co_att(x1, x2, 1024, 1025, True)

    # output1 = torch.flatten(output1, 1)
    # output2 = torch.flatten(output2, 1)
    # x = torch.cat([output1, output2], dim=1)
    # # print(x.shape)
    # print(output1.shape)
    # print(output2.shape)

    # a = [[2],[3],[4]]
    # b = [[2,2], [2,2], [2,2]]
    # a = torch.tensor(a)
    # b = torch.tensor(b)
    # c = a*b
    # print(c)
    # d = c*2
    # print(d)