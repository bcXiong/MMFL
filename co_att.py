import torch
import torch.nn as nn
import torch.nn.functional as F




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

