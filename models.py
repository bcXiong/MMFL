import torch
import torch.nn as nn
import torch.nn.functional as F
# from FLAlgorithms.trainmodel.co_att import CoAttention
class Net(nn.Module):
    def __init__(self, input_dim=2040,  mid_dim1=128, output_dim=15):
        super(Net, self).__init__()
        # self.co_att = CoAttention(1, 256)

        self.fc = torch.nn.Sequential(torch.nn.Linear(input_dim, mid_dim1),
                                      nn.ReLU(inplace=True),
                                      torch.nn.Linear(mid_dim1, output_dim))
    def forward(self, x):
        # x1 = x[:, :, 0:1020, :]
        # x2 = x[:, :, 1020:2040, :]
        #
        # output1, output2 = self.co_att(x1, x2, 1020, 1020)
        # output1 = torch.flatten(output1, 1)
        # output2 = torch.flatten(output2, 1)
        # x = torch.cat([output1, output2], dim=1)

        x = torch.flatten(x, 1)

        # x = F.relu(x)
        # print(x.shape)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)

        return output


class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim = 2040, output_dim = 10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class Mclr_CrossEntropy(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(Mclr_CrossEntropy, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

class DNN(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)
        
    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x