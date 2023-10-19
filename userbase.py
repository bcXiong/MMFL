# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from sklearn.metrics import confusion_matrix
from FLAlgorithms.trainmodel.co_att import CoAttention
class User:
    """
    Base class for users in federated learning.
    """

    def __init__(self, id, train_data, test_data, model, batch_size=0, learning_rate=0, beta=0, lamda=0,
                 local_epochs=0):
        # from fedprox
        self.model = copy.deepcopy(model)  # deepcopy深拷贝，后续再做改动时也不再变化，copy浅拷贝，后续改动可能有变化
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size, shuffle=True)
        self.testloader = DataLoader(test_data, self.batch_size, shuffle=True)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.co_att = CoAttention(1, 128).cuda()

    def set_parameters(self, model):

        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):

            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()

        # self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()  # detach就是截断反向传播的梯度流
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()  # .eval()输入结果就不会是字符串类型的了
        test_acc = 0

        y_pred = 0
        # for x, y in self.testloaderfull:
        for x, y in self.testloader:
            x = x.cuda()
            y = y.cuda()

            x1 = x[:, :, 0:1020, :]
            x2 = x[:, :, 1020:2040, :]
            output1, output2 = self.co_att(x1, x2, 1020, 1020)
            output1 = torch.flatten(output1, 1)
            output2 = torch.flatten(output2, 1)
            x = torch.cat([output1, output2], dim=1)

            output = self.model(x)
            test_acc = test_acc + (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # y_pred += np.argmax(output.cpu().detach().numpy())
            # result = confusion_matrix(y, y_pred)
            # print(result)
        # y = a*self.batch_size
            # print(num_sample)
            # print(test_acc, y.shape[0])
            # dim的不同值表示不同维度，特别的在dim=0表示二维中的行，dim=1在二维矩阵中表示列
            # @loss += self.loss(output, y)
            # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            # print(self.id + ", Test Loss:", loss)
        # print(test_acc, self.test_samples, 1)
        return test_acc, self.test_samples

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        # for x, y in self.trainloaderfull:  # error

        for x, y in self.trainloader:
            x = x.cuda()
            y = y.cuda()

            x1 = x[:, :, 0:1020, :]
            x2 = x[:, :, 1020:2040, :]
            output1, output2 = self.co_att(x1, x2, 1020, 1020)
            output1 = torch.flatten(output1, 1)
            output2 = torch.flatten(output2, 1)
            x = torch.cat([output1, output2], dim=1)

            output = self.model(x)

            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()  # .item()返回的是一个具体的数值且改变tpye

            loss = self.loss(output, y)
            # a = loss.cpu
            loss += (torch.sum(loss)).item()
            # if a>50:
            #     break

            # result = torch.sum(confusion_matrix(y, output)).item()
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)

        return train_acc, loss, self.train_samples

    def test_persionalized_model(self):
        self.model.eval()
        test_acc = 0
        self.update_parameters(self.persionalized_model_bar)
        # for x, y in self.testloaderfull:

        for x, y in self.testloader:
            x = x.cuda()
            y = y.cuda()
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # @loss += self.loss(output, y)
            # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            # print(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0]

    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(self.persionalized_model_bar)
        # for x, y in self.trainloaderfull:
        self.trainloader = self.trainloader.cuda()
        for x, y in self.trainloader:
            x = x.cuda()
            y = y.cuda()
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model)
        return train_acc, loss, self.train_samples

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)  # 指针指向下一条记录
            X = X.cuda()
            y = y.cuda()
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
            X = X.cuda()
            y = y.cuda()
        return (X, y)

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
            # print(X.shape)
            # print(y)
            X = X.cuda()
            y = y.cuda()

        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
            X = X.cuda()
            y = y.cuda()
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))  # 存在返回true
