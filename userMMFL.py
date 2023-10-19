import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import MySGD, FEDLOptimizer
from FLAlgorithms.users.userbase import User
from FLAlgorithms.trainmodel.co_att import CoAttention
# Implementation for Per-FedAvg clients

class UserPerAvg(User):
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate,beta,lamda,
                 local_epochs, optimizer, total_users, num_users):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs)
        self.total_users = total_users
        self.num_users = num_users
        self.co_att = CoAttention(1, 128).cuda()
        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            # self.loss = nn.NLLLoss()
            self.loss = nn.CrossEntropyLoss()
            # print("loss = NLLLoss")
        self.optimizer1 = torch.optim.SGD(self.co_att.parameters(), lr=self.learning_rate)
        self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.fc.Parameter): 

            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):  
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model = self.model.cuda()

        # print("xxxxxxxxxx", next(self.model.parameters()).device)
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):  # local update
            self.model.train()

            #step 1
            X, y = self.get_next_train_batch()

            x1 = X[:, :, 0:1020, :]
            x2 = X[:, :, 1020:2040, :]
            output1, output2 = self.co_att(x1, x2, 1020, 1020)
            output1 = torch.flatten(output1, 1)
            output2 = torch.flatten(output2, 1)
            X = torch.cat([output1, output2], dim=1)
            self.optimizer1.zero_grad()
            self.optimizer.zero_grad()

            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()  # SGD
            self.optimizer1.step()
            #step 2
            X, y = self.get_next_train_batch()

            x1 = X[:, :, 0:1020, :]
            x2 = X[:, :, 1020:2040, :]
            output1, output2 = self.co_att(x1, x2, 1020, 1020)
            output1 = torch.flatten(output1, 1)
            output2 = torch.flatten(output2, 1)
            X = torch.cat([output1, output2], dim=1)
            # X = X.cuda()
            # y = y.cuda()
            self.optimizer.zero_grad()
            self.optimizer1.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step(beta=self.beta)
            self.optimizer1.step()
            # clone model to user model 
            self.clone_model_paramenter(self.model.parameters(), self.local_model)

        return LOSS    

    def train_one_step(self):
        self.model.train()
        #step 1
        # X, y = self.get_next_test_batch()
        X, y = self.get_next_train_batch()

        x1 = X[:, :, 0:1020, :]
        x2 = X[:, :, 1020:2040, :]
        output1, output2 = self.co_att(x1, x2, 1020, 1020)
        output1 = torch.flatten(output1, 1)
        output2 = torch.flatten(output2, 1)
        X = torch.cat([output1, output2], dim=1)

        # X = torch.Tensor(X).type(torch.float32).cuda()
        # y = y.cuda()
        self.optimizer.zero_grad()
        self.optimizer1.zero_grad()
        self.model = self.model.cuda()
        output = self.model(X)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer1.step()
        #step 2
        X, y = self.get_next_train_batch()

        x1 = X[:, :, 0:1020, :]
        x2 = X[:, :, 1020:2040, :]
        output1, output2 = self.co_att(x1, x2, 1020, 1020)
        output1 = torch.flatten(output1, 1)
        output2 = torch.flatten(output2, 1)
        X = torch.cat([output1, output2], dim=1)

        # X = torch.Tensor(X).type(torch.float32).cuda()
        # y = y.cuda()
        self.optimizer.zero_grad()
        self.optimizer1.zero_grad()
        output = self.model(X)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step(beta=self.beta)
        self.optimizer1.step()

