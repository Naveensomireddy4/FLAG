# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import torch
import torch.nn as nn
import numpy as np
import os
import math
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        self.classes_per_task = args.num_classes
        self.round_per_task = math.floor((args.global_rounds+1)/args.tasks)
        #print(f"in clientbase {self.classes_per_task}  {self.round_per_task} ")

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None,round=0):
        if batch_size == None:
            batch_size = self.batch_size
        #datasets=["task_0","task_1","task_2","task_3","task_4"]
        idx = math.floor(round)
        task_folder = f"task_{idx}"
        # Set self.dataset to include the full path
        dataset = os.path.join(self.dataset, task_folder) 
        #self.dataset = datasets[idx]
        # print("Constructed dataset path for loading train data:", dataset)
        train_data = read_client_data(dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None,round =0):
        if batch_size == None:
            batch_size = self.batch_size
        #datasets=["task_0","task_1","task_2","task_3","task_4"]
        idx = math.floor(round)
        task_folder = f"task_{idx}"
        # Set self.dataset to include the full path
        dataset = os.path.join(self.dataset, task_folder) 
        #self.dataset = datasets[idx]    
        # print("Constructed dataset path for loading test data:", dataset)
        test_data = read_client_data(dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self,round ):
        self.round = round
        # this loads particualr dataset for that task
        testloaderfull = self.load_test_data(round = round)
        #print(f"test_loader_length_{len(testloaderfull)}")
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
    

        with torch.no_grad():
            for x, y in testloaderfull:
                # print("\n🔹 Raw y values:", y)  # Print y directly
                # print("🔹 Unique values in y:", np.unique(y.detach().cpu().numpy()))  # Check unique labels

                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                task_id = math.floor(round)
                y %=  self.classes_per_task
                # task_id helps to finding repective head
                output = self.model(x, task_id=math.floor(round))

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())

                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1

                # Ensure `y` is correctly shaped before binarization
                lb = label_binarize(y.detach().cpu().numpy().reshape(-1), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]

                y_true.append(lb.tolist())  # Convert to list to maintain consistent shape

        y_prob = np.concatenate(y_prob, axis=0)  # Convert list of arrays → single array
        y_true = np.concatenate(y_true, axis=0)  # Convert list of arrays → single array

        y_prob = torch.softmax(torch.tensor(y_prob), dim=1).numpy()

        #Ensure y_prob has the correct shape
        # print("\n🔹 y_true Shape:", y_true.shape)
        # print("🔹 y_prob Shape:", y_prob.shape)

        # print("\n🔹 Unique values in y_true:", np.unique(y_true))
        # print("🔹 Sum of each row in y_true:", np.sum(y_true, axis=1))

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro', multi_class='ovr')

        
        return test_acc, test_num, auc

    def train_metrics(self,round):
        self.round =round 
        trainloader = self.load_train_data(round = round)
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                task_id = math.floor(round)
                # print(f"Round: {round}, Task ID: {math.floor(round)}")
                # print(f"Input x shape: {x.shape}")

                output = self.model(x, task_id=math.floor(round))
                # print(f"Model output shape: {output.shape}")  # Expected: (batch_size, num_classes_per_task)

                # print(f"Before modification: y shape = {y.shape}, y min = {y.min()}, y max = {y.max()}")

                y %=  self.classes_per_task

                #print(f"After modification: y shape = {y.shape}, y min = {y.min()}, y max = {y.max()}")

                #print(f"Loss function input: output shape = {output.shape}, y shape = {y.shape}")

                loss = self.loss(output, y)  # 🔴 Potential crash point!

                #print(f"Loss computed: {loss.item()}")  # If it crashes before this, y is out of range!

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
