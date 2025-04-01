#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging


from torchvision.models import resnet18, resnet50
from flcore.servers.serveravg import FedAvg


from flcore.trainmodel.models import *
from flcore.trainmodel.custom_resnet18 import *


from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635   #98635 for AG_News and 399198 for Sogou_News
max_len=200
emb_dim=32


import os
import json
import numpy as np
import torch
from torchvision import datasets, transforms
from collections import defaultdict

def load_dataset(dataset_name):
    """Load dataset based on name"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if dataset_name == "MNIST" else 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print(f"{dataset_name} dataset loaded successfully.")
    return train_dataset, test_dataset

def split_dataset_into_tasks(dataset, classes_per_task=2):
    print(f"Splitting dataset into tasks with {classes_per_task} classes per task.")
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    tasks = []
    for i in range(0, num_classes, classes_per_task):
        task_classes = list(range(i, min(i + classes_per_task, num_classes)))
        task_indices = np.where(np.isin(labels, task_classes))[0].tolist()
        tasks.append({"classes": task_classes, "indices": task_indices})
    print(f"Dataset split into {len(tasks)} tasks.")
    
    for task_id, task in enumerate(tasks):
        print(f"Task {task_id} is getting data from classes: {task['classes']}")
    
    return tasks

def distribute_data(tasks, num_clients, alpha, dataset, data_type="train"):
    client_data = {i: [] for i in range(num_clients)}
    label_distribution = {i: defaultdict(int) for i in range(num_clients)}
    print(f"Distributing {data_type} data across {num_clients} clients using Dirichlet distribution (alpha={alpha}).")
    targets = np.array(dataset.targets)
    
    for task_id, task in enumerate(tasks):
        task_classes = task["classes"]
        task_indices = task["indices"]
        np.random.shuffle(task_indices)
        
        dirichlet_distribution = np.random.dirichlet(alpha * np.ones(num_clients), len(task_classes))
        dirichlet_distribution = np.maximum(dirichlet_distribution, 1e-5)
        dirichlet_distribution /= dirichlet_distribution.sum(axis=1, keepdims=True)
        
        client_splits = {i: [] for i in range(num_clients)}
        
        for class_idx, class_label in enumerate(task_classes):
            class_indices = np.where(targets == class_label)[0].tolist()
            proportions = dirichlet_distribution[class_idx]
            
            num_samples = len(class_indices)
            min_samples_per_client = 1
            client_samples = []
            
            remaining_samples = num_samples - (min_samples_per_client * num_clients)
            if remaining_samples < 0:
                for i in range(num_samples):
                    client_samples.append(1)
                for i in range(num_samples, num_clients):
                    client_samples.append(0)
            else:
                client_samples = [min_samples_per_client for _ in range(num_clients)]
                if remaining_samples > 0:
                    remaining_dist = np.random.dirichlet(alpha * np.ones(num_clients))
                    remaining_alloc = (remaining_dist * remaining_samples).astype(int)
                    client_samples = [client_samples[i] + remaining_alloc[i] for i in range(num_clients)]
            
            while sum(client_samples) > num_samples:
                max_client = np.argmax(client_samples)
                if client_samples[max_client] > min_samples_per_client:
                    client_samples[max_client] -= 1
            
            np.random.shuffle(class_indices)
            ptr = 0
            for client_id in range(num_clients):
                num = client_samples[client_id]
                if num > 0:
                    client_splits[client_id].extend(class_indices[ptr:ptr+num])
                    ptr += num
        
        for client_id, indices in client_splits.items():
            client_data[client_id].extend(indices)
            for idx in indices:
                label_distribution[client_id][targets[idx]] += 1
    
    for client_id, dist in label_distribution.items():
        total_samples = sum(dist.values())
        class_dist = {k: v for k, v in dist.items() if v > 0}
        print(f"Client {client_id}: {total_samples} total samples, classes: {class_dist}")
    print(f"{data_type.capitalize()} data distribution completed.")
    return client_data, label_distribution

def save_dataset_structure(task_id, train_client_data, test_client_data, train_dataset, test_dataset, output_dir, alpha, tasks):
    task_folder = os.path.join(output_dir, f"task_{task_id}")
    os.makedirs(task_folder, exist_ok=True)
    
    train_folder = os.path.join(task_folder, "train")
    test_folder = os.path.join(task_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    task_config = {
        "task_id": task_id,
        "num_classes": len(set(train_dataset.targets)),
        "alpha": alpha,
        "classes": tasks[task_id]["classes"],
        "train_data": {},
        "test_data": {}
    }
    
    # Save train data in FLAG-compatible format
    for client_id, indices in train_client_data.items():
        images, labels = [], []
        for idx in indices:
            img, label = train_dataset[idx]
            images.append(img.numpy())
            labels.append(label)
        
        # Create dictionary with 'x' and 'y' keys and wrap in 'data' key
        client_data = {
            "x": np.array(images),
            "y": np.array(labels)
        }
        np.savez(os.path.join(train_folder, f"{client_id}.npz"), data=client_data)
        print(f"{output_dir}")
        print(f"***********saving {client_id}.npz for train**********")
        task_config["train_data"][f"{client_id}"] = len(indices)
    
    # Save test data in FLAG-compatible format
    for client_id, indices in test_client_data.items():
        images, labels = [], []
        for idx in indices:
            img, label = test_dataset[idx]
            images.append(img.numpy())
            labels.append(label)
        
        client_data = {
            "x": np.array(images),
            "y": np.array(labels)
        }
        np.savez(os.path.join(test_folder, f"{client_id}.npz"), data=client_data)
        print(f"*************saving {client_id}.npz for test(********)")
        task_config["test_data"][f"{client_id}"] = len(indices)
    
    with open(os.path.join(task_folder, "config.json"), "w") as f:
        json.dump(task_config, f, indent=4)
    print(f"Task {task_id} dataset saved with config file.")

def prepare_dataset(args):
    output_dir = f"../dataset/{args.dataset}"
    num_clients = args.num_clients
    classes_per_task = args.num_classes
    alpha = args.alpha
    
    # Load dataset
    train_dataset, test_dataset = load_dataset(args.dataset)
    
    # Split into tasks
    tasks = split_dataset_into_tasks(train_dataset, classes_per_task)
    
    # Prepare data for each task
    for task_id, task in enumerate(tasks):
        train_client_data, _ = distribute_data([task], num_clients, alpha, train_dataset, "train")
        test_client_data, _ = distribute_data([task], num_clients, alpha, test_dataset, "test")
        save_dataset_structure(task_id, train_client_data, test_client_data, train_dataset, test_dataset, output_dir, alpha, tasks)

def run(args):
    # First prepare the dataset
    prepare_dataset(args)

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model based on dataset
        if model_str == 'resnet18':
            # Determine in_channels based on dataset
            in_channels = 1 if args.dataset in ["MNIST", "FashionMNIST"] else 3
            args.model = ResNet18_TIL(in_channels=in_channels, 
                                     num_classes_per_task=args.num_classes,
                                     num_tasks=args.tasks)

        else:
            raise NotImplementedError(f"Model {model_str} not implemented")

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.task_heads)  # Copy the task-specific heads
            args.model.task_heads = nn.ModuleList([nn.Identity() for _ in range(args.model.num_tasks)])  # Replace heads with identity layers
            args.model = BaseHeadSplit(args.model, args.head)  # Ensure compatibility
            server = FedAvg(args, i)
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    
    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()

if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="CIFAR10",
                        choices=["CIFAR10", "CIFAR100", "MNIST", "FashionMNIST"])
    parser.add_argument('-tasks', "--tasks", type=int, default=10)
    parser.add_argument('-ppe', "--patterns_per_experience", type=int, default=50)
    parser.add_argument('-ss', "--sample_size", type=int, default=50)
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn",
                        choices=["cnn", "resnet18"])
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")

    parser.add_argument('-alpha', "--alpha", type=float, default=0.1)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    print("=" * 50)

    run(args)
    
    print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
