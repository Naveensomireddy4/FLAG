import os
import json
import numpy as np
import torch
from torchvision import datasets, transforms
from collections import defaultdict

def load_cifar_from_torchvision():
    transform = transforms.Compose([transforms.ToTensor()])
    print("Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    print("CIFAR-10 dataset loaded successfully.")
    return train_dataset, test_dataset

def split_dataset_into_tasks(dataset, classes_per_task=20):
    print(f"Splitting dataset into tasks with {classes_per_task} classes per task.")
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    tasks = []
    for i in range(0, num_classes, classes_per_task):
        task_classes = list(range(i, min(i + classes_per_task, num_classes)))
        task_indices = np.where(np.isin(labels, task_classes))[0].tolist()
        tasks.append({"classes": task_classes, "indices": task_indices})
    print(f"Dataset split into {len(tasks)} tasks.")
    
    # Print classes for each task
    for task_id, task in enumerate(tasks):
        print(f"Task {task_id} is getting data from classes: {task['classes']}")
    
    return tasks

def distribute_data(tasks, num_clients, alpha, dataset, data_type="train"):
    client_data = {i: [] for i in range(num_clients)}
    label_distribution = {i: {class_label: 0 for class_label in range(10)} for i in range(num_clients)}
    print(f"Distributing {data_type} data across {num_clients} clients using Dirichlet distribution (alpha={alpha}).")
    targets = np.array(dataset.targets)
    
    for task_id, task in enumerate(tasks):
        task_classes = task["classes"]
        task_indices = task["indices"]
        np.random.shuffle(task_indices)
        
        # Generate Dirichlet distribution for each class
        dirichlet_distribution = np.random.dirichlet(alpha * np.ones(num_clients), len(task_classes))
        
        # Ensure no client gets zero data by adding a small epsilon to the distribution
        dirichlet_distribution = np.maximum(dirichlet_distribution, 1e-5)  # Avoid zero probabilities
        dirichlet_distribution /= dirichlet_distribution.sum(axis=1, keepdims=True)  # Renormalize
        
        client_splits = {i: [] for i in range(num_clients)}
        
        for class_idx, class_label in enumerate(task_classes):
            class_indices = np.where(targets == class_label)[0].tolist()
            proportions = dirichlet_distribution[class_idx]
            
            # Calculate cumulative proportions and split indices
            cum_proportions = np.cumsum(proportions)
            split_indices = np.array_split(class_indices, (cum_proportions[:-1] * len(class_indices)).astype(int))
            
            # Assign data to clients
            for client_id, split in enumerate(split_indices):
                client_splits[client_id].extend(split)
        
        # Assign data to clients and update label distribution
        for client_id, indices in client_splits.items():
            client_data[client_id].extend(indices)
            for idx in indices:
                label_distribution[client_id][targets[idx]] += 1
    
    # Print distribution for debugging
    for client_id, dist in label_distribution.items():
        print(f"Client {client_id} data distribution: {sum(dist.values())} samples")
    print(f"{data_type.capitalize()} data distribution completed.")
    return client_data, label_distribution
def save_dataset_structure(task_id, train_client_data, test_client_data, train_dataset, test_dataset, output_dir, alpha):
    task_folder = os.path.join(output_dir, f"task_{task_id}")
    os.makedirs(task_folder, exist_ok=True)
    
    # Create subfolders for train and test data
    train_folder = os.path.join(task_folder, "train")
    test_folder = os.path.join(task_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Initialize task configuration
    task_config = {
        "task_id": task_id,
        "num_classes": len(set(train_dataset.targets)),
        "alpha": alpha,
        "classes": tasks[task_id]["classes"],  # Add classes for this task
        "train_data": {},
        "test_data": {}
    }
    
    # Save train data and update config
    for client_id, indices in train_client_data.items():
        images, labels = [], []
        for idx in indices:
            img, label = train_dataset[idx]
            images.append(img.numpy())
            labels.append(label)
        np.savez(os.path.join(train_folder, f"{client_id}.npz"), images=np.array(images), labels=np.array(labels))
        task_config["train_data"][f"{client_id}"] = len(indices)
    
    # Save test data and update config
    for client_id, indices in test_client_data.items():
        images, labels = [], []
        for idx in indices:
            img, label = test_dataset[idx]
            images.append(img.numpy())
            labels.append(label)
        np.savez(os.path.join(test_folder, f"{client_id}.npz"), images=np.array(images), labels=np.array(labels))
        task_config["test_data"][f"{client_id}"] = len(indices)
    
    # Save the config file
    with open(os.path.join(task_folder, "config.json"), "w") as f:
        json.dump(task_config, f, indent=4)
    print(f"Task {task_id} dataset saved with config file.")
if __name__ == "__main__":
    output_dir = "./dataset/CIFAR10"
    num_clients = 3
    classes_per_task = 2
    alpha = 0.1  # Adjusted alpha for better distribution
    train_dataset, test_dataset = load_cifar_from_torchvision()
    tasks = split_dataset_into_tasks(train_dataset, classes_per_task)
    
    for task_id, task in enumerate(tasks):
        train_client_data, _ = distribute_data([task], num_clients, alpha, train_dataset, "train")
        test_client_data, _ = distribute_data([task], num_clients, alpha, test_dataset, "test")
        save_dataset_structure(task_id, train_client_data, test_client_data, train_dataset, test_dataset, output_dir, alpha)