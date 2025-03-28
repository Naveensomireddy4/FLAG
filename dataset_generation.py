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
    label_distribution = {i: {class_label: 0 for class_label in range(10)} for i in range(num_clients)}
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

def save_dataset_structure(task_id, train_client_data, test_client_data, train_dataset, test_dataset, output_dir, alpha):
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
        task_config["test_data"][f"{client_id}"] = len(indices)
    
    with open(os.path.join(task_folder, "config.json"), "w") as f:
        json.dump(task_config, f, indent=4)
    print(f"Task {task_id} dataset saved with config file.")

if __name__ == "__main__":
    output_dir = "./dataset/CIFAR10"
    num_clients = 3
    classes_per_task = 2
    alpha = 0.1
    train_dataset, test_dataset = load_cifar_from_torchvision()
    tasks = split_dataset_into_tasks(train_dataset, classes_per_task)
    
    for task_id, task in enumerate(tasks):
        train_client_data, _ = distribute_data([task], num_clients, alpha, train_dataset, "train")
        test_client_data, _ = distribute_data([task], num_clients, alpha, test_dataset, "test")
        save_dataset_structure(task_id, train_client_data, test_client_data, train_dataset, test_dataset, output_dir, alpha)
