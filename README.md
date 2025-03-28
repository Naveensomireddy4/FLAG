# Federated Learning with AGEM & FedAvg

This repository contains the implementation of Federated Learning (FL) using **Federated Averaging (FedAvg)** and **Average Gradient Episodic Memory (AGEM)**. The implementation includes both **client-side** and **server-side** functionalities.

## Code Structure

### Main Execution File
- **Main Script:** `system/main.py`  
  - This is the entry point to run the federated learning experiments.

### Model Code
- **Model Implementation:** `system/flcore/trainmodel/my.py`  
  - Defines the model architecture used in the training.

### Federated Learning Components
#### Server-Side Code
- **FedAvg Algorithm Implementation:** `system/flcore/servers/fedavg.py`
- **Server Base Class:** `system/flcore/servers/serverbase.py`

#### Client-Side Code
- **Client Base Class:** `system/flcore/clients/clientbase.py`
- **Client-Side FedAvg Implementation:** `system/flcore/clients/clientavg.py`

### AGEM Plugin
- **AGEM Algorithm Implementation:** `system/avalanche/training/plugins/agem.py`

## Notes
- The implementation uses FedAvg as the aggregation algorithm.
- The client and server implementations extend from the respective base classes for flexibility.

## 📁 **Project Structure**
- **AGEM Code**: Located at `system/avalanche/training/plugins/agem.py`
- **Client-Side Code**: Located at `system/flcore/clients`
- **Server-Side Code**: Located at `system/flcore/servers`
- **Model-Side Code**: Located at `system/flcore/trainmodel/my.py`
- **Base Classes**:
  - `clientbase.py`: Base implementation for client-side FL.
  - `serverbase.py`: Base implementation for server-side FL.

## 🚀 **Running the Code**
To execute the training process, follow these steps:

### **1. Navigate to the System Directory**
```sh
cd system
```

### **2. Run the Training Script**
```sh
python3 main.py -data CIFAR100 -m resnet18 -algo FedAvg -gr 99 -did 0 -nc 3 -nb 20 -lbs 20 -ls 2 -tasks 5 -ppe 30 -ss 30
```

## ⚙ **Command Parameters**
| Parameter | Description |
|-----------|-------------|
| `-data CIFAR100` | Uses **BloodMNIST** dataset from MedMNIST. |
| `-m resnet18` | Uses **ResNet-18** as the model architecture. |
| `-algo FedAvg` | Runs the **Federated Averaging (FedAvg)** algorithm. |
| `-gr 99` | Sets **50** global communication rounds. |
| `-did 0` | Device ID (**0 for first available GPU**). Use `-1` for CPU. |
| `-nc 3` | Number of clients (ensure it matches the dataset's clients count). |
| `-nb 20` | Number of classes in single task (ensure it matches the dataset's class count). |
| `-lbs 20` | Batch size |
| `ls 2` | Local Epoch |
| `-tasks 5` | Number of tasks |
| `ppe 30` | Patterns per Experience to get saved in episodic memory  |
| `ss 30` | Sample size to be loaded from episodic memory |

## 📌 **Dataset Preparation**
- Place your dataset by creating a `dataset` folder.
- Inside `dataset`, create subfolders for each dataset (e.g., `CIFAR100`, `CIFAR10`, `BloodMNIST`, etc.).
- Each dataset folder should contain task folders like `task_0`, `task_1`, etc.
- Each task folder should contain `train` and `test` subfolders, which should contain respective client data in `.npz` format (e.g., `0.npz`, `1.npz`, `2.npz`, etc.).

### **Disclaimer**
- The `.npz` files **must** contain data in `(x, y)` format, where `x` represents images and `y` represents labels.
- None of the `.npz` files should be empty.
- Ensure that your partitions follow the required structure to avoid any issues in training.

If your data is in image and label format, you can use `dataset_reshape.py` to process and convert it into the required `(x, y)` structure.

## 🆕 **Adding a New Model**
If you want to add a new model:
1. Add the model implementation inside `system/flcore/trainmodel`.
2. Ensure that in the forward pass, the model takes `task_id` along with the input image.
3. Import your model in `system/main.py` and add it there.

