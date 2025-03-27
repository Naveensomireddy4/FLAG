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
| `-ls 2` | Local Epoch |
| `-tasks 5` | Number of tasks |
| `-ppe 30` | Patterns per Experience to get saved in episodic memory  |
| `-ss 30` | Sample size to be loaded from episodic memory |

## 📂 **Dataset Preparation**

To use your dataset, follow these steps:

1. **Create a `dataset` folder** in the project directory.
2. Inside `dataset`, create separate dataset folders such as `CIFAR100`, `CIFAR10`, etc.
3. Each dataset folder should contain task folders such as `task_0`, `task_1`, etc.
4. Each `task_x` folder should contain two subfolders:
   - `train/` - Stores training data for respective clients.
   - `test/` - Stores testing data for respective clients.
5. Inside `train/` and `test/`, each client's data should be stored in **.npz** format:
   - Example:
     ```
     dataset/
     ├── dataset_name/
     │   ├── task_0/
     │   │   ├── train/
     │   │   │   ├── 0.npz
     │   │   │   ├── 1.npz
     │   │   │   ├── 2.npz
     │   │   │   └── ...
     │   │   ├── test/
     │   │   │   ├── 0.npz
     │   │   │   ├── 1.npz
     │   │   │   ├── 2.npz
     │   │   │   └── ...
     ├── CIFAR10/
     │   ├── task_0/
     │   ├── task_1/
     │   └── ...
     ```

Ensure that the dataset structure follows this format for proper data loading during training.

