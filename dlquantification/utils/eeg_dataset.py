# import os
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class EEGDataset(Dataset):
#     def __init__(self, folder_path, window_size=20, simulate_classes=2):
#         self.bags = []
#         self.labels = []
#
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 if file.endswith(".csv"):
#                     full_path = os.path.join(root, file)
#                     print(f"📄 Reading: {full_path}")
#                     df = pd.read_csv(full_path, sep=";")
#                     if "timestamp" in df.columns:
#                         df = df.drop(columns=["timestamp"])
#                     data = df.values.astype(np.float32)
#                     print(f"   → Shape: {data.shape}")  # Rows × Columns
#
#                     # Segment into bags
#                     num_windows = len(data) // window_size
#                     print(f"   → Can make {num_windows} bags")
#                     for i in range(num_windows):
#                         start = i * window_size
#                         end = start + window_size
#                         bag = data[start:end]
#                         self.bags.append(bag)
#                         self.labels.append(np.random.dirichlet(np.ones(simulate_classes)))
#
#         self.bags = np.stack(self.bags)
#         self.labels = np.stack(self.labels)
#
#     def __len__(self):
#         return len(self.bags)
#
#     def __getitem__(self, idx):
#         return torch.tensor(self.bags[idx]), torch.tensor(self.labels[idx])

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class EEGDataset(Dataset):
    def __init__(self, data_directory, test_split=0.2, random_seed=100, sequence_length=500):
        self.data_directory = data_directory
        self.test_split = test_split
        self.random_seed = random_seed
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []

        # Load and preprocess data
        self.load_data()

    def load_data(self):
        step_size = 250  # 50% overlap
        all_features = []

        # First pass: collect all features for global standardization
        for folder_name in os.listdir(self.data_directory):
            folder_path = os.path.join(self.data_directory, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
                    all_features.append(data[:, 1:-1])  # exclude timestamp and label

        # Fit scaler on all stacked features
        all_features = np.vstack(all_features)
        scaler = StandardScaler()
        scaler.fit(all_features)

        # Second pass: apply scaling and extract sequences
        for folder_name in os.listdir(self.data_directory):
            folder_path = os.path.join(self.data_directory, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    data = np.loadtxt(file_path, delimiter='\t', skiprows=1)

                    features = scaler.transform(data[:, 1:-1])
                    labels = data[:, -1]

                    for start in range(0, len(features) - self.sequence_length + 1, step_size):
                        end = start + self.sequence_length
                        if end <= len(features):
                            segment = features[start:end]
                            self.sequences.append(segment)
                            self.labels.append(labels[start])

        self.labels = np.array(self.labels, dtype=int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label

    def split_data(self):
        indices = list(range(len(self)))
        train_indices, test_indices = train_test_split(
            indices, test_size=self.test_split, random_state=self.random_seed
        )
        train_data = [self[i] for i in train_indices]
        test_data = [self[i] for i in test_indices]
        return train_data, test_data

if __name__ == "__main__":
    dataset = EEGDataset('/home/elx12/data_mining_project/gmnet-main/gmnet-main/dataset/EMG_Data')
    train_data, test_data = dataset.split_data()
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of testing samples: {len(test_data)}")
