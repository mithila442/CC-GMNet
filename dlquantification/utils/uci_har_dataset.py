import os
import numpy as np
import torch
from torch.utils.data import Dataset

class UCIHARDataset(Dataset):
    def __init__(self, base_dir, split="train", sequence_length=128):
        assert split in ["train", "test"], "Split must be 'train' or 'test'"
        self.sequence_length = sequence_length

        # 9 raw signal files
        signals = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z"
        ]

        signal_data = []
        for signal in signals:
            file_path = os.path.join(base_dir, split, "Inertial Signals", f"{signal}_{split}.txt")
            data = np.loadtxt(file_path)  # shape: [samples, 128]
            signal_data.append(data)

        # Stack as: [samples, 128, 9]
        X = np.stack(signal_data, axis=-1)  # [samples, 128, 9]
        y_path = os.path.join(base_dir, split, f"y_{split}.txt")
        y = np.loadtxt(y_path).astype(int)  # [samples]

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y - 1, dtype=torch.long)  # Make labels 0-based

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
