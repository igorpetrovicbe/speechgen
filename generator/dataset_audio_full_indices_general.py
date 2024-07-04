import math

import torch
from torch.utils.data import Dataset
import os


class AudioDataset(Dataset):
    def __init__(self, root_dir, sequence_length, device, vert_size, vocab_size=8192):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.vert_size = vert_size

        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.txt')]

        self.data = []
        counter = 0
        for file_name in self.file_list:
            file_path = os.path.join(self.root_dir, file_name)
            data = self.load_data(file_path)
            self.data.extend(data)
            print(counter)
            counter += 1

        self.data = torch.tensor(self.data).to(device)

    def __len__(self):
        return (len(self.data) - self.sequence_length) // self.vert_size

    def __getitem__(self, idx):
        idx *= self.vert_size
        input_indices = torch.tensor(self.data[idx:idx+self.sequence_length])

        # One-hot encode the sequences
        one_hot_target = torch.nn.functional.one_hot(torch.tensor(self.data[idx+1:idx+self.sequence_length+1]), num_classes=self.vocab_size).float()

        return input_indices, one_hot_target

    def load_data(self, file_path):
        data = [int(line) for line in open(file_path, 'r', encoding='ascii', errors='replace') if line.strip()]
        return data

    def load_data_old(self, file_path):
        with open(file_path, 'r', encoding='ascii', errors='replace') as file:
            text = file.read().split('\n')

        # Convert text to integers
        data = [int(line) for line in text if line.strip()]  # Convert non-empty lines to integers

        return data