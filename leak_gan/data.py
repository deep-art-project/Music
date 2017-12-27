from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class Real_Data_Set(Dataset):

    def __init__(self, filepath):
        self.data = np.load(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return torch.from_numpy(self.data[idx]).long()

class Dis_Data_Set(Dataset):

    def __init__(self, positive_filepath, negative_filepath):
        pos_data = np.load(positive_filepath)
        neg_data = np.load(negative_filepath)
        pos_label = np.array([1 for _ in pos_data])
        neg_label = np.array([0 for _ in neg_data])
        self.data = np.concatenate([pos_data, neg_data])
        self.label = np.concatenate([pos_label, neg_label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).long()
        label = torch.nn.init.constant(
              torch.zeros(1), int(self.label[idx])
          ).long()
        return {"data":data, "label":label}

def real_data_loader(filepath, batch_size, shuffle, num_workers,
                     pin_memory):
    dataset = Real_Data_Set(filepath)
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory)

def dis_data_loader(positive_filepath, negative_filepath,
                    batch_size, shuffle, num_workers, pin_memory):
    dataset = Dis_Data_Set(positive_filepath, negative_filepath)
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory)
