import torch.utils.data as data_utils
import torch

class LaplaceDataset(data_utils.Dataset):
    def __init__(self, z, y_):
        X, y = z, y_
        self.data = torch.from_numpy(X).float()
        self.targets = torch.from_numpy(y).float()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


def get_dataset(z, y):
    ds_train = LaplaceDataset(z, y)
    X_train, y_train = ds_train.data, ds_train.targets
    return X_train, y_train, ds_train