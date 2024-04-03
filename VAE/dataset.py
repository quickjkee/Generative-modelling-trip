import torch
import numpy as np
import PIL

from torch.utils.data import Dataset


def train_test_split(ex_idx):
    train_part = int(len(ex_idx) * 0.9)
    np.random.shuffle(ex_idx)

    train_idx = ex_idx[:train_part]
    test_idx = ex_idx[train_part:]

    return list(train_idx), list(test_idx)


class CellTrapSet(Dataset):

    def __init__(self, names, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = list(index)

        img_name = self.names[index]

        image = PIL.Image.open(self.root_dir + 'images/' + img_name).convert('L')

        sample = {0: image}

        if self.transforms:
            sample = {0: self.transforms(sample[0])}

        return sample
