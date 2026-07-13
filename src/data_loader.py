import torch
from torchvision.transforms import v2
import pydicom
import numpy as np
from torch.utils.data import Dataset
import os


class PneumoniaDataset(Dataset):
    def __init__(self, uids, data_path, targets):
        self.uids = uids
        self.data_path = data_path
        self.targets = targets
        self.transform = v2.Compose(
            [v2.Resize(size=(64, 64)), v2.ToDtype(torch.float32, scale=True)]
        )

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        path = os.path.join(self.data_path, uid + ".dcm")
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array
        image = torch.tensor(image).unsqueeze(0)
        image = self.transform(image)
        target = torch.tensor(self.targets[uid], dtype=torch.float32)
        return image, target
