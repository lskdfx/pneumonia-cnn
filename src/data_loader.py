import torch
from torchvision.transforms import v2
import pydicom
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import os
import json


def get_dataloaders(
    uids, data_path, targets, uid_to_path_parts, batch_size, val_ratio=0.2
):
    dataset = PneumoniaDataset(uids, data_path, targets, uid_to_path_parts)

    validation_size = int(len(dataset) * val_ratio)
    training_size = len(dataset) - validation_size
    train_set, validation_set = random_split(dataset, [training_size, validation_size])

    training_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=False
    )
    return training_dataloader, validation_dataloader


class PneumoniaDataset(Dataset):
    def __init__(self, uids, data_path, targets, uid_to_path_parts):
        self.uids = uids
        self.data_path = data_path
        self.targets = targets
        self.transform = v2.Compose(
            [v2.Resize(size=(64, 64)), v2.ToDtype(torch.float32, scale=True)]
        )
        self.uid_to_path_parts = uid_to_path_parts

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        study_uid, series_uid = self.uid_to_path_parts[uid]
        path = os.path.join(self.data_path, study_uid, series_uid, uid + ".dcm")
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array
        image = torch.tensor(image).unsqueeze(0)
        image = self.transform(image)
        target = torch.tensor(self.targets[uid], dtype=torch.float32)
        return image, target


def parse_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    id_to_name = {}
    for group in data["labelGroups"]:
        for label in group["labels"]:
            id_to_name[label["id"]] = label["name"].strip()

    admin_filter = {
        "Question",
        "Question Addressed",
        "Exclude",
        "Adjudicate",
        "QA",
        "Flag",
        "Flag 2",
        "Flag 3",
        "Flag 4",
        "Flag 5",
    }
    positive_labels = {
        "Lung Opacity",
        "Lung Opacity (High Prob)",
        "Lung Opacity (Med Prob)",
        "Lung Opacity (Low Prob)",
    }

    study_labels = {}
    uid_to_path_parts = {}
    for ann in data["datasets"][0]["annotations"]:
        uid = ann.get("SOPInstanceUID")
        study_uid = ann.get("StudyInstanceUID")
        series_uid = ann.get("SeriesInstanceUID")
        if study_uid is None or series_uid is None:
            continue
        uid_to_path_parts[uid] = (study_uid, series_uid)
        if uid is None:
            continue
        label = id_to_name.get(ann["labelId"])
        if label is None or label in admin_filter:
            continue
        if uid not in study_labels:
            study_labels[uid] = set()
        study_labels[uid].add(label)
    targets = {}
    for uid, labels in study_labels.items():
        if labels & positive_labels:
            targets[uid] = 1.0
        else:
            targets[uid] = 0.0
    uids = list(targets.keys())
    return uids, targets, uid_to_path_parts
