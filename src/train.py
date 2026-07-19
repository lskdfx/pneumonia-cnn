from data_loader import parse_annotations
from model import PneumoniaCNN
from data_loader import PneumoniaDataset
import torch
from torch import nn
from torch.utils.data import DataLoader

data_path = "../data/mdai_rsna_project_x9N20BZa_images_2018-07-20-153330/"
json_path = "../data/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
uids, targets, uid_to_path_parts = parse_annotations(json_path)
uids = uids[:500]

dataset = PneumoniaDataset(uids, data_path, targets, uid_to_path_parts)
batch_sizes = [4, 8, 16, 32, 64]
dataloader = DataLoader(dataset, batch_size=batch_sizes[0], shuffle=True)
for batch_size in batch_sizes:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = PneumoniaCNN()
    criteria = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(batch_size)
    for epoch in range(10):
        total_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            predictions = model(images)
            loss = criteria(predictions, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: loss={total_loss / len(dataloader):.4f}")
