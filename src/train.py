from data_loader import parse_annotations, get_dataloaders
from model import PneumoniaCNN
from data_loader import PneumoniaDataset
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate(model, dataloader):
    model.eval()
    preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to("cpu")
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze()

            preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    preds = np.array(preds)
    all_labels = np.array(all_labels)

    binary_preds = (preds >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(all_labels, binary_preds),
        "precision": precision_score(all_labels, binary_preds, zero_division=0),
        "recall": recall_score(all_labels, binary_preds, zero_division=0),
        "f1": f1_score(all_labels, binary_preds, zero_division=0),
        "auc_roc": roc_auc_score(all_labels, preds),
    }
    return metrics


data_path = "../data/mdai_rsna_project_x9N20BZa_images_2018-07-20-153330/"
json_path = "../data/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
uids, targets, uid_to_path_parts = parse_annotations(json_path)
batch_size = 32
pos_weight = torch.tensor(2.0066582914572866)
uids = uids[:500]

dataset = PneumoniaDataset(uids, data_path, targets, uid_to_path_parts)
training_dataloader, validation_dataloader = get_dataloaders(
    uids, data_path, targets, uid_to_path_parts, batch_size
)
model = PneumoniaCNN()
criteria = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"\nBatch Size: {batch_size}")
for epoch in range(10):
    model.train()
    total_loss = 0.0
    for images, labels in training_dataloader:
        optimizer.zero_grad()
        predictions = model(images)
        loss = criteria(predictions, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(training_dataloader)

    metrics = evaluate(model, validation_dataloader)
    print(
        f"Epoch {epoch + 1}: loss={average_loss:.4f} | "
        f"acc={metrics['accuracy']:.4f} | "
        f"prec={metrics['precision']:.4f} | "
        f"rec={metrics['recall']:.4f} | "
        f"f1={metrics['f1']:.4f} | "
        f"auc={metrics['auc_roc']:.4f}"
    )
