"""
Misc helper functions for training, storing and evaluating models.
"""
import io

import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import PreTrainedModel

ID_TO_SENTIMENT = {0: "Negative", 1: "Neutral", 2: "Positive"}
SENTIMENT_TO_ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID_TO_TOPIC = {0: "Other", 1: "Economics"}
TOPIC_TO_ID = {"Other": 0, "Economics": 1}


class SentimentDataset(Dataset):

    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        data['Headlines'] = data['Headlines'].astype(str)
        x = data['Headlines'].values
        y = data['Sentiment'].map(SENTIMENT_TO_ID).values
        self.x = x
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TopicDataset(Dataset):

    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        data['Headlines'] = data['Headlines'].astype(str)
        x = data['Headlines'].values
        y = data['Topic'].map(TOPIC_TO_ID).values
        self.x = x
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def save_model(model, path):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    with open(path, 'wb') as f:
        f.write(buffer.getbuffer())
    del model


def load_model(model, path, device):
    model.load_state_dict(torch.load(path))
    return model.to(device)


def test_accuracy(model, test_loader, device):
    total_correct = 0
    total_samples = 0
    model.eval()
    for features, targets in test_loader:
        features = features.to(device=device)
        targets = targets.to(device=device)

        if isinstance(model, PreTrainedModel):
            outputs = model(**features, labels=targets)
            num_samples = outputs.logits.shape[0]
        else:
            outputs = model(features)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            num_samples = outputs.shape[0]

        accuracy = _compute_accuracy(outputs, targets)
        total_correct += accuracy * num_samples
        total_samples += num_samples
    model.train()

    print(f'Test size: {total_samples}')
    print(f'Baseline accuracy: {_baseline_accuracy(test_loader)*100:.2f} %')
    print(f"Accuracy: {float(total_correct)/float(total_samples)*100:.2f} %")


def _baseline_accuracy(data_loader):
    counter = torch.zeros(3)
    for _, targets in data_loader:
        if targets.ndim == 1:
            targets = F.one_hot(targets, num_classes=3)
        counter = counter + targets.sum(axis=0)
    return (counter.max() / counter.sum()).item()


def _compute_accuracy(output, target):
    with torch.no_grad():
        if hasattr(output, 'logits'):  # Huggingface output
            logits = output.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            accuracy = (probs.argmax(dim=1) == target).sum() / len(target)
        else:  # Torch output
            probs = torch.nn.functional.softmax(output, dim=1)
            accuracy = (probs.argmax(dim=1) == target.argmax(dim=1)).sum() / len(target)
    return accuracy
