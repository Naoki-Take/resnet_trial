from collections import defaultdict
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

from datetime import datetime
import gc
import timm
from tqdm import tqdm

# resnet50のファインチューニング用のコード
model = timm.create_model('resnet50', pretrained=True, num_classes=2)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 2)

# VITに変更できます
# model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2).to("cuda")

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gc.collect()


def train_one_epoch(train_dataloader, model, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    label_correct = defaultdict(int)
    label_total = defaultdict(int)

    for data in tqdm(train_dataloader):
        inputs, labels = data[0].to("cuda"), data[1].view(-1).to("cuda")
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predictions = torch.max(outputs, 1)
        total_correct += torch.sum(predictions == labels).item()
        total_samples += labels.size(0)

        for label, prediction in zip(labels, predictions):
            label_total[label.item()] += 1
            if label == prediction:
                label_correct[label.item()] += 1

    avg_loss = running_loss / len(train_dataloader)
    total_accuracy = total_correct / total_samples
    label_accuracy = {
        label: correct / label_total[label] for label, correct in label_correct.items()}

    return avg_loss, total_accuracy, label_accuracy


def validate_one_epoch(val_dataloader, model, loss_fn):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    label_correct = defaultdict(int)
    label_total = defaultdict(int)

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader)):
            inputs, labels = data
            inputs, labels = inputs.to("cuda"), labels.view(-1).to("cuda")

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_correct += torch.sum(predicted == labels).item()
            total_samples += labels.size(0)

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    label_correct[label.item()] += 1
                label_total[label.item()] += 1

    avg_loss = running_loss / len(val_dataloader)
    total_accuracy = total_correct / total_samples
    label_accuracy = {label: (
        correct / label_total[label]) for label, correct in label_correct.items()}
    return avg_loss, total_accuracy, label_accuracy


EPOCH = 3  # 仮のエポック数
epoch_number = 0
