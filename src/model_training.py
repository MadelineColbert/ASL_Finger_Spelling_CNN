import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from src.data_handling import get_files
from src.model import VisionModel
import mlflow
from torch.utils.data import DataLoader

def training_loop(device):
    train_data, valid_data, test_data = get_files()
    #Train Data needs to be a loader with batches
    unique_classes=29
    model = VisionModel(unique_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model = train(model, criterion, optimizer, train_data, device)
    return model


def train(model:VisionModel, criterion:nn.CrossEntropyLoss, optimizer:optim.Adam, train_data:DataLoader, device):
    model.train()
    running_loss = 0
    total_correct = 0
    total_used = 0
    with mlflow.start_run():
            
        for epoch in range(2):
            for i, data in enumerate(train_data, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                #Add in MLFlow for monitoring
                optimizer.zero_grad()
                outputs = model.forward(inputs)

                predictions = F.softmax(outputs).argmax(dim=1)

                loss = criterion(outputs, labels) # Move Criterion to model?
                accuracy_vec = predictions == labels
                total_correct += accuracy_vec.sum()
                total_used += accuracy_vec.shape[0]
                loss.backward()
                optimizer.step()

                #Check in on the loss
                running_loss += loss.item()
                if i%100 == 99:
                    loss = running_loss/100
                    accuracy = total_correct/total_used
                    mlflow.log_metrics(
                        {"batch_loss": loss, "batch_accuracy": accuracy},
                        step=epoch * len(train_data) + i
                    )
                    running_loss = 0
                    total_correct = 0
                    total_used = 0
        
        mlflow.pytorch.log_model(model)
    return model

def test(model:VisionModel, test_data):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data, 0):
            pass
            inputs, labels = data
            results = model(inputs)

    accuracy = 0
    return accuracy