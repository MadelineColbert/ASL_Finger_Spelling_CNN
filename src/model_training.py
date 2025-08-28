import torch.nn as nn
import torch
import torch.optim as optim
from src.data_handling import get_files
from src.model import VisionModel

def training_loop():
    train_data, valid_data, test_data = get_files()
    #Train Data needs to be a loader with batches
    unique_classes=29
    model = VisionModel(unique_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train(model, criterion, optimizer, train_data)


def train(model:VisionModel, criterion:nn.CrossEntropyLoss, optimizer:optim.Adam, train_data):
    model.train()
    for epoch in range(2):
        running_loss = 0
        for i, data in enumerate(train_data, 0):
            inputs, labels = data

            #Add in MLFlow for monitoring

            optimizer.zero_grad()
            outputs = model.forward(inputs)

            loss = criterion(outputs, labels) # Move Criterion to model?
            loss.backward()
            optimizer.step()

            #Check in on the loss
            running_loss += loss.item()
            if i%1000 == 999:
                print(f"EPOCH: {epoch}, STEP:{i}, AVG LOSS: {running_loss/2000}")
                running_loss = 0

def test(model:VisionModel, test_data):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data, 0):
            pass
            inputs, labels = data
            results = model(inputs)

    accuracy = 0
    return accuracy