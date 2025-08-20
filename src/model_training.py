import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.data_handling import get_files, load_file
from src.model import VisionModel

def training_loop():
    train_data, valid_data, test_data = load_data()
    #Train Data needs to be a loader with batches
    model = VisionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train(model, criterion, optimizer, train_data)


def train(model:VisionModel, criterion:nn.CrossEntropyLoss, optimizer:optim.Adam, train_data):
    for epoch in range(2):
        pass
        running_loss = 0
        for i, data in enumerate(train_data, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backwards()
            optimizer.step()
            running_loss += loss.item()
            if i%2000 == 1999:
                print(f"EPOCH: {epoch}, STEP:{i}, AVG LOSS: {running_loss/2000}")
                running_loss = 0

if __name__ == "__main__":
    train()