import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from src.data_handling import get_files
from src.model import VisionModel
import mlflow
from torch.utils.data import DataLoader
import os
import cv2
from collections import defaultdict
import json
from tqdm import tqdm

def training_loop(device):
    train_data, valid_data, test_data, classes = get_files()
    #Train Data needs to be a loader with batches
    model = VisionModel(classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("asl_finger_spelling")
    with mlflow.start_run() as run:

        run_id = run.info.run_id 

        signature = train(model, criterion, optimizer, train_data, device)

        mlflow.pytorch.log_model(model, name="final_model", signature=signature)

        confusion_matrix = validation(run_id, valid_data, device)
        with open("confusion_matrix.json", "w") as f:
            json.dump(confusion_matrix, f)

        mlflow.log_artifact("confusion_matrix.json", run_id=run_id)
        #Show four images and give 4 predictions (think about how to do this)
        # visual_predictions(run_id, valid_data, device)

    return model


def train(model:VisionModel, criterion:nn.CrossEntropyLoss, optimizer:optim.Adam, train_data:DataLoader, device):
    model.train()
    running_loss = 0
    total_correct = 0
    total_used = 0
    for epoch in range(2):
        print(f"Epoch {epoch+1}")
        for i, data in tqdm(enumerate(train_data, 0), total=len(train_data)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            #Add in MLFlow for monitoring
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            predictions = F.softmax(outputs, dim=1).argmax(dim=1)

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
                    {"train_loss": loss, "train_accuracy": accuracy},
                    step=epoch * len(train_data) + i
                )
                running_loss = 0
                total_correct = 0
                total_used = 0

    signature = mlflow.models.infer_signature(inputs.detach().cpu().numpy(), outputs.detach().cpu().numpy())
    
    return signature

def validation(run_id, valid_data:DataLoader, device):
    model_uri = f"runs:/{run_id}/final_model"
    model: VisionModel = mlflow.pytorch.load_model(model_uri)
    model.eval()
    total_correct = 0
    total_used = 0
    confusion_matrix = defaultdict(dict)
    with torch.no_grad():
        for i, data in tqdm(enumerate(valid_data), total=len(valid_data)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            results = model(inputs)

            results = F.softmax(results, dim=1).argmax(dim=1)

            accuracy_vec = results == labels
            total_correct += accuracy_vec.sum()
            total_used += accuracy_vec.shape[0]

            cat_results = model.get_mappings(results)
            cat_labels = model.get_mappings(labels)
            for l, r in zip(cat_labels, cat_results):
                if confusion_matrix[l].get(r) == None:
                    confusion_matrix[l][r] = 0    
                confusion_matrix[l][r] += 1

            if i%100 == 99:
                accuracy = total_correct/total_used
                mlflow.log_metrics(
                    {"valid_accuracy": accuracy},
                    step=i
                )
                total_correct = 0
                total_used = 0
        
    return confusion_matrix

def visual_predictions(run_id, valid_data:DataLoader, device):
    model_uri = f"runs:/{run_id}/final_model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_data):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            results = model(inputs)

            for x, inp in enumerate(inputs, 0):
                numpy_inp = inp.cpu().numpy()
                cv2.imwrite(f"{x}.jpg", numpy_inp)


            results = F.softmax(results).argmax(dim=1)
            print(labels, results)
            break