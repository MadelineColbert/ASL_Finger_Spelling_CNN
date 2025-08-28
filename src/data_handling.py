import kagglehub
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch

def download_files():
    # Download latest version
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    return path
    # print("Path to dataset files:", path)


def get_files():
    path = download_files()
    batch_size=4
    # train_files = get_training_files(path)
    # test_files = get_testing_files(path)
    train_path = os.path.join(path, "asl_alphabet_train", "asl_alphabet_train")
    test_path = os.path.join(path, "asl_alphabet_test")

    TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_folder = ImageFolder(train_path, transform=TRANSFORM)
    generator = torch.Generator()
    train, valid = torch.utils.data.random_split(train_folder, [.85, .15], generator=generator)
    test = ImageFolder(test_path, transform=TRANSFORM)

    train_loader = DataLoader(train, batch_size=batch_size)
    valid_loader = DataLoader(valid, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    get_files()