import kagglehub
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
import pandas as pd

def download_files():
    # Download latest version
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    return path
    # print("Path to dataset files:", path)

def get_training_files(path):
    base_train_path = os.path.join(path, "asl_alphabet_train", "asl_alphabet_train")
    train_files = []
    for directory in os.listdir(base_train_path):
        if "del" in directory or "space" in directory or "nothing" in directory:
            continue
        for file in os.listdir(os.path.join(base_train_path, directory)):
            train_files.append(os.path.join(base_train_path, directory, file))

    return train_files

def get_testing_files(path):
    base_test_path = os.path.join(path, "asl_alphabet_test")
    test_files = []
    for directory in os.listdir(base_test_path):
        for file in os.listdir(os.path.join(base_test_path, directory)):
            if "del" in file or "space" in file or "nothing" in file:
                continue
            test_files.append(os.path.join(base_test_path, directory, file))

    return test_files

def load_file(file):
    img = Image.open(file)
    category = file.split("/")[-2] #Different from train/test
    if "test" in file.split("/")[-1]:
        category = file.split("/")[-1].split("_")[0]
    information = {"image":img, "width":img.size[0], "height":img.size[1], "channels":len(img.mode), 
                    "filename":file, "category":category}
    return information

def get_files():
    path = download_files()
    train_files = get_training_files(path)
    random.shuffle(train_files)
    train_files, valid_files = train_test_split(train_files, test_size=.15)
    test_files = get_testing_files(path)

    return train_files, valid_files, test_files

def load_data():
    train, valid, test = get_files()
    train_information = [load_file(f) for f in train]
    df = pd.DataFrame(train_information)
    unique_classes = df.category.unique()
    class_map = {c:ord(c)-65 for c in unique_classes}
    print(class_map)

    DataLoader()

if __name__ == "__main__":
    get_files()