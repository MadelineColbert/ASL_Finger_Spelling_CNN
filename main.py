import torch
from src.model_training import training_loop
# from src.data_handling import load_data
import mlflow
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))
    mlflow.set_experiment("asl_finger_spelling")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = training_loop(device)
    print("TRAINING FINISHED")

if __name__ == "__main__":
    main()
    
    