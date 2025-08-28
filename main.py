import torch
from src.model_training import training_loop
# from src.data_handling import load_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = training_loop(device)
    print("TRAINING FINISHED")

if __name__ == "__main__":
    main()
    
    