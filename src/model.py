from torch import nn
import torch.nn.Functional as F

class VisionModel(nn.Module):
    def __init__(self):
        super.__init__()
        self.fc1 = nn.Linear(, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, )

    def forward(self, inp):
        x = inp
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = VisionModel()