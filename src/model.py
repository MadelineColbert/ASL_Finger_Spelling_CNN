from torch import nn
import torch
import torch.nn.functional as F

class VisionModel(nn.Module):
    def __init__(self, unique_classes):
        super(VisionModel, self).__init__()
        size = 200
        kernel_size=3
        out_size = size
        in_channels = 3
        categories = unique_classes 

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(kernel_size, kernel_size))
        out_size = out_size - kernel_size + 1 
        out_size=int(out_size/2)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(kernel_size, kernel_size))
        out_size = out_size - kernel_size + 1 
        out_size=int(out_size/2)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=(kernel_size, kernel_size))
        out_size = out_size - kernel_size + 1 
        out_size=int(out_size/2)

        flattened_size = 32 * out_size * out_size
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, categories)

        # Initialize weights?

    def forward(self, inp):
        x = inp
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #Softmax?
        # x = F.softmax(x)
        return x