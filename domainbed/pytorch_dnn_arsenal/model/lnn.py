
import torch
import torch.nn as nn
import torch.nn.functional as F

# =================== LNN  ===================
class LNN(torch.nn.Module):
    def __init__(self):
        super(LNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = torch.nn.Linear(28*28, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100, 100)
        self.fc4 = torch.nn.Linear(100, 100)
        self.fc5 = torch.nn.Linear(100, 100)
        self.fc6 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = F.log_softmax(x, dim=1)
        return x

class LNN_tiny3(torch.nn.Module):
    def __init__(self):
        super(LNN_tiny3, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = torch.nn.Linear(7*7, 10)
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

# =================== LNN w/ Skip Connection  ===================
class LNN_SC(torch.nn.Module):
    def __init__(self):
        super(LNN_SC, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = torch.nn.Linear(28*28, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100, 100)
        self.fc4 = torch.nn.Linear(100, 100)
        self.fc5 = torch.nn.Linear(100, 100)
        self.fc6 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        residual_1 = self.fc1(x)

        x = self.fc2(residual_1)
        residual_2 = self.fc3(x)
        residual_2 += residual_1

        x = self.fc4(residual_2)
        x = self.fc5(x)
        x += residual_2

        x = self.fc6(x)
        x = F.log_softmax(x, dim=1)
        return x

class LNN_SC_tiny3(torch.nn.Module):
    def __init__(self):
        super(LNN_SC_tiny3, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = torch.nn.Linear(7*7, 10)
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        residual_1 = x
        x = self.fc2(x)
        x += residual_1
        residual_2 = x
        x = self.fc3(x)
        x += residual_2
        x = F.log_softmax(x, dim=1)
        return x

# =================== LNN  ===================
class LNN_CIFAR(torch.nn.Module):
    def __init__(self):
        super(LNN_CIFAR, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = torch.nn.Linear(32*32*3, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100, 100)
        self.fc4 = torch.nn.Linear(100, 100)
        self.fc5 = torch.nn.Linear(100, 100)
        self.fc6 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = F.log_softmax(x, dim=1)
        return x

# =================== LNN w/ Skip Connection  ===================
class LNN_SC_CIFAR(torch.nn.Module):
    def __init__(self):
        super(LNN_SC_CIFAR, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = torch.nn.Linear(32*32*3, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100, 100)
        self.fc4 = torch.nn.Linear(100, 100)
        self.fc5 = torch.nn.Linear(100, 100)
        self.fc6 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        residual_1 = self.fc1(x)

        x = self.fc2(residual_1)
        residual_2 = self.fc3(x)
        residual_2 += residual_1

        x = self.fc4(residual_2)
        x = self.fc5(x)
        x += residual_2

        x = self.fc6(x)
        x = F.log_softmax(x, dim=1)
        return x
