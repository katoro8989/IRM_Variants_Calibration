"""
Modified from https://github.com/Eulring/ALL-CNN-on-CIFAR10/blob/master/models/ALL_CNN_C.py
reference: https://github.com/fsschneider/DeepOBS/blob/master/deepobs/tensorflow/testproblems/cifar100_allcnnc.py
----------------------------------------------------------------
Total params: 1,387,108 (for CIFAR-100)
"""

import torch
from torch import nn
import torch.nn.functional as F

class ALL_CNN_C(nn.Module):
    
    def __init__(self, num_classes=100):
        
        super(ALL_CNN_C, self).__init__()
                
        self.drop_out0 = nn.Dropout2d(p = 0.2)
        self.conv1 = nn.Conv2d(3, 96, 3, padding = 1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding = 1)
        self.conv3 = nn.Conv2d(96, 96, 3, stride = 2, padding = 1)
        self.drop_out1 = nn.Dropout2d(p = 0.5)
        self.conv4 = nn.Conv2d(96, 192, 3, padding = 1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding = 1)
        self.conv6 = nn.Conv2d(192, 192, 3, stride = 2, padding = 1)
        self.drop_out2 = nn.Dropout2d(p = 0.5)
        self.conv7 = nn.Conv2d(192, 192, 3, padding = 0)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, num_classes, 1)
        self.avg = nn.AvgPool2d(6)
        
        self._init_weight()
        
    def _init_weight(self):
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)
        nn.init.xavier_normal_(self.conv6.weight)
        nn.init.xavier_normal_(self.conv7.weight)
        nn.init.xavier_normal_(self.conv8.weight)
        nn.init.xavier_normal_(self.conv9.weight)
                
    def forward(self, x):
        
        x = self.drop_out0(x)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.drop_out1(x)
            
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.drop_out2(x)    
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.avg(x)
        x = torch.squeeze(x)
        return x
