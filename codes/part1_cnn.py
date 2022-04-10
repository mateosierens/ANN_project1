from torch import nn, flatten
import torch.nn.functional as F

class cnn(nn.Module):
    def __init__(self, x=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        self.fc = nn.Linear(576, 15)
        self.dropout = nn.Dropout(0.25)# it defines a dropout layer which dropouts 25 percent of nodes

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2, stride=2)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)  # dropout 25 percent of nodes in tensor x
        x = F.max_pool2d(x, 2)
        x = flatten(x, 1)
        x = self.fc(x)
        output = F.softmax(x, dim=1)
        return output


