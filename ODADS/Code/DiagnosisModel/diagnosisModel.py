import os
from torch import nn
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class SimpleNet(nn.Module):
    def __init__(self, numClasses, dNumClasses):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(numClasses, dNumClasses)

    def forward(self, x):
        x = self.fc(x)
        return x