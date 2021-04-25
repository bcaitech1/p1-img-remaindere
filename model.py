import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

#ref : efficient-pytorch, https://github.com/lukemelas/EfficientNet-PyTorch
#model : efficientnet, pretrained, lukemales
from efficientnet_pytorch import EfficientNet

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
        
class Efficientnet_B0(nn.Module):
    def __init__(self, num_classes : int = 18):
        super(Efficientnet_B0, self).__init__()
        #self.conv2d = nn.Conv2d(1, 3, 3, stride=1) #if read img as gray scale
        self._swish = MemoryEfficientSwish()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes = 18)
        
    def forward(self, x):
        # non-linear function으로 swish 사용
        # x = self._swish(self.conv2d(x)) #if read img as gray scale
        
        # effnet
        x = self.efficientnet(x)
        return x

class Efficientnet_B2(nn.Module):
    def __init__(self, num_classes : int = 18):
        super(Efficientnet_B2, self).__init__()
        #self.conv2d = nn.Conv2d(1, 3, 3, stride=1) #if read img as gray scale
        self._swish = MemoryEfficientSwish()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b2', num_classes = 18)
        
    def forward(self, x):
        # non-linear function으로 swish 사용
        # x = self._swish(self.conv2d(x)) #if read img as gray scale
        
        # effnet
        x = self.efficientnet(x)
        return x
    
class Efficientnet_B3(nn.Module):
    def __init__(self, num_classes : int = 18):
        super(Efficientnet_B3, self).__init__()
        #self.conv2d = nn.Conv2d(1, 3, 3, stride=1) #if read img as gray scale
        self._swish = MemoryEfficientSwish()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3', num_classes = 18)
        self._dropout = nn.Dropout(p=0.7, inplace = False)
        
    def forward(self, x):
        # non-linear function으로 swish 사용
        # x = self._swish(self.conv2d(x)) #if read img as gray scale
        
        # effnet
        x = self.efficientnet(x)
        x = self._dropout(x)
        return x
    
class Efficientnet_B4(nn.Module):
    def __init__(self, num_classes : int = 18):
        super(Efficientnet_B4, self).__init__()
        #self.conv2d = nn.Conv2d(1, 3, 3, stride=1) #if read img as gray scale
        self._swish = MemoryEfficientSwish()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 18)
        
    def forward(self, x):
        # non-linear function으로 swish 사용
        # x = self._swish(self.conv2d(x)) #if read img as gray scale
        
        # effnet
        x = self.efficientnet(x)
        return x
    
class Efficientnet_B5(nn.Module):
    def __init__(self, num_classes : int = 18):
        super(Efficientnet_B5, self).__init__()
        #self.conv2d = nn.Conv2d(1, 3, 3, stride=1) #if read img as gray scale
        self._swish = MemoryEfficientSwish()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b5', num_classes = 18)
        self._dropout = nn.Dropout(p=0.7, inplace = False)
        
    def forward(self, x):
        # non-linear function으로 swish 사용
        # x = self._swish(self.conv2d(x)) #if read img as gray scale
        
        # effnet
        x = self.efficientnet(x)
        x = self._dropout(x)
        return x