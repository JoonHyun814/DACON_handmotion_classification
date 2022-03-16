import torch
import torch.nn as nn
import torch.nn.functional as F
from zmq import device

class LSTM_BaseModel(nn.Module):
    def __init__(self, device, xdim =1 , hdim = 16, n_layer = 3, class_num = 4):
        super().__init__()
        self.n_layer = n_layer
        self.hdim = hdim
        self.device = device
        self.rnn = nn.LSTM(input_size=xdim,hidden_size=hdim,num_layers=n_layer,batch_first=True)
        self.fc = nn.Linear(hdim, class_num)

    def forward(self, x):
        h0 = torch.zeros(self.n_layer,x.size(0),self.hdim, dtype=torch.float32).to(self.device)
        c0 = torch.zeros(self.n_layer,x.size(0),self.hdim, dtype=torch.float32).to(self.device)
        rnn_out,(hn,cn) = self.rnn(x.view(-1,32,1), (h0,c0))
        out = self.fc(rnn_out[:,-1:])
        return out

class BaseModel(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.l1 = nn.Linear(32,16)
        self.l2 = nn.Linear(16,16)
        self.l3 = nn.Linear(16,8)
        self.l4 = nn.Linear(8,8)
        self.l5 = nn.Linear(8,4)

    def forward(self,x):
        x = self.l1(x)
        x = nn.ReLU()(x)
        x = self.l2(x)
        x = nn.ReLU()(x)
        x = self.l3(x)
        x = nn.ReLU()(x)
        x = self.l4(x)
        x = nn.ReLU()(x)
        x = self.l5(x)
        x = nn.ReLU()(x)
        return(x)

################################Resnet###########################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                                stride=stride,padding=padding))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock,self).__init__()

        self.resblk = nn.Sequential(
            ConvBlock(in_channels,out_channels,kernel_size,
                                stride,padding),
            nn.ReLU(),
            ConvBlock(out_channels,out_channels,kernel_size,
                                stride,padding)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + self.resblk(x)
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, device,in_channels=1, out_channels=4):
        super(ResNet, self).__init__()

        self.conv2_x=nn.Sequential(ConvBlock(in_channels, 16, kernel_size=3, stride=1, padding=1),
            ResBlock(16,16),
            ResBlock(16,16))
        self.conv3_x=nn.Sequential(ConvBlock(16, 32, kernel_size=3, stride=1, padding=1),
            ResBlock(32,32),
            ResBlock(32,32))
        self.conv4_x=nn.Sequential(ConvBlock(32, 64, kernel_size=2, stride=2, padding=0),
            ResBlock(64,64),
            ResBlock(64,64))

        self.avpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.conv2_x(x.view(-1,1,4,8))
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.avpool(x)
        x = x.view(-1,64)
        out = self.fc(x)
        return out