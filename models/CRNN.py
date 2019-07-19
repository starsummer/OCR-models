import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import transforms
# from models.fracPickup import fracPickup
import pdb
from PIL import Image

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden,num_layers=2, bidirectional=True, dropout=0.3)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class Residual_block(nn.Module):
    def __init__(self,c_in,c_out,stride):
        super(Residual_block,self).__init__()
        self.downsample = None
        flag = False
        if isinstance(stride,tuple):
            if stride[0] > 1:
                self.downsample = nn.Sequential(nn.Conv2d(c_in,c_out,3,stride,1),nn.BatchNorm2d(c_out,momentum=0.01))
                flag = True
        else:
            if stride > 1:
                self.downsample = nn.Sequential(nn.Conv2d(c_in,c_out,3,stride,1),nn.BatchNorm2d(c_out,momentum=0.01))
                flag = True
        if flag:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in,c_out,3,stride,1),
                                    nn.BatchNorm2d(c_out,momentum=0.01))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in,c_out,1,stride,0),
                                    nn.BatchNorm2d(c_out,momentum=0.01))
        self.conv2 = nn.Sequential(nn.Conv2d(c_out,c_out,3,1,1),
                                   nn.BatchNorm2d(c_out,momentum=0.01))  
        self.relu = nn.ReLU()

    def forward(self,x):
        residual = x 
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(residual + conv2)

class ResNet(nn.Module):
    def __init__(self,c_in):
        super(ResNet,self).__init__()
        self.block0 = nn.Sequential(nn.Conv2d(c_in,64,7,1,1),nn.BatchNorm2d(64)) #48*160
        self.block1 = self._make_layer(64,128,(2,2),2) #24*80
        self.block2 = self._make_layer(128,256,(2,2),2) #12*40
        self.block3 = self._make_layer(256,512,(2,2),2) #6*20
        self.block4 = self._make_layer(512,512,(2,1),2) #3*20
        self.block5 = self._make_layer(512,512,(3,1),1) #1*20

    def _make_layer(self,c_in,c_out,stride,repeat=3):
        layers = []
        layers.append(Residual_block(c_in,c_out,stride))
        for i in range(repeat - 1):
            layers.append(Residual_block(c_out,c_out,1))
        return nn.Sequential(*layers)
    def forward(self,x):
        block0 = self.block0(x)
        block1 = self.block1(block0)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        return block5

class CRNN(nn.Module):

    def __init__(self, num_classes, model_cfg):
        super(CRNN, self).__init__()

        self.resnet = ResNet(model_cfg['nc'])
        self.birnn = BidirectionalLSTM(512, model_cfg['hidden_size'], num_classes)        

    def forward(self, input, test=False):
        # conv features
        # conv = self.cnn(input)
        conv = self.resnet(input)
        # print(conv.size())
        b, c, h, w = conv.size() 
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2,0,1).contiguous()
        output = self.birnn(conv)
        # print(output.size())
        output =  F.log_softmax(output, dim=2)
        return output

if __name__ == '__main__':
    model = CRNN(1,26,512)
    model = model.cuda()
    input = torch.zeros(4,1,48,900,device=torch.device('cuda'))
    # text = torch.LongTensor([[1,2,3,4,26],[2,2,2,2,26],[3,3,3,3,26],[4,4,4,4,26]]).cuda()
    model(input)
