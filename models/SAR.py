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

class Residual_block(nn.Module):
    def __init__(self,c_in,c_out,stride):
        super(Residual_block,self).__init__()
        self.downsample = None
        flag = False
        if isinstance(stride,tuple):
            if stride[0] > 1 or not c_in==c_out:
                self.downsample = nn.Sequential(nn.Conv2d(c_in,c_out,3,stride,1),nn.BatchNorm2d(c_out))
                flag = True
        else:
            if stride > 1 or not c_in==c_out:
                self.downsample = nn.Sequential(nn.Conv2d(c_in,c_out,3,stride,1),nn.BatchNorm2d(c_out))
                flag = True
        if flag:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in,c_out,3,stride,1),
                                    nn.BatchNorm2d(c_out))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in,c_out,1,stride,0),
                                    nn.BatchNorm2d(c_out))
        self.conv2 = nn.Sequential(nn.Conv2d(c_out,c_out,3,1,1),
                                   nn.BatchNorm2d(c_out))  
        self.relu = nn.ReLU()

    def forward(self,x):
        residual = x 
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(residual + conv2)

class ResNet_6_40(nn.Module):
    def __init__(self,c_in):
        super(ResNet_6_40,self).__init__()
        self.block0 = nn.Sequential(nn.Conv2d(c_in,64,3,1,1),nn.BatchNorm2d(64)) #48*160
        self.block1 = self._make_layer(64,128,(2,2),2) #24*80
        self.block2 = self._make_layer(128,256,(2,2),2) #12*40
        self.block3 = self._make_layer(256,512,(2,1),2) #6*40
        self.block4 = self._make_layer(512,512,(1,1),2) #6*40

    def _make_layer(self,c_in,c_out,stride,repeat=3):
        layers = []
        layers.append(Residual_block(c_in,c_out,stride))
        for i in range(repeat - 1):
            layers.append(Residual_block(c_out,c_out,1))
        return nn.Sequential(*layers)
    def forward(self,x):
        block0 = self.block0(x)
        # print(f"block0:{block0.size()}")
        block1 = self.block1(block0)
        # print(f"block1:{block1.size()}")
        block2 = self.block2(block1)
        # print(f"block2:{block2.size()}")
        block3 = self.block3(block2)
        # print(f"block3:{block3.size()}")
        block4 = self.block4(block3)

        return block4


class Lstm_encoder(nn.Module):
    """docstring for Lstm_encoder"""
    def __init__(self,input_size=512,hidden_size=512):
        super(Lstm_encoder, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=2,batch_first=True)
        self.max_pool = nn.AdaptiveMaxPool2d((1, None)) #the height is to 1

    def forward(self,conv):
        conv = self.max_pool(conv)
        b, c, h, w = conv.size()
        assert h == 1
        conv = conv.squeeze(2) #64*512*25
        conv = conv.permute(0,2,1).contiguous() #64*25*512
        gru_output,hidden = self.gru(conv)
        gru_output = gru_output[:,-1,:]
        # print(f'out size:{out.size()}')
        # print(f'hidden size:{hidden.size()}')
        holistic_feature = hidden

        return holistic_feature,gru_output

class Attention_cell(nn.Module):
    """docstring for Attention_cell"""
    def __init__(self, num_classes,input_size=512,hidden_size=512):
        super(Attention_cell, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=2,batch_first=True)
        self.hidden_liner = nn.Linear(hidden_size,hidden_size)
        self.conv_3_3 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        self.score_conv = nn.Conv2d(512,1,kernel_size=1,stride=1,padding=0)
        self.liner = nn.Linear(input_size+hidden_size,num_classes)
        self.dropout = nn.Dropout2d(p=0.5)
    def forward(self,i,conv,pre_hidden,pre_gru_output, pre_embedding):
        b, c, h, w = conv.size() 

        # pre_hidden = pre_hidden.unsqueeze(2)
        # print(f'pre_hidden:{pre_hidden.size()}')
        # print(pre_gru_output.size())
        # hidden_feature = self.conv_1_1(pre_hidden[-1].unsqueeze(2)) #(b,512,1)->(b,512,1)
        hidden_feature = self.hidden_liner(pre_gru_output.squeeze(1)) #(b,512)
        # print(f'hidden_feature{hidden_feature.size()}')
        hidden_feature = hidden_feature.unsqueeze(2).unsqueeze(3) #(b,512,1,1)
        hidden_feature = hidden_feature.expand(b,c,h,w).contiguous() #(b,c,h,w)
        # print(f'hidden_feature{hidden_feature.size()}')

        conv_feature = self.conv_3_3(conv) #(b,c,h,w)

        # attention = torch.tanh(conv_feature)
        # attention = torch.tanh(self.dropout(hidden_featureconv_feature)) #(b,c,h,w)
        attention = torch.tanh(hidden_feature*conv_feature)
        attention = self.score_conv(attention) #(b,1,h,w)
        # print(f'attention size:{attention.size()}')
        attention = attention.squeeze(1).view(b,-1) #(b,h*w)
        attention = F.softmax(attention,1).view(b,h,w).unsqueeze(1) #(b,1,h,w) 
        # print(f'attention size:{attention.size()}')
        # print(attention[0])


        attention_feature = conv * attention #(b,c,h,w)*(b,1,h,w) = (b,c,h,w)
        attention_feature = torch.sum(attention_feature,dim=(2,3)).unsqueeze(1) #(b,1,c)
        # print(f'attention features size:{attention_feature.size()}')
        # print(f'attention_feature size:{attention_feature.size()}')
        pre_embedding =pre_embedding.unsqueeze(1) #(b,1,512)
        gru_output, hidden = self.gru(pre_embedding,pre_hidden)

        concat_feature = torch.cat((attention_feature,gru_output),2).squeeze(1) #(b,1,c)+(b,1,c) -> (b,c)
        # prob = F.softmax(self.liner(concat_feature),dim=1)
        prob = self.liner(concat_feature)
        return prob,hidden,gru_output

class Attention_2d(nn.Module):
    """docstring for attention_2d"""
    def __init__(self,num_classes,input_size=512,hidden_size=512,num_embeddings=512):
        super(Attention_2d, self).__init__()
        self.num_classes = num_classes
        self.char_embeddings = nn.Embedding(self.num_classes,num_embeddings)
        self.attention_cell = Attention_cell(self.num_classes)

    def forward(self,conv,text_index,holistic_feature,gru_output, test=False):
        if test:
            steps = 20
        else:
            steps = text_index.size(0) #最长文本长度
        # print(f'step:{steps}')

        nB = conv.size(0)
        probs = torch.zeros(steps,nB,self.num_classes,device=torch.device('cuda'))
        for i in range(steps):
            if i == 0:
                init_hidden = holistic_feature
                start_index =  torch.zeros(nB, dtype=torch.long, device = torch.device('cuda'))
                pre_embedding = self.char_embeddings(start_index)
                prob,pre_hidden,pre_gru_output =  self.attention_cell(i,conv,init_hidden,gru_output,pre_embedding)

            else:
                if test:
                    pre_embedding = self.char_embeddings(prob.argmax(1))
                else:
                    pre_embedding = self.char_embeddings(text_index[i-1])

                prob,pre_hidden,pre_gru_output = self.attention_cell(i,conv,pre_hidden,pre_gru_output, pre_embedding)
            probs[i,:,:] = prob
            # print(f'probs size:{probs.size()}')
        return probs


class SAR(nn.Module):
    def __init__(self,num_classes, model_cfg):
        super(SAR, self).__init__()
        self.resnet = ResNet_6_40(model_cfg['nc'])
        self.lstm_encoder = Lstm_encoder()
        self.attention_2d = Attention_2d(num_classes=num_classes,hidden_size=model_cfg['hidden_size'], num_embeddings=model_cfg['num_embeddings'])

    def forward(self, input, text, text_rev='', test=False,global_step=1):
        # conv features
        # conv = self.cnn(input)
        conv = self.resnet(input)
        if global_step%10000 == 0:
            print('SAR: the conv size is {}'.format(conv.size()))
        # print(f'conv size:{conv.size()}')
        # b, c, h, w = conv.size() 
        # assert h == 1, "the height of conv must be 1"
        # conv = conv.squeeze(2)
        holistic_feature,gru_output = self.lstm_encoder(conv)
        # print(f'holistic_feature size:{holistic_feature.size()} ')
        output = self.attention_2d(conv, text, holistic_feature,gru_output, test)
        # pdb.set_trace()
        return output

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    model = SAR(1,26,512)
    model = model
    input = torch.randn(4,1,48,160)
    text = torch.LongTensor([[1,2,3,4,26],[2,2,2,2,26],[3,3,3,3,26],[4,4,4,4,26]]) #[b,seq_len]
    text = text.permute(1,0).contiguous() #[seq_len, b]
    model(input,text)
    # print('start')
    # input =torch.randn(1,512,4,25)
    # encoder = Lstm_encoder()
    # output =encoder(input)
    # print(output.size())