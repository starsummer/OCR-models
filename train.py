import torch
import torch.nn as nn
import tools.dataloader as dataloader
import tools.utils as utils
from models.CRNN import CRNN
from models.SAR import SAR
import torch.optim as optim
import numpy as np
import pdb
import time
import tools.eval as Eval
import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='config/config_SAR.py')
parser.add_argument('--gpu', type=str, default='0')
opt = parser.parse_args()

model_cfg,train_cfg,data_cfg = utils.read_config_file(opt.config_file) #读取配置文件
index2char,char2index = utils.get_dict(data_cfg['dict_file']) #获取char-index对应字典

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
torch.backends.cudnn.benchmark = True

nclass = len(index2char) #类别总数
print('the dice length is {}'.format(len(index2char)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = dataloader.get_train_loader(data_cfg,char2index) ##获取数据loader


if model_cfg['method'] == 'CRNN':
    criterion = nn.CTCLoss(zero_infinity=True)
    model = CRNN(nclass,model_cfg).to(device)

elif model_cfg['method'] == 'SAR':
    criterion = torch.nn.CrossEntropyLoss(ignore_index=char2index['PAD'])
    model = SAR(nclass,model_cfg).to(device)    

if not model_cfg['load_model_path'] == '':
    model.load_state_dict(torch.load(model_cfg['load_model_path']))

optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])

def adjust_learning_rate(optimizer,global_step,init_lr=train_cfg['learning_rate'], decay_rate=train_cfg['decay_rate'],decay_steps=train_cfg['decay_steps'],min_lr=train_cfg['min_lr']):
    iter_num = global_step//decay_steps
    new_leatning_rate = max(init_lr * math.pow(decay_rate,iter_num),min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_leatning_rate
    print(f'adjust learning rate to {new_leatning_rate}')


global_step = -1
batch100_loss = 0
## train
for epoch in range(train_cfg['total_epochs']):
    train_iter = iter(train_loader)
    i = -1
    loss_arr = []
    line_acc_arr = []
    min_dis = 0 
    total_dis = 0
    start_time = time.time()
    while(i<len(train_loader)-1):
        global_step +=1
        i += 1
        # if i%100 == 0 : print(i)
        # pdb.set_trace()
        try:
            images,text = train_iter.next()
        except Exception as e:
            print('Reason:', e)
            continue
        text_len = [len(item) for item in text]
        # print(f'max{np.max(text_len)}')
        # print(f'min{np.min(text_len)}')
        text_index = utils.text2index(text, char2index)
        text_index = torch.from_numpy(text_index)

        images = images.to(device)
        text_index = text_index.to(device)

        if model_cfg['method'] == 'CRNN':
            probs = model(images) # [len,b,class]
            target_length = [item.tolist().index(char2index['EOF'])+1 for item in text_index]
            target_length = torch.from_numpy(np.array(target_length)).to(device)
            input_length = torch.full(size=(images.size(0),), fill_value=probs.size(0), dtype=torch.long).to(device)
            loss = criterion(probs,text_index,input_length,target_length)
            # print(torch.autograd.gradcheck(lambda logits: torch.nn.functional.ctc_loss(probs,text_index,input_length,target_length, reduction='sum', zero_infinity=True), images, raise_exception=False))
        elif model_cfg['method'] == 'SAR':
            text_index = text_index.permute(1,0).contiguous()
            probs = model(images,text_index,global_step=global_step)
            probs_view = probs.view(-1,nclass)
            text_index_view = text_index.view(-1)  
            loss = criterion(probs_view,text_index_view) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # pdb.set_trace()
        loss_arr.append(loss.item())

        # print(loss.item())
        probs = probs.permute(1,0,2).contiguous()
        pred = probs.argmax(2)
        if model_cfg['method'] == 'CRNN':
            pred_text = utils.index2text(pred.tolist(),index2char,True)
        else:
            pred_text = utils.index2text(pred.tolist(),index2char,False)
        # print(pred_text)
        line_acc = Eval.get_line_acc(text,pred_text)
        min_dis_,total_dis_ = Eval.get_edit_distance(text,pred_text)
        line_acc_arr.append(line_acc)
        min_dis += min_dis_
        total_dis += total_dis_

        if i %  100 == 0:
            batch100_loss = np.mean(loss_arr)
            print(f'\nepoch:{epoch} step:{i} loss:{batch100_loss}  time:{time.time()-start_time} \
                line_acc:{np.mean(line_acc_arr)}  acc:{min_dis}/{total_dis} {(total_dis-min_dis)/total_dis if not total_dis == 0 else 0}')
            print(f'pred:{pred_text[0]}')
            print(f'true:{text[0]}')
            start_time = time.time()
            loss_arr = []
            line_acc_arr = []
            min_dis = 0
            total_dis = 0

        if i % train_cfg['save_step'] == 0 and i>0:
            model.eval()
            test_loader_dict = dataloader.get_test_loader(data_cfg,char2index)
            Eval.valid(model,model_cfg,test_loader_dict, index2char,device)
            model.train()
            save_path = os.path.join(train_cfg['workdir'],'checkpoints')
            utils.mkdir(save_path)
            torch.save(model.state_dict(),f'{save_path}/{model_cfg["method"]}_{epoch}_{i}.pth')

        if i%train_cfg['decay_steps'] == 0:
            adjust_learning_rate(optimizer,global_step)















