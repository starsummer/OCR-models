import os
import tools.utils as utils
import numpy as np

def minEditDistance(str1, str2):
    # str2 = str2.replace('EOF','')
    str1 = str1.lower()
    str2 = str2.lower()

    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1
    # 创建矩阵
    matrix = [0 for n in range(len_str1 * len_str2)]
    #矩阵的第一行
    for i in range(len_str1):
        matrix[i] = i
    # 矩阵的第一列
    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1
     # 根据状态转移方程逐步得到编辑距离
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i-1] == str2[j-1]:
                cost = 0
            else:
                cost = 1
            matrix[j*len_str1+i] = min(matrix[(j-1)*len_str1+i]+1,
                    matrix[j*len_str1+(i-1)]+1,
                    matrix[(j-1)*len_str1+(i-1)] + cost)
    return matrix[-1] 


def get_line_acc(list1,list2):
    assert len(list1) == len(list2)
    cnt = 0
    for i in range(len(list1)):
        # print(list1[i],list2[i])
        if list1[i].lower() == list2[i].lower():
            cnt += 1
    acc = cnt*1.0/len(list1)
    # print(f'the list acc is {acc}\n')
    return acc

def get_edit_distance(list1,list2):
    assert len(list1) == len(list2)
    total_length = 0
    dis = 0
    for i in range(len(list2)):
        total_length += len(list1[i])
        dis += minEditDistance(list1[i], list2[i])
    return dis,total_length


def filter_list(line_list):
    rel = [filter_line(line) for line in line_list]

def filter_line(line):
    line = re.sub(r'\s+',' ',line)
    return line


#测试
def valid(model,model_cfg,test_loader_dict,index2char,device):
    for dataset_name in test_loader_dict:
        test_loader = test_loader_dict[dataset_name] #获取测试集loader
        test_iter = iter(test_loader)
        max_iter = len(test_loader)
        line_acc_arr = []
        min_dis = 0
        total_dis = 0
        for i in range(max_iter):
            try:
                images,text = test_iter.next()
            except Exception as e:
                print('Reason:', e)
                continue
            images = images.to(device)
            # text_index = torch.from_numpy(text_index)
            # text_index = text_index.to(device)
            if model_cfg['method'] == 'CRNN':
                probs = model(images) # [len,b,class]
            elif model_cfg['method'] == 'SAR':
                probs = model(images,'',test=True)    

            probs = probs.permute(1,0,2).contiguous()
            pred = probs.argmax(2)
            if model_cfg['method'] == 'CRNN':
                pred_text = utils.index2text(pred.tolist(),index2char,True)
            else:
                pred_text = utils.index2text(pred.tolist(),index2char,True)
            # print(pred_text)
            line_acc = get_line_acc(text,pred_text)
            min_dis_,total_dis_ = get_edit_distance(text,pred_text)
            line_acc_arr.append(line_acc)
            min_dis += min_dis_
            total_dis += total_dis_
        print(f'{dataset_name}: line_acc: {np.mean(line_acc_arr)}  acc:{total_dis-min_dis}/{total_dis} {(total_dis-min_dis)/total_dis if not total_dis == 0 else 0}')





if __name__ == '__main__':
    # print(minEditDistance('abc','abcdfsa'))
    a =['a','ab','abc']
    b =['v','ab','acv']
    print(get_line_acc(a,b))
    print(get_edit_distance(a,b))


