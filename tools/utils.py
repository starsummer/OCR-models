import numpy as np
import os,sys
from importlib import import_module
import re
def get_dict(dict_file):
    index2char = dict()
    char2index = dict()
    with open(dict_file) as f:
        i =0
        index2char[i] = 'blank'
        char2index['blank'] = i    

        i += 1
        index2char[i] = 'EOF'
        char2index['EOF'] = i    

        i += 1
        index2char[i] = 'PAD'
        char2index['PAD'] = i
        i += 1
        for line in f:
            line = line.replace('\n','')
            index2char[i]=line
            char2index[line]=i
            i+=1
    return index2char,char2index
    # print(index2char)

def text2index(text,char2index,pad_same = True): #将字符转化为字典中对应的index,以及EOF,再加pad补齐为相同长度
    max_len = 0
    for label in text:
        if len(label)>max_len:
            max_len = len(label)
    max_len +=2 # EOF,PAD
    rel = []
    for label in text:
        text_index = []
        for char in label:
            if char not in char2index:
                print(char + ' not in dict')
                continue
            text_index.append(char2index[char])
        text_index.append(char2index['EOF'])
        if pad_same:
            pad_length = max_len-len(label)-1
            for j in range(pad_length):
                text_index.append(char2index['PAD'])

        rel.append(text_index)
    rel = np.array(rel)
    return rel

def index2text(index_arr,index2char,ctc=False):
    line_arr = []
    for index_seq in index_arr:
        line = ''

        if ctc: #
            for index in index_seq:
                char = index2char[index]
                if char == 'EOF':
                    break
                if char == 'blank':
                    line += '&&blank&&'
                else:
                    line += char
            items = [remove_same(item) for item in line.split('&&blank&&')]
            line = ''.join(items)
        else: #直接将index转化为text
            for index in index_seq:
                char = index2char[index]
                if char == 'EOF':
                    break
                line += char
        line_arr.append(line)
    return line_arr

def mkdir(path):
    if os.path.exists(path):
        return
    os.makedirs(path)

def read_config_file(file_name):
    assert file_name.endswith('.py')
    config_dir = os.path.dirname(file_name)
    sys.path.insert(0,config_dir)
    model_name = os.path.basename(file_name).split('.')[0]
    mod = import_module(model_name)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }
    
    return cfg_dict['model_cfg'],cfg_dict['train_cfg'],cfg_dict['data_cfg']

def remove_same(line):
    rel = ''
    last_char = ''
    for char in line:
        if char == last_char:
            continue
        rel += char
        last_char = char
    return rel




if __name__ == '__main__':
    # model_cfg,train_cfg,data_cfg = read_config_file('../config/config_SAR.py')
    # print(model_cfg)
    lines = ['1,1,0,1,0','1,2,3,4','1,2,3,3,3,3,0,3']
    print(index2text(lines))

    pass
