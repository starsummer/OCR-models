## introduce
Thid is the implementations of CRNN and SAR，and the 1-D attention model and 2-D attention model will be added later.

## train 
A lmdb data './data/debug' is provided for you to debug.You can train directly through the command below.
#### CRNN
'''
python train.py --config_file config/config_CRNN.py
'''

#### SAR 
'''
python train.py --config_file config/config_SAR.py
'''

## environment
python3.7,pytorch1.1 is OK.
