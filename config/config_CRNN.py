model_cfg = {
    'method': 'CRNN',
    'nc': 3, # 1或3，输入图片通道
    'hidden_size': 512, 
    'load_model_path': '', 
    'test_mode': False,
    'num_embeddings': 512, 
}

train_cfg = {
    'total_epochs': 3 ,
    'batch_size': 64,
    'test_batch_size': 16, 
    'workdir': './work_dirs/CRNN',
    'learning_rate': 0.001, #初始学习率
    'decay_rate': 0.9, #学习率衰减速率
    'decay_steps': 20000, #每多少步衰减
    'min_lr': 0.00001, #最小学习率
    'save_step': 10000, #存储model
    'optimizer': 'Adam',
}

data_cfg  = {
    'dict_file': 'Dict/dict_eng.txt',  #字典
    'train_lmdb': ['./data/debug'], 
    'test_root' : '',
    'test_lmdb': ['./data/debug'],
    'target_H': 48, #resize到固定宽度
    'fix_width': True, #如果fix width，图片都会resize到max width
    'max_width': 160,
    'nc': model_cfg['nc'],
    'batch_size': train_cfg['batch_size'],
    'test_batch_size': train_cfg['test_batch_size'],
}
