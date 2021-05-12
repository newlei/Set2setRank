# -- coding:UTF-8
import numpy as np 
import pandas as pd 
import scipy.sparse as sp 

import torch.utils.data as data
import pdb
from torch.autograd import Variable
import torch
import math
import random 
# from prefetch_generator import BackgroundGenerator

class SelfData(data.Dataset):
    def __init__(self,train_dict=None,num_item=0, num_ng=1, is_training=None, data_set_count=0):
        super(SelfData, self).__init__() 
        self.num_item = num_item
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count 
        self.set_all_item = np.array(range(num_item))
        self.set_all_item = np.concatenate((self.set_all_item,self.set_all_item),axis=0)
        
        self.all_pair_len=0
        for user_id in self.train_dict:
            positive_list=self.train_dict[user_id]
            len_pos = int(len(positive_list)/2)
            self.all_pair_len+=len_pos 
        # all_pair_len = self.num_ng*self.data_set_count
        self.features_fill = (np.zeros([self.all_pair_len,3+num_ng])).astype(np.int)

    def ng_sample(self):#60s
        features_fill_idx_start =0
        features_fill_idx_end =0
        for user_id in self.train_dict:
            positive_list=self.train_dict[user_id]
            len_pos = int(len(positive_list)/2)
            uid_list=np.array([user_id]*len_pos).reshape(-1,1)
            pos_list=np.array(list(positive_list)).reshape(-1,1)
            np.random.shuffle(pos_list)
            np.random.shuffle(self.set_all_item)
            neg_list=self.set_all_item[:len_pos*self.num_ng].reshape(len_pos,-1) 
            x1=np.concatenate((uid_list, pos_list[:len_pos],pos_list[len_pos:2*len_pos],neg_list), axis=1)
            features_fill_idx_end = features_fill_idx_start+len_pos
            self.features_fill[features_fill_idx_start:features_fill_idx_end]=x1
            features_fill_idx_start = features_fill_idx_end

    def __len__(self):  
        return self.all_pair_len 
    
    def __getitem__(self, idx):
        features = self.features_fill
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] 
        return torch.LongTensor(features[idx])


class SelfDataSpeed(data.Dataset):
    def __init__(self,train_dict=None, is_training=None, load_path=None,load_num=0):
        super(SelfDataSpeed, self).__init__()  
        self.train_dict = train_dict 
        self.is_training = is_training 
        self.all_pair_len=0 
        self.load_num=load_num
        self.load_path=load_path
        path_one= self.load_path+'/'+str(0)+'.npy'
        self.features_fill=np.load(path_one)
        self.all_pair_len = len(self.features_fill)

    def ng_sample(self):#60s
        id_load=np.random.randint(self.load_num)
        path_one= self.load_path+'/'+str(id_load)+'.npy'
        self.features_fill=np.load(path_one)

    def __len__(self):
        return self.all_pair_len 
    
    def __getitem__(self, idx):
        features = self.features_fill 
        return torch.LongTensor(features[idx])

