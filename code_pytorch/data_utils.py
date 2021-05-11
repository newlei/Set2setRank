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
#         pdb.set_trace()
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] 
        return user, item_i, item_j 
#         return torch.LongTensor(user), torch.LongTensor(item_i), torch.LongTensor(item_j)


class BPRData(data.Dataset):
    def __init__(self,train_dict=None,num_item=0, num_ng=1, is_training=None, data_set_count=0,all_rating=None):
        super(BPRData, self).__init__()

        self.num_item = num_item
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count
        self.all_rating=all_rating
        self.set_all_item=set(range(num_item)) 

    def ng_sample(self):
        # assert self.is_training, 'no need to sampling when testing'
        # print('ng_sample----is----call-----') 
        self.features_fill = []
        for user_id in self.train_dict:
            positive_list=self.train_dict[user_id]#self.train_dict[user_id]
            all_positive_list = self.all_rating[user_id]
            #item_i: positive item ,,item_j:negative item   
            # temp_neg=list(self.set_all_item-all_positive_list)
            # random.shuffle(temp_neg)
            # count=0
            # for item_i in positive_list:
            #     for t in range(self.num_ng):   
            #         self.features_fill.append([user_id,item_i,temp_neg[count]])
            #         count+=1  
            for item_i in positive_list:   
                for t in range(self.num_ng):
                    item_j=np.random.randint(self.num_item)
                    while item_j in all_positive_list:
                        item_j=np.random.randint(self.num_item)
                    self.features_fill.append([user_id,item_i,item_j]) 
      
    def __len__(self):  
        return self.num_ng*self.data_set_count#return self.num_ng*len(self.train_dict) 
         

    def __getitem__(self, idx):
        features = self.features_fill  
        
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] 
        return user, item_i, item_j 


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
            # pdb.set_trace()

    def __len__(self):  
        return self.all_pair_len
        # return self.num_ng*self.data_set_count#return self.num_ng*len(self.train_dict) 
    
    def __getitem__(self, idx):
        features = self.features_fill
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] 
        return torch.LongTensor(features[idx])


class SelfresData(data.Dataset):
    def __init__(self,train_dict=None,batch_size=0,num_item=0,all_pos=None):
        super(SelfresData, self).__init__() 
      
        self.train_dict = train_dict 
        self.batch_size = batch_size
        self.all_pos_train=all_pos 

        self.features_fill = []
        for user_id in self.train_dict:
            self.features_fill.append(user_id)
        self.set_all=set(range(num_item))
   
    def __len__(self):  
        return math.ceil(len(self.train_dict)*1.0/self.batch_size)#这里的self.data_set_count==batch_size
         

    def __getitem__(self, idx): 
        
        user_test=[]
        item_test=[]
        split_test=[]
        for i in range(self.batch_size):#这里的self.data_set_count==batch_size 
            index_my=self.batch_size*idx+i 
            if index_my == len(self.train_dict):
                break   
            user = self.features_fill[index_my]
            item_i_list = list(self.train_dict[user])
            item_j_list = list(self.set_all-self.all_pos_train[user])
            # pdb.set_trace() 
            u_i=[user]*(len(item_i_list)+len(item_j_list))
            user_test.extend(u_i)
            item_test.extend(item_i_list)
            item_test.extend(item_j_list)  
            split_test.append([(len(item_i_list)+len(item_j_list)),len(item_j_list)]) 
           
        #实际上只用到一半去计算，不需要j的。
        return torch.from_numpy(np.array(user_test)), torch.from_numpy(np.array(item_test)), split_test           
 
 


class SpeedData(data.Dataset):
    def __init__(self,train_dict=None,num_item=0, num_ng=1, is_training=None, data_set_count=0):
        super(SpeedData, self).__init__()

        self.num_item = num_item
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count 
        self.set_all_item = np.array(range(num_item))

        all_pair_len = self.num_ng*self.data_set_count
        self.features_fill = (np.zeros([all_pair_len,3])).astype(np.int)

    def ng_sample(self):#60s
        features_fill_idx_start =0
        features_fill_idx_end =0
        for user_id in self.train_dict:
            positive_list=self.train_dict[user_id]
            len_pos = len(positive_list)
            
            uid_list=np.array([user_id]*len_pos).reshape(-1,1)
            pos_list=np.array(list(positive_list)).reshape(-1,1)
            np.random.shuffle(self.set_all_item)
            neg_list=self.set_all_item[:len_pos].reshape(-1,1)
            x1=np.concatenate((uid_list, pos_list,neg_list), axis=1)
            features_fill_idx_end = features_fill_idx_start+len_pos
            self.features_fill[features_fill_idx_start:features_fill_idx_end]=x1
            features_fill_idx_start = features_fill_idx_end
            # pdb.set_trace()

    def __len__(self):  
        return self.num_ng*self.data_set_count#return self.num_ng*len(self.train_dict) 
    
    def __getitem__(self, idx):
        features = self.features_fill
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] 
        return user, item_i, item_j 



    # def ng_sample2(self):#60s
    #     features_fill_idx_start =0 
    #     features_fill_idx_end =0 
    #     for user_id in self.train_dict:
    #         positive_list=self.train_dict[user_id]
    #         len_pos = len(positive_list)
            
    #         uid_list=np.array([user_id]*len_pos).reshape(-1,1)
    #         pos_list=np.array(list(positive_list)).reshape(-1,1)
    #         np.random.shuffle(self.set_all_item)
    #         neg_list=self.set_all_item[:len_pos].reshape(-1,1)
    #         x1=np.concatenate((uid_list, pos_list,neg_list), axis=1)
    #         features_fill_idx_end = features_fill_idx_start+len_pos
    #         self.features_fill[features_fill_idx_start:features_fill_idx_end]=x1
    #         features_fill_idx_start = features_fill_idx_end

    # def ng_sample1(self): #150s
    #     self.features_fill = []
    #     for user_id in self.train_dict:
    #         positive_list=self.train_dict[user_id]
    #         for item_i in positive_list:
    #             for t in range(self.num_ng):
    #                 item_j=np.random.randint(self.num_item)
    #                 # while item_j in all_positive_list:
    #                 #     item_j=np.random.randint(self.num_item)
    #                 self.features_fill.append([user_id,item_i,item_j]) 