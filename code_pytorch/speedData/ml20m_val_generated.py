# -- coding:UTF-8
 
import argparse
import os
import numpy as np
import math
import sys
 
import pdb
from collections import defaultdict
import time
import evaluate
from shutil import copyfile 
from multiprocessing import Pool
from itertools import repeat
import copy

dataset_base_path='../MovieLens_20M'
 
##gowalla
user_num=138493
item_num=26744

path_save_base=dataset_base_path+'/val_sample_neg5'
if (os.path.exists(path_save_base)):
    print('has results save path')
else:
    os.makedirs(path_save_base)   
# result_file=open(path_save_base+'/results.txt','w+')#('./log/results_gcmc.txt','w+')
 

training_user_set,training_item_set,training_set_count = np.load(dataset_base_path+'/training_set.npy',allow_pickle=True)
testing_user_set,testing_item_set,testing_set_count = np.load(dataset_base_path+'/testing_set.npy',allow_pickle=True)  
val_user_set,val_item_set,val_set_count = np.load(dataset_base_path+'/val_set.npy',allow_pickle=True)    
user_rating_set_all = np.load(dataset_base_path+'/user_rating_set_all.npy',allow_pickle=True).item()


 

def ng_sample(num_ng,num_item,dataset_save_path,id):
    # assert self.is_training, 'no need to sampling when testing'
    # print('ng_sample----is----call-----')
    train_dict = copy.deepcopy(training_user_set) 
    print(id) 
    np.random.seed(id) 
    all_pair_len=0
    for user_id in train_dict:
        positive_list=train_dict[user_id]
        len_pos = int(len(positive_list)/2)
        all_pair_len+=len_pos 
    set_all_item = np.array(range(num_item))
    set_all_item = np.concatenate((set_all_item,set_all_item),axis=0)
    features_fill = (np.zeros([all_pair_len,3+num_ng])).astype(np.int)

    features_fill_idx_start =0
    features_fill_idx_end =0
    for user_id in train_dict:
        positive_list=train_dict[user_id]
        len_pos = int(len(positive_list)/2)
        uid_list=np.array([user_id]*len_pos).reshape(-1,1)
        pos_list=np.array(list(positive_list)).reshape(-1,1)
        np.random.shuffle(pos_list)
        np.random.shuffle(set_all_item)
        neg_list=set_all_item[:len_pos*num_ng].reshape(len_pos,-1) 
        x1=np.concatenate((uid_list, pos_list[:len_pos],pos_list[len_pos:2*len_pos],neg_list), axis=1)
        features_fill_idx_end = features_fill_idx_start+len_pos
        features_fill[features_fill_idx_start:features_fill_idx_end]=x1
        features_fill_idx_start = features_fill_idx_end
        # pdb.set_trace()

    path_save=dataset_save_path+'/'+str(id)+'.npy'
    np.save(path_save,features_fill)
    print(str(id)+'end')

p = Pool(10)
arg_all = zip(repeat(5),repeat(item_num),repeat(path_save_base),range(0,20))       
p.starmap(ng_sample,arg_all) 
p.close()
p.join()
print('Pool  end') 
exit()
