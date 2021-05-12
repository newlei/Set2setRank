# -- coding:UTF-8 
import torch
# print(torch.__version__) 
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys

os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [1]))

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
 
import torch.nn.functional as F
import torch.autograd as autograd 

import pdb
from collections import defaultdict
import time
import data_utils 
import evaluate
from shutil import copyfile

dataset_base_path='../MovieLens_20M'

user_num=138493
item_num=26744 
factor_num=64
batch_size=2048*512*3
top_k=20
num_negative_test_val=-1##all 
 
start_i_test=50
end_i_test=300
setp=5

run_id="Set2setRank_BPR"
print(run_id)
dataset='ml20m'
path_save_base='./log/'+dataset+'_'+run_id
if (os.path.exists(path_save_base)):
    print('has results save path')
else:
    print('error') 
    pdb.set_trace() 
result_file=open(path_save_base+'/results_hdcg_hr_all.txt','a')

path_save_model_base='./model/'+dataset+'/'+run_id

testing_user_set,testing_item_set,testing_set_count = np.load(dataset_base_path+'/testing_set.npy',allow_pickle=True)     
user_rating_set_all = np.load(dataset_base_path+'/user_rating_set_all.npy',allow_pickle=True).item()


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

print('--------test processing-------')
count, best_hr = 0, 0
for epoch in range(start_i_test,end_i_test,setp): 
    # pdb.set_trace() 
    PATH_user_emb=path_save_model_base+'/epoch'+str(epoch)+'user_e.npy'
    PATH_item_emb=path_save_model_base+'/epoch'+str(epoch)+'item_e.npy'
    user_e = np.load(PATH_user_emb)
    item_e = np.load(PATH_item_emb)
    all_pre=np.matmul(user_e,item_e.T)
    
    HR, NDCG = [], [] 
    set_all=set(range(item_num))
    #spend 461s  
    HR_all, NDCG_all = [0]*11, [0]*11
    evl_range=[10,20,30,40,50,60,70,80,90,100,110]  
    print('start--evl---')
    test_start_time = time.time()
    for u_i in testing_user_set: 
        item_i_list = list(testing_user_set[u_i])
        index_end_i=len(item_i_list)
        item_j_list = list(set_all-user_rating_set_all[u_i])#training_user_set[u_i]-val_user_set[u_i]-testing_user_set[u_i])
        item_i_list.extend(item_j_list)
        pre_one=all_pre[u_i][item_i_list]
        indices=largest_indices(pre_one, evl_range[-1])
        indices=list(indices[0])

        for top_k_index in range(len(evl_range)):
            top_k_now=evl_range[top_k_index] 
            hr_t,ndcg_t=evaluate.hr_ndcg(indices,index_end_i,top_k_now) 
            HR_all[top_k_index]+=hr_t
            NDCG_all[top_k_index]+=ndcg_t

        elapsed_time = time.time() - test_start_time  
    hr_test=np.array(HR_all)/len(testing_user_set)#round(np.mean(HR),4)
    ndcg_test=np.array(NDCG_all)/len(testing_user_set)#round(np.mean(NDCG),4)   

    str_print_evl="epoch:"+str(epoch)+'time:'+str(round(elapsed_time,2))+"\t test"+" hit:"+str(hr_test)+' ndcg:'+str(ndcg_test) 
    print("run_id: "+run_id)
    print(str_print_evl)

    result_file.write(str_print_evl)
    result_file.write('\n')
    result_file.flush()
