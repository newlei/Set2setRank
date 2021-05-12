# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys

os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [0]))

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
import s_data_utils 
import evaluate
from shutil import copyfile


 
dataset_base_path='../MovieLens_20M'
user_num=138493
item_num=26744 
factor_num=64
batch_size=2048*512*1
top_k=20
num_negative_test_val=-1##all

run_id="Set2setRank(A)_BPR"
print(run_id)
dataset='ml20m'
path_save_base='./log/'+dataset+'_'+run_id
if (os.path.exists(path_save_base)):
    print('has results save path')
else:
    os.makedirs(path_save_base)   
result_file=open(path_save_base+'/results.txt','w+') 

path_save_model_base='./model/'+dataset+'/'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)


training_user_set,training_item_set,training_set_count = np.load(dataset_base_path+'/datanpy2/training_set.npy',allow_pickle=True)
testing_user_set,testing_item_set,testing_set_count = np.load(dataset_base_path+'/datanpy2/testing_set.npy',allow_pickle=True)  
val_user_set,val_item_set,val_set_count = np.load(dataset_base_path+'/datanpy2/val_set.npy',allow_pickle=True)    
user_rating_set_all = np.load(dataset_base_path+'/datanpy2/user_rating_set_all.npy',allow_pickle=True).item()


class SelfModel(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(SelfModel, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)  

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)  

    def fun_z(self,z_ai,z_aj,flag):
        if flag:#pos pair 
            distance_pair=torch.abs(z_ai - z_aj)
            result = torch.where(distance_pair>0.5,torch.zeros_like(distance_pair)+0.5,distance_pair) 
            return result#-(1-result).log()#torch.exp(result)
        else: 
            distance_pair = -((z_ai - z_aj).sigmoid().log()) 
            return distance_pair 
    def forward(self, one_batch):    

        users_embeddings=self.embed_user.weight
        items_embeddings=self.embed_item.weight

        """Predicts the score of batch user for item."""
        user_batch=one_batch[:,0]
        item_batch=one_batch[:,1:] 
        user_i = F.embedding(user_batch,users_embeddings)#[batch,factor_num]
        item_i = F.embedding(item_batch,items_embeddings)#[batch,12,factor_num]
        pos_num_sel = item_i.shape[1]-5
        z_ai = (user_i*item_i[:,0,:]).sum(dim=-1) 
        z_aj1 = (user_i*item_i[:,1,:]).sum(dim=-1) 
        z_aj2 = (user_i*item_i[:,2,:]).sum(dim=-1) 
        z_aj3 = (user_i*item_i[:,3,:]).sum(dim=-1)  
        z_ak = (user_i*item_i[:,pos_num_sel,:]).sum(dim=-1) 
           
        pos_sim1 =self.fun_z(z_ai,z_aj1,True)
        pos_sim2 =self.fun_z(z_ai,z_aj2,True)
        pos_sim3 =self.fun_z(z_ai,z_aj3,True) 
        
        neg_sim1 = self.fun_z(z_ai,z_ak,False)
        neg_sim2 = self.fun_z(z_aj1,z_ak,False)
        neg_sim3 = self.fun_z(z_aj2,z_ak,False) 
        neg_sim4 = self.fun_z(z_aj3,z_ak,False)  
        if pos_num_sel ==2:
            loss_m6 =  (neg_sim1+neg_sim2)
            loss_m6_min = (neg_sim1+1*neg_sim2) 
        elif pos_num_sel ==3:
            loss_m6 =  (neg_sim1+neg_sim2+neg_sim3)
            loss_m6_min = (neg_sim1+1*neg_sim2+neg_sim3) 
        else:# pos_num_sel ==4:
            loss_m6 =  (neg_sim1+neg_sim2+neg_sim3+neg_sim4)
            loss_m6_min = (neg_sim1+1*neg_sim2+1*neg_sim3+neg_sim4) 
        for  i in range(pos_num_sel+1,item_i.shape[1]):
            item_i_one = item_i[:,i,:]
            z_ak= (user_i*item_i_one).sum(dim=-1)#+user_i_bias + item_i_bias[:,i]
            neg_sim1= self.fun_z(z_ai,z_ak,False)
            neg_sim2= self.fun_z(z_aj1,z_ak,False)
            neg_sim3= self.fun_z(z_aj2,z_ak,False)
            neg_sim4 = self.fun_z(z_aj3,z_ak,False)  
            if pos_num_sel ==2:
                one_pn =  (neg_sim1+neg_sim2) 
            elif pos_num_sel ==3:
                one_pn =  (neg_sim1+neg_sim2+neg_sim3) 
            else:#  pos_num_sel ==4:
                one_pn =  (neg_sim1+neg_sim2+neg_sim3+neg_sim4)  
            loss_m6 +=  one_pn#(neg_sim1+neg_sim2)#M6
            loss_m6_min = torch.where(one_pn>loss_m6_min,one_pn,loss_m6_min) 
        if pos_num_sel ==2: 
            loss_posdis = -((pos_sim1-loss_m6_min).sigmoid().log()).mean()
        elif pos_num_sel ==3:
            loss_posdis = -((pos_sim1+pos_sim2-loss_m6_min).sigmoid().log()).mean()
        else:#  pos_num_sel ==4:
            loss_posdis = -((pos_sim1+pos_sim2+pos_sim3-loss_m6_min).sigmoid().log()).mean()
        loss_pre = 1*loss_posdis + 1*(loss_m6).mean()
        l2_regulization =0.01*(user_i**2+(item_i**2).sum(dim=1)).sum(dim=-1).mean() 
        loss = loss_pre+l2_regulization#/0.1#/20.0
        return loss,l2_regulization 
    def getemb(self):
        users_embeddings=self.embed_user.weight
        items_embeddings=self.embed_item.weight 
        return users_embeddings,items_embeddings




train_load_path1=dataset_base_path+'/traing_sample_neg5'
train_dataset1 = s_data_utils.SelfDataSpeed(
        train_dict=training_user_set, is_training=True,\
        load_path=train_load_path1,load_num=400)
train_loader1 = DataLoader(train_dataset1,
        batch_size=batch_size, shuffle=True, num_workers=4)


train_load_path2=dataset_base_path+'/traing_sample_pos3_neg5'
train_dataset2 = s_data_utils.SelfDataSpeed(
        train_dict=training_user_set, is_training=True,\
        load_path=train_load_path2,load_num=400)
train_loader2 = DataLoader(train_dataset2,
        batch_size=batch_size, shuffle=True, num_workers=4)


train_load_path3=dataset_base_path+'/traing_sample_pos4_neg5'
train_dataset3 = s_data_utils.SelfDataSpeed(
        train_dict=training_user_set, is_training=True,\
        load_path=train_load_path3,load_num=400)
train_loader3 = DataLoader(train_dataset3,
        batch_size=batch_size, shuffle=True, num_workers=4)


test_load_path1=dataset_base_path+'/testing_sample_neg5' 
testing_dataset_loss1 = s_data_utils.SelfDataSpeed(
        train_dict=testing_user_set, is_training=False,\
        load_path=test_load_path1,load_num=20)
testing_loader_loss1 = DataLoader(testing_dataset_loss1,
        batch_size=batch_size, shuffle=False, num_workers=4)


test_load_path2=dataset_base_path+'/testing_sample_pos3_neg5' 
testing_dataset_loss2 = s_data_utils.SelfDataSpeed(
        train_dict=testing_user_set, is_training=False,\
        load_path=test_load_path2,load_num=20)
testing_loader_loss2 = DataLoader(testing_dataset_loss2,
        batch_size=batch_size, shuffle=False, num_workers=4)


test_load_path3=dataset_base_path+'/testing_sample_pos4_neg5' 
testing_dataset_loss3 = s_data_utils.SelfDataSpeed(
        train_dict=testing_user_set, is_training=False,\
        load_path=test_load_path3,load_num=20)
testing_loader_loss3 = DataLoader(testing_dataset_loss3,
        batch_size=batch_size, shuffle=False, num_workers=4)

val_load_path1=dataset_base_path+'/val_sample_neg5'
val_dataset_loss1 = s_data_utils.SelfDataSpeed(
        train_dict=val_user_set, is_training=False,\
        load_path=val_load_path1,load_num=10)
val_loader_loss1 = DataLoader(val_dataset_loss1,
        batch_size=batch_size, shuffle=False, num_workers=4)

val_load_path2=dataset_base_path+'/val_sample_pos3_neg5'
val_dataset_loss2 = s_data_utils.SelfDataSpeed(
        train_dict=val_user_set, is_training=False,\
        load_path=val_load_path2,load_num=10)
val_loader_loss2 = DataLoader(val_dataset_loss2,
        batch_size=batch_size, shuffle=False, num_workers=4)
 
val_load_path3=dataset_base_path+'/val_sample_pos4_neg5'
val_dataset_loss3 = s_data_utils.SelfDataSpeed(
        train_dict=val_user_set, is_training=False,\
        load_path=val_load_path3,load_num=10)
val_loader_loss3 = DataLoader(val_dataset_loss3,
        batch_size=batch_size, shuffle=False, num_workers=4)


model = SelfModel(user_num, item_num, factor_num)
model=model.to('cuda')

optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.005)#, betas=(0.5, 0.99))




########################### TRAINING #####################################
print('--------training processing-------')
count, best_hr = 0, 0
for epoch in range(350):
    model.train() 
    start_time = time.time()
    pos_num_sel =np.random.randint(3)
    if pos_num_sel==0: 
        train_loader1.dataset.ng_sample()
        train_loader = train_loader1
        testing_loader_loss1.dataset.ng_sample()
        testing_loader_loss = testing_loader_loss1
        val_loader_loss1.dataset.ng_sample()
        val_loader_loss = val_loader_loss1
    elif pos_num_sel==1: 
        train_loader2.dataset.ng_sample()
        train_loader = train_loader2
        testing_loader_loss2.dataset.ng_sample()
        testing_loader_loss = testing_loader_loss2
        val_loader_loss2.dataset.ng_sample()
        val_loader_loss = val_loader_loss2
    else: #pos_num_sel==2: 
        train_loader3.dataset.ng_sample()
        train_loader = train_loader3
        testing_loader_loss3.dataset.ng_sample()
        testing_loader_loss = testing_loader_loss3
        val_loader_loss3.dataset.ng_sample()
        val_loader_loss = val_loader_loss3
    elapsed_time = time.time() - start_time
    print(run_id+' train data of ng_sample is  end '+str(elapsed_time))

    train_loss_sum=[]
    train_loss_sum2=[]
    for list_one in train_loader:
        list_one = list_one.cuda()  
        model.zero_grad()
        loss_get,l2_regu_get = model(list_one)
        loss = loss_get.mean()
        l2_regu = l2_regu_get.mean()
        loss.backward()
        optimizer_bpr.step()
        count += 1
        # pdb.set_trace()
        train_loss_sum.append(loss.item())  
        train_loss_sum2.append(l2_regu.item())  

    elapsed_time = time.time() - start_time
    train_loss=round(np.mean(train_loss_sum[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    train_loss2=round(np.mean(train_loss_sum2[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+'\t'
    tain_loss_str = 'train loss:'+str(train_loss)+"="+str(train_loss2)+"+" 
    str_print_train += format(tain_loss_str,'<26')
    print('--train--',elapsed_time)

    get_users_embedding, get_items_embedding = model.getemb()
    user_e=get_users_embedding.cpu().detach().numpy()
    item_e=get_items_embedding.cpu().detach().numpy() 
    PATH_user_emb=path_save_model_base+'/epoch'+str(epoch)+'user_e.npy'
    PATH_item_emb=path_save_model_base+'/epoch'+str(epoch)+'item_e.npy'
    np.save(PATH_user_emb,user_e)
    np.save(PATH_item_emb,item_e)
    model.eval()
    # ######test and val########### 
    val_loss=evaluate.metrics_loss(model,val_loader_loss,batch_size) 
#     testing_loader_loss.dataset.ng_sample()
    test_loss=evaluate.metrics_loss(model,testing_loader_loss,batch_size)
    val_loss_str = ' val loss:'+str(round(val_loss,4))
    test_loss_str = ' test loss:'+str(round(test_loss,4))
    val_test_str = format(val_loss_str,'<19')+format(test_loss_str,'<20')
    print(str_print_train+val_test_str) 
    result_file.write(str_print_train+val_test_str) 
    result_file.write('\n') 
    result_file.flush()
    