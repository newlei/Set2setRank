import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore import Model
from mindspore import dataset as ds
import mindspore.ops as ops
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.numpy as msnp
import numpy as np
import math
import time
import pdb
import os

import evaluate


user_num=138493
item_num=26744 
factor_num=64

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.
    Args:
        network (Cell): The target network to wrap.
    """
    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, img_A, img_B):
        loss, _, _, _, _, _, _ = self.network(img_A, img_B)
        return loss


class BPR(nn.Cell):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR, self).__init__()
        """ 
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        three module: LineGCN, AvgReadout, Discriminator
        """      
        self.user_idx_all = ms.Tensor(np.array(range(user_num)))
        self.item_idx_all = ms.Tensor(np.array(range(item_num)))
        self.embed_user = nn.Embedding(user_num, factor_num, embedding_table='normal')
        self.embed_item = nn.Embedding(item_num, factor_num, embedding_table='normal')
        self.sum = ops.ReduceSum(keep_dims=True)
        self.sigmoid = nn.Sigmoid()
        self.log = ops.Log()
        self.abs = ops.Abs()
        self.zeroslike = ops.ZerosLike()
        self.squeeze = ops.Squeeze()

    def fun_z(self,z_ai,z_aj,flag):
        if flag:#pos pair 
            distance_pair=self.abs(z_ai - z_aj) 
            result = msnp.where(distance_pair>0.5,self.zeroslike(distance_pair)+0.5,distance_pair)
            return result
        else: 
            # distance_pair = -((z_ai - z_aj).sigmoid().log())
            distance_pair = -self.log(self.sigmoid(z_ai - z_aj))
            return distance_pair 

    def construct(self, user_id,item_i,item_j):
        user_emb = self.embed_user(user_id)
        

        item_i_emb = self.embed_item(item_i)
        item_j_emb = self.embed_item(item_j)
        
        z_ai = self.sum(user_emb*item_i_emb[:,0,:],-1)
        z_aj = self.sum(user_emb*item_i_emb[:,1,:],-1)
        z_ak = self.sum(user_emb*item_j_emb[:,0,:],-1)
        pos_sim1 =self.fun_z(z_ai,z_aj,True)
        pos_sim2 =self.fun_z(z_aj,z_ai,True)
        neg_sim1 = self.fun_z(z_ai,z_ak,False)
        neg_sim2 = self.fun_z(z_aj,z_ak,False)
        loss_m6 =  (neg_sim1+neg_sim2)#M6
        loss_m6_min = (neg_sim1+neg_sim2)

        for  i in range(1,item_i.shape[1]):
            item_i_one = item_j_emb[:,i,:]
            z_ak= self.sum(user_emb*item_i_one,-1)#+user_i_bias + item_i_bias[:,i]
            neg_sim1= self.fun_z(z_ai,z_ak,False)
            neg_sim2= self.fun_z(z_aj,z_ak,False)
            one_pn = (neg_sim1+neg_sim2)
            loss_m6 +=  one_pn
            loss_m6_min = msnp.where(one_pn>loss_m6_min,one_pn,loss_m6_min)
        
        # loss_posdis = -((pos_sim1*2-loss_m6_min).sigmoid().log()).mean() 
        loss_posdis = -self.log(self.sigmoid(pos_sim1*2-loss_m6_min)).mean()
        loss_pre = 0.5*loss_posdis + 1*(loss_m6).mean()

        l2_regulization =user_emb**2+self.squeeze(self.sum(item_i_emb**2,1))+self.squeeze(self.sum(item_j_emb**2,1))
        l2_regulization = 0.001*self.sum(l2_regulization,-1).mean()
        loss = loss_pre+l2_regulization
        return loss
        # predict_ui = self.sum(user_emb*item_i_emb,-1)#.sum(dim=-1)
        # predict_uj = self.sum(user_emb*item_j_emb,-1)#.sum(dim=-1)
        # loss = -self.log(self.sigmoid(predict_ui - predict_uj))
        # return loss.mean()

    def get_emb(self):
        users_embeddings=self.embed_user(self.user_idx_all)#.weights
        items_embeddings=self.embed_item(self.item_idx_all)#.weights  
        return users_embeddings,items_embeddings



# bpr = BPR(user_num=10, item_num=10, factor_num=64)
# bpr(ms.Tensor([2,1]),ms.Tensor([1,1]),ms.Tensor([3,4]))
# pdb.set_trace()

class MyAccessible:
    def __init__(self,train_dict=None,num_item=0, num_ng=1, data_set_count=0,batch_size=1):
        self.num_item = num_item
        self.train_dict = train_dict
        self.num_ng = num_ng
        # self.data_set_count = data_set_count
        self.set_all_item = np.array(range(num_item))#set(range(num_item))
        self.batch_size = batch_size

        self.all_pair_len=0
        for user_id in self.train_dict:
            positive_list=self.train_dict[user_id]
            len_pos = int(len(positive_list)/2)
            self.all_pair_len+=len_pos
        self.features_fill = (np.zeros([self.all_pair_len,3+num_ng])).astype(np.int)

        self.len_all = math.ceil(1.0*self.all_pair_len/self.batch_size)

    def permutation(self):  
        range_data = self.all_pair_len
        seed = np.random.randint(low=0, high=np.iinfo(np.int32).max, dtype=np.int32)
        state = np.random.RandomState(seed=seed)
        output = np.arange(range_data, dtype=np.int32)  
        state.shuffle(output)
        #add last batch
        sup_data = (self.len_all*self.batch_size) - self.all_pair_len
        if sup_data>0: 
            output = np.append(output, output[:sup_data], axis=0)  
        return output

    def ng_sample(self): 
        self.indices_rand = self.permutation() 

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

    def __getitem__(self, index):  
        # return self.features_fill[:,0],self.features_fill[:,1],self.features_fill[:,2]
        start_index = index*self.batch_size
        end_index = start_index+self.batch_size 
        index_get = self.indices_rand[start_index:end_index] 
        data_batch = self.features_fill[index_get]
        return data_batch[:,0], data_batch[:,1:3], data_batch[:,3:]

    def __len__(self):
        return self.len_all


class TrainOneStep(nn.Cell):
    def __init__(self, model_bpr, optimizer, sens=1.0):
        super(TrainOneStep, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.model_bpr = model_bpr
        self.model_bpr.set_grad()
        self.model_bpr.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ms.ParameterTuple(model_bpr.trainable_params())
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, user_batch, item_i_batch, item_j_batch):
        weights = self.weights
        loss_batch = self.model_bpr(user_batch, item_i_batch,item_j_batch) 
        sens_batch = ops.Fill()(ops.DType()(loss_batch), ops.Shape()(loss_batch), self.sens)
        grads_batch = self.grad(self.model_bpr, weights)(user_batch, item_i_batch,item_j_batch, sens_batch)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_batch = self.grad_reducer(grads_batch)
        return ops.depend(loss_batch, self.optimizer(grads_batch))


training_user_set,training_item_set,training_set_count = np.load('../code_pytorch/MovieLens_20M/training_set.npy',allow_pickle=True)
testing_user_set,testing_item_set,testing_set_count = np.load('../code_pytorch/MovieLens_20M/testing_set.npy',allow_pickle=True)  
val_user_set,val_item_set,val_set_count = np.load('../code_pytorch/MovieLens_20M/val_set.npy',allow_pickle=True)    
user_rating_set_all = np.load('../code_pytorch/MovieLens_20M/user_rating_set_all.npy',allow_pickle=True).item()

data_loader=MyAccessible(train_dict=training_user_set,num_item=item_num, num_ng=5, data_set_count=training_set_count,batch_size=2048*32)
dataset = ds.GeneratorDataset(source=data_loader, column_names=["user","item_i","item_j"])

loss_bpr = BPR(user_num=user_num, item_num=item_num, factor_num=64)
optimizer_bpr = nn.Adam(loss_bpr.trainable_params(),learning_rate=0.005)
net_bpr = TrainOneStep(loss_bpr, optimizer_bpr)

run_id='t0'
result_file=open('./'+run_id+'_results.txt','a')
print(run_id)

path='../model/'+run_id 
if (os.path.exists(path)):
    print('has model save path')
else:
    os.makedirs(path)

max_epoch=100
for epoch in range(max_epoch):
    start_time = time.time()
    data_loader.ng_sample()
    ng_time = time.time() - start_time
    print(' train data of ng_sample is end',str(round(ng_time,1)))
    
    train_loss_sum=[]
    for user,item_i,item_j in dataset: 
        loss_res=net_bpr(user, item_i, item_j) 
        train_loss_sum.append(loss_res.mean().asnumpy())
    train_loss=round(np.mean(train_loss_sum[:-1]),4) 
    elapsed_time = time.time() - start_time
    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+'\t train loss:'+str(train_loss)
    print(str_print_train)


    u_emb,i_emb=loss_bpr.get_emb()
    user_e = u_emb.asnumpy()
    item_e = i_emb.asnumpy()
    PATH_user_emb='../model/'+run_id+'/epoch'+str(epoch)+'user_e.npy'
    PATH_item_emb='../model/'+run_id+'/epoch'+str(epoch)+'item_e.npy'
    np.save(PATH_user_emb,user_e)
    np.save(PATH_item_emb,item_e)
    
    if epoch%10 == 0:
        print('--epoch/10=0')
    else:
        result_file.write(str_print_train)
        result_file.write('\n')
        result_file.flush()
        continue
    
    all_pre=np.matmul(user_e,item_e.T)
    HR, NDCG = [], [] 
    set_all=set(range(item_num))
    #spend 461s  
    # HR_all, NDCG_all = [0]*11, [0]*11
    # evl_range=[10,20,30,40,50,60,70,80,90,100]
    HR_all, NDCG_all = [0]*2, [0]*2
    evl_range=[10]
    # pdb.set_trace()
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
    print(hr_test,ndcg_test)
    str_hr_ndcg='\t'+str(hr_test[0])+str(ndcg_test[0])
    result_file.write(str_print_train+str_hr_ndcg)
    result_file.write('\n')
    result_file.flush()


