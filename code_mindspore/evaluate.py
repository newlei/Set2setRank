# -- coding:UTF-8
import numpy as np 
import time
import pdb
import math 

def hr_ndcg(indices_sort_top,index_end_i,top_k): 
    hr_topK=0
    ndcg_topK=0

    ndcg_max=[0]*top_k
    temp_max_ndcg=0
    for i_topK in range(top_k):
        temp_max_ndcg+=1.0/math.log(i_topK+2)
        ndcg_max[i_topK]=temp_max_ndcg

    max_hr=top_k
    max_ndcg=ndcg_max[top_k-1]
    if index_end_i<top_k:
        max_hr=(index_end_i)*1.0
        max_ndcg=ndcg_max[index_end_i-1] 
    count=0
    for item_id in indices_sort_top:
        if item_id < index_end_i:
            hr_topK+=1.0
            ndcg_topK+=1.0/math.log(count+2) 
        count+=1
        if count==top_k:
            break 
    hr_t=hr_topK/max_hr
    ndcg_t=ndcg_topK/max_ndcg  
    # hr_t,ndcg_t,index_end_i,indices_sort_top
    # pdb.set_trace() 
    return hr_t,ndcg_t




  
def metrics(model, test_val_loader, top_k, num_negative_test_val, batch_size):
    HR, NDCG = [], [] 
    test_loss_sum=[]
    # pdb.set_trace()  
 
    test_start_time = time.time()
    for user, item_i, item_j in test_val_loader:  
        # start_time = time.time()
        # pdb.set_trace()
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j #index to split

        prediction_i, prediction_j,loss_test,loss2_test = model(user, item_i, torch.cuda.LongTensor([0])) 
        test_loss_sum.append(loss2_test.item())  
        # pdb.set_trace()   
        elapsed_time = time.time() - test_start_time
        print('time:'+str(round(elapsed_time,2)))
        courrent_index=0
        courrent_user_index=0
        for len_i,len_j in item_j:
            index_end_i=(len_i-len_j).item()  
            #pre_error=(prediction_i[0][courrent_index:(courrent_index+index_end_i)]- prediction_i[0][(courrent_index+index_end_i):(courrent_index+index_end_j)])#.sum() 
            #loss_test=nn.MSELoss((pre_error).sum())#-(prediction_i[0][courrent_index:(courrent_index+index_end_i)]- prediction_i[0][(courrent_index+index_end_i):(courrent_index+index_end_j)]).sigmoid().log()#.sum()   
            _, indices = torch.topk(prediction_i[0][courrent_index:(courrent_index+len_i)], top_k)   
            hr_t,ndcg_t=hr_ndcg(indices.tolist(),index_end_i,top_k)  
            # print(hr_t,ndcg_t,indices,index_end_i)
            # pdb.set_trace()
            HR.append(hr_t)
            NDCG.append(ndcg_t) 

            courrent_index+=len_i 
            courrent_user_index+=1 

 
    test_loss=round(np.mean(test_loss_sum[:-1]),4)  
 
    return test_loss,round(np.mean(HR),4) , round(np.mean(NDCG),4) 

 