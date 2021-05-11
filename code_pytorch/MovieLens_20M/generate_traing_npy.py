import pdb
from collections import defaultdict
import numpy as np
import random

training_path='./train.txt' 
testing_path='./test.txt'
val_path='./val.txt'

def txt2npy(file_path):
    num_u_train=0
    links_file = open(file_path,'r') 
    user_set=  defaultdict(set) 
    item_set=  defaultdict(set) 
    num_data = 0
    for one_user in links_file:
        one_user=one_user.strip() 
        split_str = one_user.split(" ") 
        u_id=int(split_str[0])
        for i in range(len(split_str)-1):
            v_id=int(split_str[i+1])
            user_set[u_id].add(v_id)
            item_set[v_id].add(u_id)
            num_data+=1
    return user_set, item_set, num_data
train_set_user,train_set_item,num_u_train = txt2npy(training_path)
np.save('./training_set.npy',[train_set_user,train_set_item,num_u_train])
 
exit()

test_set_user,test_set_item,num_u_test = txt2npy(testing_path)
np.save('./testing_set.npy',[test_set_user,test_set_item,num_u_test]) 

val_set_user,val_set_item,num_u_val = txt2npy(val_path)
np.save('./val_set.npy',[val_set_user,val_set_item,num_u_val]) 

 