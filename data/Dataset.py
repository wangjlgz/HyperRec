import pickle
import math

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from collections import defaultdict
import datetime
import operator

data_path = 'data/'

def data_partition_neg(args):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    User_time = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_train_valid = {}
    user_test = {}
    neg_test = {}

    time_set_train = set()
    time_set_test = set()

    user_train_time = {}
    user_valid_time = {}
    user_train_valid_time = {}
    user_test_time = {}
    # assume user/item index starting from 1
    path_to_data = data_path + args.data + '/' + args.data + '_all.txt'

    # for Amazon    
    t_map = {1997:[1], 1998:[1], 1999:[1], 2000:[1], 2001:[1], 2002:[1], 2003:[1], 2004:[1], 2005:[1], 2006:[1], 2007:[1], 2008:[1], 2009:[2,3], 2010:[4,5], \
    2011:[6,7], 2012:[8,9], 2013:[10,11], 2014:[12,13], 2015:[14,15], 2016:[16,17], 2017:[18,19], 2018:[20]}

     
    m_map = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1}

    f = open(path_to_data, 'r')
    for line in f:
        u, i, t, d = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        year = int(datetime.datetime.fromtimestamp(int(t)).strftime("%Y")) # Day of the year as a decimal number [001,366]
        month = int(datetime.datetime.fromtimestamp(int(t)).strftime("%m"))

        
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)   

        temp_map = t_map[year]
        if len(temp_map) == 1:
            User_time[u].append(temp_map[0])
        else:
            User_time[u].append(temp_map[m_map[month]]) 


    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []

            user_train_time[user] = User_time[user]
            user_valid_time[user] = []
            user_test_time[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
            
            neg_test[user] = [User[user][-1]]

            user_train_time[user] = User_time[user][:-2]
            time_set_train.update(user_train_time[user])
            user_valid_time[user] = []
            user_valid_time[user].append(User_time[user][-2])
            user_test_time[user] = []
            user_test_time[user].append(User_time[user][-1])
            time_set_test.update(user_test_time[user])


        user_train_valid[user] = user_train[user] + user_valid[user]
        user_train_valid_time[user] = user_train_time[user] + user_valid_time[user]


    skip = 0
    neg_f = data_path + args.data + '/' + args.data + '_test_neg.txt'
    with open(neg_f, 'r') as file:
        for line in file:
            skip += 1
            if skip==1:
                continue
            user_id, item_id = line.rstrip().split('\t')
            u = int(user_id)
            i = int(item_id)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)

            neg_test[u].append(i)

    sequences = np.zeros((usernum + 1, 101),dtype=np.int64)
    for user in User:
        sequences[user][:] = neg_test[user]

    neg_test = sequences.copy()

    return [user_train, user_valid, user_train_valid, user_test, (user_train_time, user_valid_time, \
        user_train_valid_time, user_test_time, time_set_train, time_set_test), neg_test, itemnum+1, usernum+1]
