import numpy as np
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


import tensorflow as tf
import sys
from tqdm import tqdm
import math
import random
from collections import defaultdict
import os
import argparse
import logging
from time import time
import datetime

from data import Dataset
from interactions import Interactions
import hypergraph_utils as hgut
from model import *
from sampler import *



parser = argparse.ArgumentParser()

# data arguments
parser.add_argument('--seq_len', type=int, default=50, help='max sequence length (default: 20)')
parser.add_argument('--data', type=str, default='newAmazon', help='data set name (default: newAmazon)')
parser.add_argument('--T', type=int, default=1)

# use BPR to pretrain
parser.add_argument('--n_iter_bpr', type=int, default=20)
parser.add_argument('--bpr_batch_size', type=int, default=512)
parser.add_argument('--worker_bpr', type=int, default=2, help='number of sampling workers (default: 2)')

parser.add_argument('--n_iter', type=int, default=200)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--l2_reg', type=float, default=0.0)
parser.add_argument('--clip', type=float, default=1., help='gradient clip (default: 1.)')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout (default: 0.5)')
parser.add_argument('--graph_dropout', type=float, default=0.6, help='dropout (default: 0.6)')
parser.add_argument('--eval_interval', type=int, default=10, help='eval/test interval')
parser.add_argument('--worker', type=int, default=5, help='number of sampling workers (default: 10)')
parser.add_argument('--eval_batch_size', type=int, default=1024, help='eval/test batch size (default: 128)')

parser.add_argument('--emsize', type=int, default=100, help='dimension of item embedding (default: 100)')
parser.add_argument('--n_hyper', type=int, default=2, help='number of layers for Hypergraph (default: 2)')
parser.add_argument('--num_blocks', type=int, default=2, help='num_blocks')
parser.add_argument('--num_heads', type=int, default=1, help='num_heads')
parser.add_argument('--pos_fixed', type=int, default=0, help='trainable positional embedding usually has better performance')

args = parser.parse_args()
tf.set_random_seed(args.seed)

args.neg_size = 1

train_set, val_set, train_val_set, test_set, data_time, neg_test, num_items, num_users = Dataset.data_partition_neg(args)



print(data_time[-1])
print(data_time[-2])

subgraphs_mapping_i, subgraphs_G, subgraphs_mapping_u = hgut.subgraph_con(train_set, data_time[0], data_time[-2])

subgraphs_mapping_i, reversed_subgraphs_mapping_i, sorted_time, subgraphs_sequence_i, reversed_subgraphs_mapping_last_i = hgut.subgraph_key_building(subgraphs_mapping_i, num_items)

subgraphs_mapping_u, reversed_subgraphs_mapping_u, sorted_time_u, subgraphs_sequence_u, reversed_subgraphs_mapping_last_u = hgut.subgraph_key_building(subgraphs_mapping_u, num_users)

assert sorted_time == sorted_time_u

train_data = Interactions(train_set, data_time[0], num_users, num_items, sorted_time)
train_data.to_sequence(subgraphs_mapping_i,subgraphs_mapping_u,subgraphs_sequence_i, subgraphs_sequence_u, args.seq_len, args.T)


#for bpr pretrain
bpr_tuples = list(zip(train_data.user_ids, train_data.item_ids))
train_dic = {}
for i in train_set:
	train_dic[i] = set(train_set[i])

neg_test_dy = np.zeros((num_users, 101),dtype=np.int64)
for i in range(1,num_users):
	for j in range(101):
		neg_test_dy[i][j] = subgraphs_sequence_i[neg_test[i][j]][-1]


print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(args)

print ('#Item: ', num_items - 1)
print ('#User: ', num_users - 1)

model = NeuralSeqRecommender(args, num_items, num_users, subgraphs_G, reversed_subgraphs_mapping_i, reversed_subgraphs_mapping_u, reversed_subgraphs_mapping_last_i, sorted_time, args.n_hyper)

lr = args.lr

k_list = [1, 5, 10, 15, 20, 50, 100]

def evaluation(sess):
	num_users = train_data.num_users
	num_items = train_data.num_items
	batch_size = 2048
	num_batches = int((num_users-1) / batch_size) + 1
	user_indexes = np.arange(1, num_users)
	item_indexes = np.arange(num_items)
	item_indexes_1 = np.arange(1,num_items)
	pred_list = None
	test_sequences = train_data.test_sequences.sequences
	test_sequences_orig = train_data.test_sequences.sequences_orig
	rankings_list = {}

	for batchID in range(num_batches):
		start = batchID * batch_size
		end = start + batch_size

		if batchID == num_batches - 1:
			if start < num_users - 1:
				end = num_users - 1
			else:
				break

		batch_user_index = user_indexes[start:end]
		batch_test_sequences = test_sequences[batch_user_index]
		batch_test_sequences_orig = test_sequences_orig[batch_user_index]
		batch_test_neg = neg_test[batch_user_index]
		batch_test_neg_dy = neg_test_dy[batch_user_index]

		feed_dict = {model.inp: np.array(batch_test_sequences), model.inp_ori: np.array(batch_test_sequences_orig), model.dropout: 0., model.dropout_graph: 0.}
		feed_dict[model.test_item_batch] = np.array(batch_test_neg)
		feed_dict[model.u_list] = np.array(batch_user_index)
		feed_dict[model.u_seq] = np.array(train_data.test_sequences.sequences_user_tracking[batch_user_index])


		feed_dict[model.test_item_batch_dy] = np.array(batch_test_neg_dy)
		predictions_batch = sess.run([model.test_logits_batch], feed_dict=feed_dict)


		predictions_batch = predictions_batch[0]

		if batchID == 0:
			pred_list = predictions_batch
		else:
			pred_list = np.append(pred_list, predictions_batch, axis=0)

	HT = [0.0000 for k in k_list]
	NDCG = [0.0000 for k in k_list]
	MRR = 0.0000
	valid_user = 0

	for user_id in test_set:
		valid_user += 1.0
		pre_temp = -pred_list[user_id-1] 
		rank = pre_temp.argsort().argsort()[0]

		MRR += 1.0/(rank+1)
		for k in range(len(k_list)):  
			if rank < k_list[k]:
				NDCG[k] += 1.0 / np.log2(rank + 2)
				HT[k] += 1

	return [NDCG[k]*1.0 / valid_user for k in range(len(k_list))], [HT[k]*1.0 / valid_user for k in range(len(k_list))], MRR*1.0 / valid_user


def main():
	global lr
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()
	sess.run(init)


	sequences_np = train_data.sequences.sequences
	sequences_original_np = train_data.sequences.sequences_orig
	targets_np = train_data.sequences.targets_orig
	targets_np_dy = train_data.sequences.targets
	users_np = train_data.sequences.user_ids
	train_matrix = train_data.tocsr()

	n_train = sequences_np.shape[0]
	print("Total training records:{}".format(n_train))

	record_indexes = np.arange(n_train)
	

	#use BPR to pretrain
	try:
		bpr_train_sampler = WarpSampler_bpr(
					train_tuples = bpr_tuples, 
					train_dic = train_dic,  
					num_user = num_users,
					neg_sample_rate = args.neg_size,
					num_item = num_items,  
					batch_size = args.bpr_batch_size, 
					n_workers = args.worker_bpr)

		
		bpr_num_batch = int(len(bpr_tuples) / args.bpr_batch_size)
		for epoch in range(0, args.n_iter_bpr):

			print ('Pre Training epoch:%d' % (epoch))
			epoch_loss_bpr = 0

			for step in tqdm(range(bpr_num_batch), total=bpr_num_batch, ncols=70, leave=False, unit='b'):
				batch_bpr = bpr_train_sampler.next_batch()
				batch_loss_bpr,  _ = sess.run(fetches=[model.loss_bpr, model.optimizer_bpr],
													feed_dict={model.triple_bpr: batch_bpr, model.lr: 0.05})

				epoch_loss_bpr += batch_loss_bpr

			print ('Training epoch:%d, training loss bpr: %f' % (epoch, epoch_loss_bpr))

			t2 = time()
		bpr_train_sampler.close()


	except Exception as e:
		print(str(e))
		bpr_train_sampler.close()
		exit(1)

	

	try:
		batch_size = args.batch_size
		num_batches = int(n_train / batch_size) + 1

		train_sampler = Sampler(
					data=train_data, 
					train_dic = train_dic, 
					subgraphs_sequence_i = subgraphs_sequence_i, 
					n_items=num_items, 
					n_users=num_users,
					batch_size=args.batch_size, 
					max_len=args.seq_len,
					neg_size=args.neg_size,
					n_workers=args.worker,
					neg_method='rand')


		for epoch_num in range(args.n_iter):
			print ('Training epoch:%d' % (epoch_num))
			t1 = time()


			epoch_loss = 0.0
			batchID = 0

			for step in tqdm(range(num_batches), total=num_batches, ncols=70, leave=False, unit='b'):


				batch_sample = train_sampler.next_batch()
				batch_record_index = batch_sample[0]
				batch_users = users_np[batch_record_index]
				batch_sequences = sequences_np[batch_record_index]
				batch_sequences_original = sequences_original_np[batch_record_index]

				batch_targets = targets_np[batch_record_index]
				batch_targets_dy = targets_np_dy[batch_record_index]
				batch_u_seq = train_data.sequences.sequences_user_tracking[batch_record_index]
				batch_neg = batch_sample[1] 
				batch_neg_dy = batch_sample[2]

				inp = np.array(batch_sequences)
				feed_dict = {model.inp: inp, model.inp_ori:batch_sequences_original, model.lr: lr, model.dropout: args.dropout}
				feed_dict[model.u_list] = np.array(batch_users)
				feed_dict[model.pos] = np.array(batch_targets)
				feed_dict[model.pos_dy] = np.array(batch_targets_dy)
				feed_dict[model.neg] = np.array(batch_neg)
				feed_dict[model.neg_dy] = np.array(batch_neg_dy)
				feed_dict[model.dropout_graph] = args.graph_dropout
				feed_dict[model.u_seq] = np.array(batch_u_seq)


				_, train_loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)


				epoch_loss += train_loss
				batchID += 1

			t2 = time()
			output_str = "Epoch %d [%.1f s]  loss=%.4f" % (epoch_num, t2 - t1, epoch_loss)
			print(output_str)

			if (epoch_num) % args.eval_interval == 0 or epoch_num == args.n_iter-1:

				NDCG, HR, MRR = evaluation(sess)
				print ('Evaluating Model')

				print('NDCG', ', '.join(str(e) for e in NDCG))
				print('Hit', ', '.join(str(e) for e in HR))
				print('MRR', str(MRR))
				print("Evaluation time:{}".format(time() - t2))



	except Exception as e:
		train_sampler.close()
		print(str(e))
		exit(1)


if __name__ == '__main__':
    main()