import numpy as np

import scipy.sparse as sp


class Interactions(object):


    def __init__(self, user_item_sequence, user_time_sequence, num_users, num_items, sorted_time):
        user_ids, item_ids, time_ids = [], [], []
        for uid in user_item_sequence:
            for iid, tid in zip(user_item_sequence[uid],user_time_sequence[uid]):
                user_ids.append(uid)
                item_ids.append(iid)
                time_ids.append(tid)

        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)
        time_ids = np.asarray(time_ids)

        self.num_users = num_users
        self.num_items = num_items

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.time_ids = time_ids

        self.sorted_time = sorted_time

        self.sequences = None
        self.test_sequences = None

    def __len__(self):

        return len(self.user_ids)

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))


    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()


    def to_sequence(self, subgraphs_mapping_i, subgraphs_mapping_u, subgraphs_sequence_i, subgraphs_sequence_u, sequence_length=5, target_length=1):


        max_sequence_length = sequence_length + target_length

        user_ids = self.user_ids#[sort_indices]
        item_ids = self.item_ids#[sort_indices]
        time_ids = self.time_ids#[sort_indices]

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)


        num_subsequences = sum([c - 1 for c in counts])

        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64) # training length=sequence_length
        sequences_original = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_user_tracking = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64) #
        sequences_targets = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequences_targets_time = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequences_targets_original = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences, dtype=np.int64)
        sequence_users_dy = np.empty(num_subsequences, dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_sequences_original = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_sequences_user_tracking = np.zeros((self.num_users, sequence_length),
                             dtype=np.int64) #
        test_users = np.empty(self.num_users, dtype=np.int64)
        test_users_dy = np.empty(self.num_users, dtype=np.int64)

        _uid = None
        last_graph_time = max(subgraphs_mapping_u.keys())
        for i, (uid,
                item_seq, time_seq) in enumerate(_generate_sequences(user_ids,
                                                           item_ids,
                                                           time_ids,
                                                           indices,
                                                           max_sequence_length)):

            if uid != _uid:  #for test
                test_sequences[uid][:] = [subgraphs_mapping_i[tt][ii] for ii,tt in zip(item_seq[-sequence_length:],time_seq[-sequence_length:])]
                test_sequences_original[uid][:] = item_seq[-sequence_length:]
                test_sequences_user_tracking[uid][:] = [subgraphs_mapping_u[tt][uid] for tt in time_seq[-sequence_length:]]
                test_users[uid] = uid
                test_users_dy[uid] = subgraphs_sequence_u[uid][last_graph_time]
                _uid = uid
                

            sequences_targets[i][:] = [subgraphs_sequence_i[ii][tt] for ii,tt in zip(item_seq[-target_length:],time_seq[-target_length:])]
            sequences_targets_original[i][:] = item_seq[-target_length:]
            sequences_targets_time[i][:] = time_seq[-target_length:]


            time_temp = time_seq[-target_length:][0]
            temp = []
            tempU = []
            for ii,tt in zip(item_seq[:sequence_length],time_seq[:sequence_length]):
                if tt == time_temp:
                    temp.append(subgraphs_sequence_i[ii][tt])
                    tempU.append(subgraphs_sequence_u[uid][tt])
                else:
                    temp.append(subgraphs_mapping_i[tt][ii])
                    tempU.append(subgraphs_mapping_u[tt][uid])
            sequences[i][:] = temp
            sequences_original[i][:] = item_seq[:sequence_length]
            sequences_user_tracking[i][:] = tempU
            sequence_users[i] = uid
            sequence_users_dy[i] = subgraphs_sequence_u[uid][time_temp]



        self.sequences = SequenceInteractions(sequence_users, sequence_users_dy, sequences, sequences_original, sequences_user_tracking, sequences_targets_time, sequences_targets, sequences_targets_original)
        self.test_sequences = SequenceInteractions(test_users, test_users_dy, test_sequences, test_sequences_original, test_sequences_user_tracking)

        print(sequences.shape)

class SequenceInteractions(object):


    def __init__(self,
                 user_ids,
                 user_ids_dy,
                 sequences,
                 sequences_orig,
                 sequences_user_tracking,
                 sequences_time=None,
                 targets=None,
                 targets_orig=None):
        self.user_ids = user_ids
        self.user_ids_dy = user_ids_dy
        self.sequences = sequences
        self.sequences_orig = sequences_orig
        self.sequences_user_tracking = sequences_user_tracking
        self.sequences_time = sequences_time
        self.targets = targets
        self.targets_orig = targets_orig

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]


def _sliding_window(tensor, tensorT, window_size, step_size=1):
    for i in range(len(tensor), 1, -step_size):
        if i - window_size >= 0:
            yield tensor[i - window_size:i], tensorT[i - window_size:i]
        else:
            num_paddings = window_size - i
            yield np.pad(tensor[:i], (num_paddings, 0), 'constant'), np.pad(tensorT[:i], (num_paddings, 0), 'constant')


def _generate_sequences(user_ids, item_ids, time_ids,
                        indices,
                        max_sequence_length):
    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq, seqT in _sliding_window(item_ids[start_idx:stop_idx], time_ids[start_idx:stop_idx],
                                   max_sequence_length):
            yield (user_ids[i], seq, seqT)
