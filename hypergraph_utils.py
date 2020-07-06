import numpy as np
import scipy.sparse as sp



def subgraph_con(interation, time, timestamp):
    subgraphs = {}
    subgraphs_G = {}

    subgraphs_mapping_u = {}
    subgraphs_mapping_i = {}
    for t in timestamp:
        subgraphs[t] = {'u':[], 'i':[]}
        subgraphs_mapping_u[t] = {}
        subgraphs_mapping_i[t] = {}

    for user in interation:
        iteml = interation[user]
        timel = time[user]
        for i, t in zip(iteml, timel):
            if not i in subgraphs_mapping_i[t]:
                subgraphs_mapping_i[t][i] = len(subgraphs_mapping_i[t])
            if not user in subgraphs_mapping_u[t]:
                subgraphs_mapping_u[t][user] = len(subgraphs_mapping_u[t])

            subgraphs[t]['u'].append(subgraphs_mapping_u[t][user])
            subgraphs[t]['i'].append(subgraphs_mapping_i[t][i])

    for t in timestamp: 
        col = subgraphs[t]['u']
        row = subgraphs[t]['i']
        data = np.ones(len(col))

        sg = sp.coo_matrix((data, (row, col)), shape=(len(subgraphs_mapping_i[t]), len(subgraphs_mapping_u[t])))
        print('Done constructing subgraph', str(t))
        print(len(subgraphs_mapping_i[t]), len(subgraphs_mapping_u[t]), len(data))

        subgraphs_G[t] = {}
        subgraphs_G[t]['G'], subgraphs_G[t]['E'] = generate_G_from_H(sg)


    return subgraphs_mapping_i, subgraphs_G, subgraphs_mapping_u

def subgraph_key_building(subgraphs_mapping_i, num_items):
    reversed_subgraphs_mapping_i = {}
    for t in subgraphs_mapping_i:
        reversed_subgraphs_mapping_i[t] = [0]*len(subgraphs_mapping_i[t])
        for i in subgraphs_mapping_i[t]:
            reversed_subgraphs_mapping_i[t][subgraphs_mapping_i[t][i]] = i
    
    sorted_time = sorted(list(subgraphs_mapping_i.keys()))

    if not 0 in subgraphs_mapping_i:
        subgraphs_mapping_i[0] = {}
        for i in range(num_items):
            subgraphs_mapping_i[0][i] = i

    cumuindex = num_items
    cumuindex_record = {}
    for t in sorted_time:
        cumuindex_record[t] = cumuindex
        for i in subgraphs_mapping_i[t]:
            subgraphs_mapping_i[t][i] += cumuindex
        cumuindex += len(subgraphs_mapping_i[t])

    ##### get the latest dynamic mapping the subgraph
    subgraphs_sequence_i = {}
    for i in range(1,num_items):
        subgraphs_sequence_i[i] = np.array([i] * (3 + len(sorted_time)))
        
    for t in sorted_time:
        for i in subgraphs_mapping_i[t]:
             subgraphs_sequence_i[i][t+1:] = subgraphs_mapping_i[t][i]

    reversed_subgraphs_mapping_last_i = {}
    for t in subgraphs_mapping_i:
        if t==0:
            continue
        reversed_subgraphs_mapping_last_i[t] = [0]*len(subgraphs_mapping_i[t])
        for i in subgraphs_mapping_i[t]:
            reversed_subgraphs_mapping_last_i[t][subgraphs_mapping_i[t][i]-cumuindex_record[t]] = subgraphs_sequence_i[i][t]
        
    return subgraphs_mapping_i, reversed_subgraphs_mapping_i, sorted_time, subgraphs_sequence_i, reversed_subgraphs_mapping_last_i



def generate_G_from_H(H):

    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.array(H.sum(1)) 
    # the degree of the hyperedge
    DE = np.array(H.sum(0))

    invDE2 = sp.diags(np.power(DE, -0.5).flatten())
    DV2 =  sp.diags(np.power(DV, -0.5).flatten())
    W = sp.diags(W)
    HT = H.T


    invDE_HT_DV2 = invDE2 * HT * DV2
    G = DV2 * H * W * invDE2 * invDE_HT_DV2
    return G, invDE_HT_DV2