import random
import subprocess
import numpy as np
import torch
from scipy import sparse
import time
from collections import defaultdict
import csv
import networkx as nx
import pandas as pd
import datetime
import math
from scipy.stats import pearsonr
from math import exp
import csv
import pickle
from igraph import Graph as IGraph
# import warnings
# warnings.filterwarnings("ignore")

all_hash = pd.read_csv('./data/all_hash_10_train.csv', names=['index', 'address', 'label'])
all_index = pd.read_csv('./data/all_index_adj_together_with_start_risk_with_count_with_all.csv')
# all_hash=pd.read_csv('./data/all_hash_1191809_with_train_642.csv',names = ['index','address','label'])
# all_index=pd.read_csv('./data/all_index_adj_4136463_with_count_with_all.csv')

nodes = all_hash["index"]
edges = all_index["Unnamed: 0"]
print("ethereum network has %d nodes and %d edges" % (len(nodes), len(edges)))

From=all_index["index_from"]
To=all_index["index_to"]
Len=len(From)
edge0=[]
for i in range(Len):
    edge0.append((From[i],To[i]))
edge1=[]
for i in edge0:
    if i not in edge1:
        edge1.append(i)
g=IGraph.TupleList(edge1,directed=False,vertex_name_attr='name')
close=dict()
ccvs=[]
sum_close=0
for p in zip(g.vs,g.closeness()):
    ccvs.append({"index":p[0]["name"],"cc":p[1]})
    close[p[0]["name"]]=p[1]
    sum_close+=p[1]

N=len(nodes)
q=0.85
a=(1-q)*1.0/N
#a=(1-q)
reliable = dict()
score = dict()
max_to_count = max(all_index["to_count"])
max_from_count = max(all_index["from_count"])
max_all_count = max(all_index["all_count"])
max_value = max(all_index["value"])
for edge in edges:
# 第二种方案（度）
    score[edge] = ((2*math.log(all_index["from_count"][edge],10)-math.log(max_from_count,10))/math.log(max_from_count,10) + (2*math.log(all_index["to_count"][edge],10)-math.log(max_to_count,10))/math.log(max_to_count,10))/2
for node_t in nodes:
    reliable[node_t] = 1
    if all_hash["label"][node_t] == 0:
        reliable[node_t] = 1
    if all_hash["label"][node_t] == 1:
        reliable[node_t] = 0
iter = 0

##### ITERATIONS START ######
outcount_for_node = dict()
reliable_value_all = dict()
for node_t in nodes:
    outcount_for_node[node_t] = 0
    reliable_value_all[node_t] = 0
for edge in edges:
    outcount_for_node[all_index["index_from"][edge]] += 1

dr = 0
while iter < 400:
    print('-----------------')
    print("Epoch number %d with dr = %f" % (iter, dr))
    if np.isnan(dr):
        print('over')
        break
    dr=0
    ############################################################
    print("Updating reliable of nodes")
    for node in nodes:
        b=0
        if all_hash["label"][node] == 0 or all_hash["label"][node] == 1:
            continue
        mask= all_index["index_to"]==node
        pos=np.flatnonzero(mask)
        if len(pos) == 0:
            reliable_for_node=a
        else:
            for i in pos:
                p=all_index["index_from"][i]
    #基础公式
                b+=reliable[p]*1.0/outcount_for_node[p]
            reliable_for_node = a + b * q
    #改进公式1
                #b += reliable[p] * 1.0 / outcount_for_node[p]
            #reliable_for_node=(1-q)*close[node]/sum_close+b*q
    #改进公式2
                #b += reliable[p] * 1.0 / outcount_for_node[p]*score[i]
            #reliable_for_node=a+b*q
    # 改进公式3
                #b += reliable[p] * 1.0 / outcount_for_node[p]*score[i]
        # reliable_for_node=(1-q)*close[node]/sum_close+b*q
        x = reliable_for_node
        # if x < 0.00:
        #     x = 0.0
        # if x > 1.0:
        #     x = 1.0

        dr += abs(reliable[node] - x)
        reliable[node] = x

    iter += 1
    if dr < 0.01:
        print("The propagation equation reaches convergence after " + str(iter) + " more iterations!")
        break


# fw = open(
#     "./results/small-dataset-20210731-reliable.csv" ,"w")
# fw.write("outcount,reliable\n")
fw = open(
    "./results/small-dataset-20210804-reliable-basic-new.csv" ,"w")
fw.write("outcount,reliable\n")

for node in nodes:
    fw.write("%s,%s\n" % (str(outcount_for_node[node]), str(reliable[node])))
fw.close()
