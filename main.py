import numpy as np
import torch
import pandas as pd
from subtime.graph.NxGraph import *
from data.data_gen_easy import TimeData, TimeData_ins
from subtime.search.TD import td
from evaluation.evaluation import *
import networkx as nx
import random

n_vars = 5
edge_prob = 0.4
adj = generate_dag_with_forward_backward_fork(n_vars, edge_prob)
data_gen = TimeData_ins(adj, k=1, n=200)
mag, sdag, bcer = td(
    data=data_gen,
    node_names=data_gen.observation.columns,
    indep_test='fisherz',
    proxy_test='proxy_test',
    alpha=0.05,
    subsampling_factor=data_gen.k
)
# Evaluation
prec = precision(sdag.edges, data_gen.sdag.edges)
rec = recall(sdag.edges, data_gen.sdag.edges)
f1 = compute_f1(prec, rec)
nodes = sorted(sdag.nodes)
adj_pre = nx.to_numpy_array(sdag, nodelist=nodes, dtype=int)
shd = backRE(adj, adj_pre)
print(shd)
print(f1)