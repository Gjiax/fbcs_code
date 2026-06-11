import numpy as np
import pandas as pd
import networkx as nx

from data.structual_equations import rand_steq,distribution,function
from subtime.graph.NxGraph import adj_to_dag,adj_to_dag_edge_types,get_parents,sdag_to_ftdag,ftdag_to_mag,sdag_to_ftdag_edge_types

# The blow code generates random structual equations inside the TimeData class

class TimeData_ins:
    def __init__(self,adj,k,n,**kwargs):
        # d - number of vertices
        # n - sample size
        # k - subsampling factor
        self.adj = adj
        self.d = self.adj.shape[0]
        self.k = k
        self.n = n
        
        self.data = dict()
        
        self.sdag = adj_to_dag_edge_types(self.adj)
        for u, v, data in self.sdag.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            #print(f"{u} -> {v}: {edge_type}")
        #self.ftdag = sdag_to_ftdag(self.sdag,self.k)
        self.ftdag_ins = sdag_to_ftdag_edge_types(self.sdag, 10)
        self.ssteq, self.ftsteq = rand_steq(self.sdag, self.ftdag_ins)
        self.generate(**kwargs)
        self.data = pd.DataFrame(self.data)
        self.data = self.data.reindex(columns=list(self.ftdag_ins.nodes))

        self.mag_ins = ftdag_to_mag(self.ftdag_ins, self.d, self.k)
        self.mag_ske = self.mag_ins[0] + self.mag_ins[1]
        #self.ssteq, self.ftsteq = rand_steq(self.sdag,self.ftdag)
        

        
        self.observation = dict()
        for node in self.data.keys():
            var,time = node.split('_'); time = int(time)
            if time==0 or time==self.k+1:
                self.observation[node] = self.data[node]
        self.observation = pd.DataFrame(self.observation)
        
        
    def generate(self,**kwargs):
        # V_i = sum_j f_j(PA_j) + exo_i
        for node in nx.topological_sort(self.ftdag_ins):
            var,time = node.split('_'); time = int(time)
            exodist = self.ssteq[var]
            parents = get_parents(self.ftdag_ins,node)
            if len(parents)==0:
                assert time==0, 'Node {} at subtime {} has no parents.'.format(var,time)
                self.data[node] = distribution(exodist,**kwargs)(size=self.n)
            else:
                exo = distribution(exodist,**kwargs)(size=self.n)
                for parent in parents:
                    key = '{}->{}'.format(parent,node) 
                    func = self.ftsteq[key]
                    exo += function(func,**kwargs)(self.data[parent])
                self.data[node] = exo


class TimeData:
    def __init__(self, adj, k, n, **kwargs):
        # d - number of vertices
        # n - sample size
        # k - subsampling factor
        self.adj = adj
        self.d = self.adj.shape[0]
        self.k = k
        self.n = n

        self.data = dict()

        self.sdag = adj_to_dag(self.adj)
        #self.ftdag = sdag_to_ftdag(self.sdag,self.k)
        self.ftdag_ins = sdag_to_ftdag(self.sdag, 10)
        self.generate(**kwargs)
        #self.mag = ftdag_to_mag(self.ftdag,self.d,self.k)
        self.mag = ftdag_to_mag(self.ftdag_ins, self.d, self.k)
        # self.mag_ske = self.mag[0] + self.mag[1] # skeleton_of_mag = directed_edges + bidirected edges
        self.mag_ske = self.mag[0] + self.mag[1]
        #self.ssteq, self.ftsteq = rand_steq(self.sdag,self.ftdag)
        self.ssteq, self.ftsteq = rand_steq(self.sdag, self.ftdag_ins)
        self.generate(**kwargs)

        self.data = pd.DataFrame(self.data)
        self.data = self.data.reindex(
            columns=list(self.ftdag_ins.nodes))

        self.observation = dict()
        for node in self.data.keys():
            var, time = node.split('_');
            time = int(time)
            #if subtime == 0 or subtime == 10:
            if time == 0 or time == self.k + 1:
                self.observation[node] = self.data[node]
        self.observation = pd.DataFrame(self.observation)

    def generate(self, **kwargs):
        seed=(2025)
        # V_i = sum_j f_j(PA_j) + exo_i
        for node in nx.topological_sort(self.ftdag_ins):
            var, time = node.split('_');
            time = int(time)
            exodist = self.ssteq[var]
            parents = get_parents(self.ftdag_ins, node)
            if len(parents) == 0:
                assert time == 0, 'Node {} at subtime {} has no parents.'.format(var, time)
                self.data[node] = distribution(exodist, **kwargs)(
                    size=self.n)  # distribution(exodist,**kwargs)返回要调用的函数，size后面生成随机数
            else:
                exo = distribution(exodist, **kwargs)(size=self.n)
                for parent in parents:
                    key = '{}->{}'.format(parent, node)
                    func = self.ftsteq[key]
                    exo += function(func, **kwargs)(self.data[parent])
                self.data[node] = exo

if __name__ == "__main__":
    adj = np.array([[0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    k=2
    n=100
    time_data = TimeData_ins(adj, k, n)

    time_series_data = time_data.data
    observation_data = time_data.observation
