from __future__ import annotations

import subtime
import warnings
from itertools import combinations, permutations
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
from numpy import ndarray
import pandas as pd
from subtime.graph.GraphClass import CausalGraph
from subtime.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from subtime.utils.cit import CIT
from subtime.search import SkeletonDiscovery, SkeletonDiscovery_Time
import re
import random

def pc(
    data: ndarray, 
    alpha = 0.05,
    indep_test= str,
    stable: bool = True, 
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False, 
    show_progress: bool = True,
    node_names: List[str] | None = None,
    **kwargs
):
    """
    Implement PC_skeleton algorithm
    Args: data, a numpy array with shape (n,d), n is sample size, d is vertex number
          alpha, significance level
          indep_test, string, choose from ['info_test','fisherz','kci','d_separation',...]
    """
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    ind_test = CIT(data, indep_test, **kwargs)
    cg = SkeletonDiscovery.skeleton_discovery(data, alpha, ind_test, stable,
                                              background_knowledge=background_knowledge, verbose=verbose,
                                              show_progress=show_progress, node_names=node_names,**kwargs)

    return cg




def pc_time(
        data: ndarray,
        alpha=0.05,
        indep_test=str,
        stable: bool = True,
        background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = True,
        node_names: List[str] | None = None,
        **kwargs
):
    """
    Implement PC_skeleton_time algorithm
    Args: data, a numpy array with shape (n,d), n is sample size, d is vertex number
          alpha, desired FDR level
          indep_test, string, choose from ['info_test','fisherz','kci','d_separation',...]
    Return:
         mag = [list_of_direct_edges, list_of_bidirected_edges]
         list_of_direct_edges = [(X1_0,X3_3), (X4_0,X5_3), ...] each tuple is always directed
    """
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    msg = 'The PC_time algorithm should be used very carefully. \n\n' + \
          'We use the following rules to decide sepset(a,b): \n' + \
          '1. subtime(a)=subtime(b)=0: delete the edge without testing (for simulation, there is always no edge; for real-world, we dont have data at -k-1 and any detected edge can be unreliable) \n' + \
          '2. subtime(a)=0,subtime(b)=k+1 (and vice versa): search sepset in subtime=0 (actually unnecessary for simulation data) \n' + \
          '3. subtime(a)=subtime(b)=k+1: search sepset in subtime=0 \n'
    # warnings.warn(msg)

    ind_test = CIT(data, indep_test, **kwargs)
    cg_upate = SkeletonDiscovery_Time.skeleton_discovery_time(data, alpha, ind_test, stable,
                                                      background_knowledge=background_knowledge,
                                                       show_progress=show_progress,node_names=node_names)
    #第一阶段：条件及为空 得到相关矩阵
    cg, p_matrix=SkeletonDiscovery_Time.skeleton_discovery_selected_pairs_with_pmatrix(data, alpha, ind_test, stable,
                                                                                      node_names=node_names)
    # print(cg.G.get_graph_edges())
    mag = [list(), list()]  # directed edges and bidirected edges
    d = len(cg.G.nodes)
    for i in range(d):
        for j in range(d):
            if cg.G.graph[i, j] != 0 and cg.G.graph[j, i] != 0:
                name_i, time_i = cg.G.nodes[i].name, int(cg.G.nodes[i].name.split('_')[1])
                name_j, time_j = cg.G.nodes[j].name, int(cg.G.nodes[j].name.split('_')[1])
                if time_i < time_j:
                    mag[0].append((name_i, name_j))
                if time_i == time_j and ((name_j, name_i) not in mag[0] + mag[1]):
                    mag[1].append((name_i, name_j))
    #第二阶段： 筛选条件集
    # cg_upate, cross_time_p_matrix ,cross_time_condition_sets= SkeletonDiscovery_Time.update_graph_with_cross_time_ci_2(
    #     cg=cg,
    #     data=data,
    #     alpha=alpha,
    #     indep_test=indep_test,
    #     node_names=node_names,
    #     p_matrix=p_matrix
    # )
    mag_pre = [list(), list()]  # directed edges and bidirected edges
    d = len(cg_upate.G.nodes)
    for i in range(d):
        for j in range(d):
            if cg_upate.G.graph[i, j] != 0 and cg_upate.G.graph[j, i] != 0:
                name_i, time_i = cg_upate.G.nodes[i].name, int(cg_upate.G.nodes[i].name.split('_')[1])
                name_j, time_j = cg_upate.G.nodes[j].name, int(cg_upate.G.nodes[j].name.split('_')[1])
                if time_i < time_j:
                    mag_pre[0].append((name_i, name_j))
                if time_i == time_j and ((name_j, name_i) not in mag_pre[0] + mag_pre[1]):
                    mag_pre[1].append((name_i, name_j))
    return mag, mag_pre

def pc_time_0(
        data: ndarray,
        alpha=0.05,
        indep_test=str,
        stable: bool = True,
        background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = True,
        node_names: List[str] | None = None,
        **kwargs
):
    """
    Implement PC_skeleton_time algorithm
    Args: data, a numpy array with shape (n,d), n is sample size, d is vertex number
          alpha, desired FDR level
          indep_test, string, choose from ['info_test','fisherz','kci','d_separation',...]
    Return:
         mag = [list_of_direct_edges, list_of_bidirected_edges]
         list_of_direct_edges = [(X1_0,X3_3), (X4_0,X5_3), ...] each tuple is always directed
    """
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    msg = 'The PC_time algorithm should be used very carefully. \n\n' + \
          'We use the following rules to decide sepset(a,b): \n' + \
          '1. subtime(a)=subtime(b)=0: delete the edge without testing (for simulation, there is always no edge; for real-world, we dont have data at -k-1 and any detected edge can be unreliable) \n' + \
          '2. subtime(a)=0,subtime(b)=k+1 (and vice versa): search sepset in subtime=0 (actually unnecessary for simulation data) \n' + \
          '3. subtime(a)=subtime(b)=k+1: search sepset in subtime=0 \n'
    # warnings.warn(msg)

    ind_test = CIT(data, indep_test, **kwargs)
    # #第一阶段：条件及为空 得到相关矩阵
    cg, p_matrix=SkeletonDiscovery_Time.skeleton_discovery_selected_pairs_with_pmatrix(data, alpha, ind_test, stable,
                                                                                      node_names=node_names)
    mag = [list(), list()]  # directed edges and bidirected edges
    d = len(cg.G.nodes)
    for i in range(d):
        for j in range(d):
            if cg.G.graph[i, j] != 0 and cg.G.graph[j, i] != 0:
                name_i, time_i = cg.G.nodes[i].name, int(cg.G.nodes[i].name.split('_')[1])
                name_j, time_j = cg.G.nodes[j].name, int(cg.G.nodes[j].name.split('_')[1])
                if time_i < time_j:
                    mag[0].append((name_i, name_j))
                if time_i == time_j and ((name_j, name_i) not in mag[0] + mag[1]):
                    mag[1].append((name_i, name_j))
    return mag
def inject_mag_to_dataframe(data, coeff=0.5, inplace=False):

    df = pd.DataFrame(data.observation)

    edges = [(f"{u}_0", f"{v}_{data.k+1}") for u, v in data.sdag.edges]
    for source, target in edges:
        df[source] = coeff * df[target] + (1 - coeff) * df[source]
    for source, target in data.mag_ins[1]:
        t_source = get_time_index(source)
        t_target = get_time_index(target)
        if t_source == 0 or t_target == 0:
            continue

        if source not in df.columns or target not in df.columns:
            raise ValueError(f" {source} or {target}not")

        df[target] = coeff * df[source] + (1 - coeff) * df[target]


    if inplace:
        data[:] = df.to_numpy()
        return None
    else:
        return df.to_numpy()

def get_time_index(var_name):

    match = re.search(r'_(\d+)$', var_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"error: {var_name}")