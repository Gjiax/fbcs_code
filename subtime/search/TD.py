import networkx as nx
from subtime.graph.NxGraph import detect_dicycle_in_mag,mag_to_pddag,mag_to_pddag_new,is_certain,all_confound_structure,all_direct_paths,search_set_M,get_set_M,get_set_S
from subtime.search.PC import pc_time,pc_time_0,inject_mag_to_dataframe
from subtime.utils.BY_procedure import Benjamini_Yekutieli
from subtime.utils.cit import CIT
from subtime.utils.ProxyTest.proxy_test import ProxyCITest
from evaluation.evaluation import *
from typing import List
from numpy import ndarray
from subtime.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

def td(
        data,
        indep_test=str,
        proxy_test=str,
        node_names: List[str] | None = None,
        subsampling_factor=int,
        alpha: float = 0.05,
        stable: bool = True,
        background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = False,
        proxy_ratio=1,
        **kwargs):
    """
    Implement the Time-Discovery algorithm
    Args: data: observation data, nd.array with shape (n,d), n-sample size, d-node numbers
          indep_test: method for CI Test, choose from [info_test, kci, d_separation, fisherz, ...]
                      if choose d_separation, please provide the gt-ftdag by specifying "true_dag = data_gen.ftdag"
          proxy_test: method for Proxy Test, choose from [proxy_test, d_separation]
          node_names: nodes names of data, e.g., [X1_0,X1_5,X2_0,X2_5...]
          subsampling_factor: k-1, e.g. sub_factor=1 means A(0), A(2), A(4) can be observed
          alpha (default 0.1): level of FDR
          stable: whether use stable-PC
          proxy_ratio (default 1.0): proxy-test conditioning on the observed confounders

          ProxyTest-related args, levelx,levelw,levely,ratio, should be specified in **kwargs
    Example:
          adj = erdoes_renyi(5,0.3)
          data_gen = TimeData(adj,k=1,n=200)
          mag,sdag = td(data = data_gen.observation.values,node_names = data_gen.observation.columns,
                        indep_test = 'd_separation', # set to 'fihserz','kci','info_test', ... if you don't want Oracle test
                        proxy_test = 'd_separation', # set to 'proxy_test' if you don't want Oracle test
                        alpha = 0.05, subsampling_factor=data_gen.k,
                        true_dag = data_gen.ftdag # if you need Oracle CI/Proxy Test)
          # Evaluation
          prec,rec = precision_skeleton(mag,data_gen.mag_ske),recall_skeleton(mag,data_gen.mag_ske)
          print('Precision: {}, Recall: {} of MAG'.format(prec,rec))
          prec,rec = precision(sdag.edges,data_gen.sdag.edges), recall(sdag.edges,data_gen.sdag.edges)
          print('Precision: {}, Recall: {} of sDAG'.format(prec,rec))
    """
    # 1. recover MAG
    kwargs['oracle_citest_node_names'] = node_names
    #mag_old = pc_time_0(data.observation.values, alpha=alpha, node_names=node_names, stable=stable,
                  # background_knowledge=background_knowledge, indep_test=indep_test,
                  # show_progress=show_progress, **kwargs)
    mag_old = pc_time_0(data.observation.values, alpha=alpha, node_names=node_names, stable=stable,
                        background_knowledge=background_knowledge, indep_test=indep_test,
                        show_progress=show_progress, **kwargs)
    #detect_dicycle_in_mag(mag)
    data_new =inject_mag_to_dataframe(data, coeff=0.3, inplace=False)
    mag, mag_pre = pc_time(data_new, alpha=alpha, node_names=node_names, stable=stable,
                           background_knowledge=background_knowledge, indep_test=indep_test,
                           show_progress=show_progress, **kwargs)
    ctbcer = evaluate_bidirectional_elimination(mag[0], mag_pre[0])
    print(f"BCER: {ctbcer:.4f}")
    mag_ske = mag_pre[0] + mag_pre[1]
    # mag_ske=[('X1_0', 'X1_2'), ('X1_0', 'X2_2'), ('X1_0', 'X3_2'), ('X2_0', 'X2_2'), ('X2_0', 'X3_2'), ('X3_0', 'X3_2'),
    #  ('X4_0', 'X3_2'), ('X4_0', 'X4_2'), ('X4_0', 'X5_2'), ('X5_0', 'X5_2'), ('X2_0', 'X3_0'), ('X1_2', 'X3_2'),('X2_2', 'X3_2'),
    #  ('X3_2', 'X4_2'),('X3_2', 'X5_2'), ('X4_0', 'X5_0'), ('X4_2', 'X5_2')]
    # 2. construct PD-DAG from MAG 先确定能确定的边
    static_node_names = list()
    for name in [name.split('_')[0] for name in node_names]: #取 _ 前面的部分
        if name not in static_node_names: #确保 static_node_names 不包含重复元素
            static_node_names.append(name)
    #pddag = mag_to_pddag(mag_pre, subsampling_factor, static_node_names)
    pddag = mag_to_pddag_new(data, mag_pre, subsampling_factor, static_node_names)

    # 3. initiate proxy searching (step-3 of alg.1) 从不确定的边中找确定的边或删除
    proxytester = ProxyCITest(data_new, node_names) #只创建了一个实例
    while not is_certain(pddag):
        # rule-a
        pddag = rulea(pddag, subsampling_factor)
        if is_certain(pddag):
            break
        #rule-b
        pddag = ruleb(pddag, mag_pre, proxy_test, proxytester, proxy_ratio, alpha, **kwargs)
    return mag_ske, pddag, ctbcer



def rulea(pddag, k):

    for edge in pddag.edges.data():
        start, end, style = edge;
        style = style['style']
        shorts, longs = all_direct_paths(pddag, start, end, k) # 从 start 到 end 的直接路径：shorts：长度小于等于 k 的路径。longs：长度大于 k 的路径。
        confounds = all_confound_structure(pddag, start, end, k - 1, k - 1) # 计算 从 start 到 end 可能存在的混杂因子（Confounder）。

        rulea = len(shorts) > 0 or (len(longs) > 0 and len(confounds) > 0)

        if not rulea:
            edge[2]['style'] = '->'
    return pddag


def ruleb(pddag, mag, proxy_test, proxytester, proxy_ratio, alpha, **kwargs):

    minMsize = len(pddag.nodes)
    minEdge, minMset, minSset = None, None, None

    for edge in pddag.edges.data():
        start, end, style = edge;
        style = style['style']
        if style == '->':
            continue
        Mset = search_set_M(mag, pddag.nodes, start, end)
        Sset = get_set_S(mag, pddag.nodes, Mset, start, end)
        if len(Mset) < minMsize:
            minMsize = len(Mset)
            minEdge = edge
            minMset = Mset
            minSset = Sset

    A, B, _ = minEdge
    k = mag[0][0][1].split('_')[1]

    if proxy_test == 'd_separation':
        minMset = {M + '_1' for M in minMset}
        pvalue = nx.d_separated(kwargs['true_dag'], {A + '_0'}, {B + '_' + k}, minMset.union(minSset))
    else:
        minMset = {M + '_' + k for M in minMset}
        pvalue = proxytester(X=A + '_0', Y=B + '_' + k, W=list(minMset), C=list(minSset), ratio=proxy_ratio)

    if pvalue > alpha:
        pddag.remove_edge(A, B)
    else:
        minEdge[-1]['style'] = '->'
    return pddag


def td_fdr(
        data: ndarray,
        indep_test=str,
        proxy_test=str,
        node_names: List[str] | None = None,
        alpha: float = 0.1,
        stable: bool = True,
        background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = False,
        proxy_ratio = 1,  
        **kwargs):
    """
    Implement the Time-Discovery algorithm with FDR control
    Args: data: observation data, nd.array with shape (n,d), n-sample size, d-node numbers
          indep_test: method for CI Test, choose from [info_test, kci, d_separation, fisherz, ...]
                      if choose d_separation, please provide the gt-ftdag by specifying "true_dag = data_gen.ftdag"
          proxy_test: method for Proxy Test, choose from [proxy_test, d_separation]
          node_names: nodes names of data, e.g., [X1_0,X1_5,X2_0,X2_5...]
          alpha (default 0.1): level of FDR
          stable: whether use stable-PC
          proxy_ratio (default 1.0): proxy-test conditioning on the observed confounders

          ProxyTest-related args, levelx,levelw,levely,ratio, should be specified in **kwargs
    Example:
          adj = erdoes_renyi(5,0.3)
          data_gen = TimeData(adj,3,600)
          mag,sdag = td(data = data_gen.observation.values,
                        indep_test = 'info_test'
                        proxy_test = 'proxy_test'
                        node_names = data_gen.observation.columns
                        alpha = 0.05,
                        true_dag = data_gen.ftdag) # if you need Oracle CI/Proxy Test)
          # Evaluation
          prec,rec = precision_skeleton(mag,data_gen.mag_ske),recall_skeleton(mag,data_gen.mag_ske)
          print('Precision: {}, Recall: {} of MAG'.format(prec,rec))
          prec,rec = precision(sdag.edges,data_gen.sdag.edges), recall(sdag.edges,data_gen.sdag.edges)
          print('Precision: {}, Recall: {} of sDAG'.format(prec,rec))
    """
    kwargs['oracle_citest_node_names'] = node_names

    if 'true_mag' in kwargs.keys():
        mag = kwargs['true_mag'] # you can also specify indep_test='d_separation' and run the pd_fdr_time() to obtain the mag.
        # we tried and found the results are exactly identical. so here we directly used the gt-mag to save subtime.
    else:
        mag = pc_fdr_time(data, alpha=alpha,
                          node_names=node_names, stable=stable, background_knowledge=background_knowledge,
                          indep_test=indep_test,
                          show_progress=show_progress, **kwargs)

    detect_dicycle_in_mag(mag)
    mag_ske = mag[0] + mag[1]

    static_node_names = list()
    for name in [name.split('_')[0] for name in node_names]:
        if name not in static_node_names:
            static_node_names.append(name)

    proxytest = ProxyCITest(data, node_names)
    pvalues = list()
    k = mag[0][0][1].split('_')[1]

    for A in static_node_names:
        for B in static_node_names:
            if A != B:
                flag1 = (A + '_0', B + '_' + k) in mag[0]
                flag2 = (A + '_' + k, B + '_' + k) in mag[1]
                flag3 = (B + '_' + k, A + '_' + k) in mag[1]
                if flag1 and (flag2 or flag3):
                    Mset = get_set_M(mag, static_node_names, A, B)
                    Sset = get_set_S(mag, static_node_names, Mset, A, B)

                    # At ind Bt+k | Mt+1 cup St
                    if proxy_test == 'd_separation':
                        Mset = {M + '_1' for M in Mset}
                        pvalue = nx.d_separated(kwargs['true_dag'], {A + '_0'}, {B + '_' + k}, Mset.union(Sset))
                    else:
                        Mset = {M + '_' + k for M in Mset}
                        pvalue = proxytest(X=A + '_0', Y=B + '_' + k, W=list(Mset), C=list(Sset), ratio=proxy_ratio)
                    pvalues.append(((A, B), pvalue))

    reject_id, _ = Benjamini_Yekutieli(pvalues, alpha)
    sdag = nx.DiGraph()
    for edge in reject_id:
        sdag.add_edge(edge[0], edge[1])
    return mag_ske, sdag

def proxy_ci(pddag, mag, proxytester, proxy_ratio=1.0, alpha=0.05):
    k = mag[0][0][1].split('_')[1]
    for edge in pddag.edges.data():
        start, end, style = edge;
        if style['style'] == '->':
            continue
        Mset = search_set_M(mag, pddag.nodes, start, end)
        Sset = get_set_S(mag, pddag.nodes, Mset, start, end)
        Mset = {M + '_' + k for M in Mset}
        pval = proxytester(X=start + '_0', Y=end + '_' + k, W=list(Mset), C=list(Sset), ratio=proxy_ratio)
        if pval > alpha:
            pddag.remove_edge(start, end)
        else:
            style['style'] = '->'
    return  pddag
