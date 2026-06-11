# Operation functions on nx.DiGraphs

import networkx as nx
import numpy as np
import random

def erdoes_renyi(d,p):
    '''
    erdoes renyi model for adjmatrix, adj[i,j]=1 => i->j, each edge ~ B(p)
    '''
    np.random.seed(137)
    fully_connect = np.triu(np.ones((d,d)),k=1)
    rand_mask = np.random.binomial(n=1,p=p,size=d**2).reshape(d,d)
    adjmatrix = fully_connect*rand_mask
    return adjmatrix


import numpy as np
import networkx as nx

def generate_dag_with_forward_backward_fork(n_vars: int, edge_prob: float, seed: int = None):
    """
    生成一个随机 DAG，包含至少一个叉结构 B→A 和 B→C，满足 A < B < C。

    参数：
        n_vars: 节点数量
        edge_prob: 添加边的概率
        seed: 随机种子

    返回：
        adj: 邻接矩阵 (n_vars x n_vars)，包含至少一个合法的“向前+向后”叉结构
    """

    #np.random.seed(250)
    if seed is not None:
        np.random.seed(seed)

    while True:
        # 初始化邻接矩阵
        adj = np.zeros((n_vars, n_vars), dtype=int)

        # 随机添加边（无自环）
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and np.random.rand() < edge_prob:
                    adj[i, j] = 1

        # 是否是 DAG？
        G = nx.DiGraph(adj)
        if not nx.is_directed_acyclic_graph(G):
            continue

        # 检查是否存在满足 A < B < C 的叉结构：B → A 且 B → C
        found_valid_fork = False
        for B in range(1, n_vars - 1):  # B不能是第一个或最后一个节点
            for A in range(0, B):
                for C in range(B + 1, n_vars):
                    if adj[B, A] == 1 and adj[B, C] == 1:
                        found_valid_fork = True
                        break
                if found_valid_fork:
                    break
            if found_valid_fork:
                break

        if found_valid_fork:
            return adj


# two test summary dags
adj0 = np.array([[0,1,0,1],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
adj1 = np.array([[0,1,0,1],[0,0,1,1],[0,0,0,1],[0,0,0,0]])


def adj_to_dag(adj):
    '''
    adj matrix -> nx.DiGraph, X1,...,Xd
    '''
    d = adj.shape[0]
    try:
        dag = nx.from_numpy_matrix(adj,create_using=nx.DiGraph)
    except AttributeError:
        dag = nx.DiGraph(np.array(adj)) # for networkx>3.0
        
    mapping = dict(zip(range(d),['X{}'.format(ind+1) for ind in range(d)]))
    dag = nx.relabel_nodes(dag, mapping)
    return dag
def adj_to_dag_edge_types(adj):
    '''
    给每条边随机选择类型，'lagged', 'instant', 'both'
    adj matrix -> nx.DiGraph, X1,...,Xd
    '''
    random.seed(549)
    d = adj.shape[0]
    try:
        dag = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
    except AttributeError:
        dag = nx.DiGraph(np.array(adj))  # for networkx>3.0
    # 重命名节点为 X1, X2, ...
    mapping = dict(zip(range(d), ['X{}'.format(ind + 1) for ind in range(d)]))
    dag = nx.relabel_nodes(dag, mapping)
    # 给每条边赋予随机的因果类型标签
    edge_types = ['lagged', 'instant', 'both']
    for u, v in dag.edges():
        dag[u][v]['edge_type'] = random.choice(edge_types)

    return dag
def get_parents(dag, node):
    '''
    node -> list of nodes
    pa is the parent of node i.i.f. (pa,node) in edges and (node,pa) not in edges
    '''
    assert node in dag.nodes, '{} not in dag.'.format(node)
    parents = list()
    for pa in dag.nodes:
        if (pa, node) in dag.edges and (node, pa) not in dag.edges:
            parents.append(pa)
    return parents

def _get_ancestors(dag,node,ancestors,first_level=False):
    '''
    recursive approach to collect all ancestors
    call by _get_ancestors(dag,node,[],True)
    '''
    assert node in dag.nodes, '{} not in dag.'.format(node)
    if node in ancestors:
        return
    
    if not first_level:
        ancestors.append(node)
    
    parents = get_parents(dag,node)
    
    if parents:
        for parent in parents:
            _get_ancestors(dag,parent,ancestors)
            
def get_ancestors(dag,node):
    '''
    wrapper for ancestor finding
    '''
    ancestors = list()
    _get_ancestors(dag,node,ancestors,True)
    return ancestors

def get_children(mag, node):
    '''
    node -> list of nodes; Note the input is a MAG instead of a DAG
    ch is the child of node i.i.f. (node,ch) is in edges and (ch,node) not in edges
    '''
    children = list()
    for diedge in mag[0]:
        start,end = diedge[0].split('_')[0],diedge[1].split('_')[0]
        if start==node and end!=node:
            children.append(end)
    return children


def _get_descendants(mag, node, descendants, first_level=False):
    '''
    recursive approach to collect all descendants; Note the input is a MAG instead of a DAG
    call by _get_descendants(dag,node,[],True)
    '''
    if node in descendants:
        return

    if not first_level:
        descendants.append(node)

    children = get_children(mag, node)

    if children:
        for child in children:
            _get_descendants(mag, child, descendants)


def get_descendants(mag, node):
    '''
    wrapper for descendant finding; Note the input is a MAG instead of a DAG
    '''
    descendants = list()
    _get_descendants(mag, node, descendants, True)
    return descendants

def is_direct(dag,path):
    '''
    whether the path is a direced one in the dag; we defaultly also check whether the path is a connected one
    检查给定的路径 path 是否是一个在 DAG（有向无环图）中的直接路径。直接路径的定义是：路径中的每对相邻节点之间都有一条有向边，且方向与路径的方向一致。
    '''
    for start,end in [path[i:i+2] for i in range(len(path) - 1)]:
        if (start,end) in dag.edges:
            continue
        elif (end,start) in dag.edges:
            return False
        else:
            raise ValueError('{}-{} not in dag.'.format(start,end))
    return True

def len_path(path):
    '''
    the length of a path [start,V1,...,Vn,end] is defined as n
    '''
    return len(path)-2

def all_direct_paths(dag, start, end, maxlen):
    '''
    Return all directed path from start to end with 0<len<=maxlen
    - shortpaths contain those with 0<len<=maxlen-1
    - longpaths contain those with exactly len=maxlen
    '''
    assert (start, end) in dag.edges, '{}->{} not in dag'.format(start, end)
    shortpaths = list()
    longpaths = list()
    #cutoff = maxlen + 1
    for path in nx.all_simple_paths(dag, source=start, target=end):
       # print(path)
        if is_direct(dag, path):
            length = len_path(path) #路径长度被定义为中间节点数
            if 0 < length and length <= maxlen - 1:
            #if length <= maxlen - 1:
                shortpaths.append(path)
            elif length == maxlen:
                longpaths.append(path)

    return shortpaths, longpaths


def all_direct_confound_paths(dag, start, end, maxlen):
    '''
    Return all directed path from start to end with 0<=len<=maxlen
    this is for confounding structure detection
    '''
    paths = list()
    for path in nx.all_simple_paths(dag, source=start, target=end, cutoff=maxlen + 1):
        if is_direct(dag, path):
            paths.append(path)
    return paths


def all_confound_structure(dag, start, end, maxr, maxq):
    '''
    找到start、end之间所有的混杂
    Return a list, containing all (r,q)-confounding structure between start and end
    0<=r<=maxr; 0<=q<=maxq; the default value for maxr/maxq should be k-1
    '''
    common_ancs = set(get_ancestors(dag, start)) & set(get_ancestors(dag, end))
    confounds = list()
    for anc in common_ancs:
        # directed path from common anc to start/end
        to_start = all_direct_confound_paths(dag, anc, start, maxr) #从共同祖先到 start 的直接路径
        _to_end = all_direct_confound_paths(dag, anc, end, maxq) #从共同祖先到 end 的直接路径

        # start should not in any to_end path
        to_end = list()
        for path in _to_end:
            if start not in path:
                to_end.append(path)
        if len(to_end) > 0 and len(to_start) > 0:
            confound = {'anc': anc, 'to_start': to_start, 'to_end': to_end}
            confounds.append(confound)
    return confounds



# Graph transforms
def sdag_to_ftdag_edge_types(sdag, k):
    '''
    从带因果类型边标签的 summary DAG 构造 full subtime DAG
    支持 edge_type: 'lagged', 'instant', 'both'
    '''
    ftdag = nx.DiGraph()

    # Step 1: 创建所有时间点的节点
    variables = list(sdag.nodes)
    time_nodes = ['{}_{}'.format(var, t) for var in variables for t in range(k + 2)]
    ftdag.add_nodes_from(time_nodes)

    # Step 2: 添加边，根据 edge_type 类型展开
    # add edges
    for _node in list(ftdag.nodes):
        var, time = _node.split('_')
        time = int(time)

        spa = get_parents(sdag, var)

        for _spa in spa:
            edge_data = sdag.get_edge_data(_spa, var)
            if edge_data is None:
                continue
            etype = edge_data.get("edge_type", "lagged")

            # 瞬时因果边 parent(t) -> var(t)，允许 t=0
            if etype in ['instant', 'both']:
                parent_node = f"{_spa}_{time}"
                ftdag.add_edge(parent_node, _node)

            # 滞后因果边 parent(t-1) -> var(t)，仅限 t>0
            if time > 0 and etype in ['lagged', 'both']:
                parent_node = f"{_spa}_{time - 1}"
                ftdag.add_edge(parent_node, _node)

        # 自身滞后边 var(t-1) -> var(t)，仅限 t>0
        if time > 0:
            prev_self = f"{var}_{time - 1}"
            ftdag.add_edge(prev_self, _node)

    return ftdag
def sdag_to_ftdag(sdag,k):
    '''
    summary dag -> full subtime dag; k the under sampling factor, t=0,1,...,k+1
    '''
    ftdag = nx.DiGraph()
    
    # add nodes
    nodes = list()
    for _node in sdag.nodes:
        nodes += ['{}_{}'.format(_node,t) for t in range(k+2)]
    ftdag.add_nodes_from(nodes)
    
    # add edges
    for _node in ftdag.nodes:
        # variable name and subtime step
        var,time = _node.split('_'); time = int(time)
        if time==0:
            continue
        # parents in summary dag
        spa = get_parents(sdag,var)
        # parents in fulltime dag
        ftpa = ['{}_{}'.format(var,time-1)]
        for _spa in spa:
            ftpa.append('{}_{}'.format(_spa,time-1))

        for _ftpa in ftpa:
            ftdag.add_edge(_ftpa,_node)    

    return ftdag


def ftdag_to_mag(ftdag,d,k):
    '''
    from the ground-truth ftdag, return the gt-mag (directed-edges and bidirected-edges)
    '''
    # directed edges
    causes = ['X{}_0'.format(ind+1) for ind in range(d)]
    effects = ['X{}_{}'.format(ind+1,k+1) for ind in range(d)]
    diedges = list()
    for cause in causes:
        for effect in effects:
            if cause in get_ancestors(ftdag,effect):
                diedges.append((cause,effect))
    # bidirected edges            
    biedges = list()
    for node1 in effects:
        for node2 in effects:
            if node1==node2:
                continue
            anc1 = get_ancestors(ftdag,node1)
            anc2 = get_ancestors(ftdag,node2)
            intersaction = set(anc1)&set(anc2)
            # whether a *latent* confounder
            flag = False 
            for _node in intersaction:
                _,time = _node.split('_'); time = int(time)
                if time>0:
                    flag = True
                    break
            if flag and (node1,node2) not in biedges and (node2,node1) not in biedges:
                biedges.append((node1,node2))
    return diedges,biedges


def mag_to_skeleton(mag):
    '''
    MAG (nx.DiGraph with directed and bidirected edges) -> its skeleton (list of undirected** edges)
    '''
    skeleton = list()
    for edge in mag.edges:
        begin,end = edge
        if (end,begin) not in mag.edges:
            # directed edge
            skeleton.append(edge)
        else:
            # bidirected edge
            if (end,begin) not in skeleton:
                skeleton.append(edge)
    return skeleton


def detect_dicycle_in_mag(mag):
    '''
    Arg: mag[0], a list of tuples with directed edges, e.g., [('X1_0', 'X1_3'),('X3_0', 'X4_3'),...]
    '''
    tmp = nx.DiGraph()
    for edge in mag[0]:
        a,b = edge
        name_a,name_b = a.split('_')[0],b.split('_')[0]
        if name_a!=name_b:
            tmp.add_edge(name_a,name_b)
    
    cycle = None
    try:
        cycle = nx.find_cycle(tmp, orientation="original")
    except:
        pass

    if cycle is not None:
        raise Exception('Directed cycle detected in MAG: {}'.format(cycle))

def search_set_M(mag,node_names,A,B):
    '''
    Search for the set M (Neurips version)
    For A(t) and B(t+k), the set M contain A and any vertex Mi(t+1) Mi neq B
      such that A(t)->Mi(t+k) in the MAG and Mi is not B's descendant
    Args: mag, [list_of_directed_edges, list_of_bidirected_edges]
          node_names, [X1,X2,...,Xd]
          A, B two variables of interest
          在最大祖先图（MAG）中搜索集合 Mset，用于阻断从 A(t) 到 B(t+k) 的间接因果路径
    '''
    k = mag[0][0][1].split('_')[1] #从 MAG 的第一条有向边的第二个节点中提取时间步 k
    Mset = {A}
    for node in node_names:
        flag1 = node!= B #检查条件 1：Mi 不是 B
        flag2 = (A+'_0',node+'_{}'.format(k)) in mag[0] #检查条件 2：在 MAG 中存在从 A(t) 到 Mi(t+k) 的有向边
        flag3 = node not in get_descendants(mag,B) #检查条件 3：Mi 不是 B 的后代

        if flag1 and flag2 and flag3:
            Mset.add(node)
    return Mset
        
def get_set_M(mag,node_names,A,B):
    '''
    Search for the set M (Annals version)
    Args: mag, [list_of_directed_edges, list_of_bidirected_edges]
          node_names, [X1,X2,...,Xd]
          A, B two variables of interest
    '''
    k = mag[0][0][1].split('_')[1]
    Mset = {A}
    for node in node_names:
        flag1 = node!=B
        flag2 = (A+'_0', node+'_'+k) in mag[0]
        flag3 = (node+'_0', B+'_'+k) in mag[0]

        if flag1 and flag2 and flag3:
            Mset.add(node)
            
    need_more_search = True
    while need_more_search:
        need_more_search = False
        for node in node_names:
            flag1 = node not in Mset
            flag2 = node!=B
            flag3 = (A+'_0', node+'_'+k) in mag[0]
            flag4 = False 
            for M in Mset:
                if (node+'_0', M+'_'+k) in mag[0]:
                    flag4 = True
                    break

            if flag1 and flag2 and flag3 and flag4:
                Mset.add(node)
                need_more_search = True
    #Mset = {M+'_1' for M in Mset}
    return Mset

def get_set_S(mag,node_names,Mset,A,B):
    '''
    Search for the set S (both Annals and Neurips version)
    Args: mag, [list_of_directed_edges, list_of_bidirected_edges]
          node_names, [X1,X2,...,Xd]
          Mset, the searched M set
          A, B, two variables of interest
          在最大祖先图（MAG）中搜索集合 Sset，用于阻断从 A(t) 到 B(t+k) 的后门路径（backdoor paths）。集合 Sset 包含所有满足特定条件的混杂变量。
    '''
    k = mag[0][0][1].split('_')[1]
    Sset = set()

    for node in node_names:
        flag1 = node!=A  #检查条件 1：S 不是 A
        flag2 = (node+'_0', B+'_'+k) in mag[0] #检查条件 2：在 MAG 中存在从 S(t) 到 B(t+k) 的有向边
        flag3 = False #检查条件 3：在 MAG 中存在从 S(t) 到 M(t+k) 的有向边
        for M in Mset:
            if (node+'_0', M+'_'+k) in mag[0]:
                flag3 = True
                break

        if flag1 and (flag2 or flag3):
            Sset.add(node)
    Sset = {S+'_0' for S in Sset}
    return Sset


def mag_to_pddag(mag, k, static_node_names):
    '''
    Step-2 of Alg. 1 of NeurIPS paper: construct the PD-DAG from MAG
    Args: - mag, a list of [list_of_diedge, list_of_bidedges]
          - k, subsampling_factor-1, e.g. k=1 means A(0), A(2), A(4) can be observed
    '''
    pddag = nx.DiGraph()
    pddag.add_nodes_from(static_node_names)

    for cause in pddag.nodes:
        for effect in pddag.nodes:
            diedge = ('{}_0'.format(cause), '{}_{}'.format(effect, k + 1))
            bidedge = ('{}_{}'.format(cause, k + 1), '{}_{}'.format(effect, k + 1))
            if diedge in mag[0] and (bidedge in mag[1] or (bidedge[1], bidedge[0]) in mag[1]):
                # A(t)->B(t+k) and A(t+k)<->B(t+k)
                pddag.add_edge(cause, effect, style='-->')

    # check whether there is directed cycles in pddag (which shouldn't happen)
    try:
        dicycle = nx.find_cycle(pddag, orientation='original')
        #print('Warning: directed cycles detected in pc-dag', dicycle)
    except nx.exception.NetworkXNoCycle:
        pass
    return pddag


def mag_to_pddag_new(data, mag, k, static_node_names):
    pddag = nx.DiGraph()
    pddag.add_nodes_from(static_node_names)

    for cause in static_node_names:
        for effect in static_node_names:
            if cause == effect:
                continue

            A_t0 = f'{cause}_0'
            A_tk = f'{cause}_{k+1}'
            B_t0 = f'{effect}_0'
            B_tk = f'{effect}_{k+1}'

            diedge = (A_t0, B_tk)
            reverse_diedge = (B_t0, A_tk)
            bidedge_tk = (A_tk, B_tk)
            bidedge_tk_rev = (B_tk, A_tk)
            bidedge_t0 = (A_t0, B_t0)
            bidedge_t0_rev = (B_t0, A_t0)

            # Rule 1: 滞后 + 瞬时边
            if diedge in mag[0] and (bidedge_tk in mag[1] or bidedge_tk_rev in mag[1]):
                pddag.add_edge(cause, effect, style='->')

            # Rule 2: 仅瞬时双向边（满足条件）
            elif (
                (bidedge_tk in mag[1] or bidedge_tk_rev in mag[1]) and
                (bidedge_t0 in mag[1] or bidedge_t0_rev in mag[1]) and
                (diedge not in mag[0]) and
                (reverse_diedge not in mag[0])
            ):
                pddag.add_edge(cause, effect, style='->')
    sedges = set(data.sdag.edges())
    for edge in list(pddag.edges.data()):
        start, end, style = edge;
        if (start,end) not in sedges :
            pddag.remove_edge(start, end)
    try:
        dicycle = nx.find_cycle(pddag, orientation='original')
        #print('Warning: directed cycles detected in pd-dag:', dicycle)
    except nx.exception.NetworkXNoCycle:
        pass

    return pddag



def is_certain(pddag):
    '''
    Judge whether the PD-DAG is fully assured; i.e. all --> edges are removed or turned to ->
    '''
    for edge in pddag.edges.data():
        style = edge[-1]['style']
        if style == '-->':
            return False
    return True