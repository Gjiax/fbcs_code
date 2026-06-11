from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy import ndarray
from typing import List
from tqdm import tqdm

from subtime.graph.GraphClass import CausalGraph
from subtime.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from subtime.utils.PCUtils.Helper import append_value
from subtime.utils.cit import CIT


def skeleton_discovery_time(
    data: ndarray,
    alpha: float,
    indep_test: CIT,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None,
    maxFanIn: int = -1,
) -> CausalGraph:
    """
    Perform skeleton discovery for subtime-series MAG
    We use the following rules to decide sepset(a,b):
        1. subtime(a)=subtime(b)=0: delete the edge without testing (for simulation, there is always no edge;
            for real-world, we dont have data at -k-1 and any detected edge can be unreliable)
        2. subtime(a)=0,subtime(b)=k+1 (and vice versa): search sepset in subtime=0 (actually unnecessary for simulation data)
        3. subtime(a)=subtime(b)=k+1: search sepset in subtime=0
    The validity of the above rules are proved in Claim-1 in the Annals paper

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : class CIT, the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]

    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)
    # the 'depth' argument is the same as the 'n' argument in PC algorithm
    # when depth=1, mean searching for sep-set with cardinality=1, ...
    # so, we can set a threshold for max-depth, namely maxFanIn, to control the searching range and reduce comutation cost
    depth = -1
    if show_progress:
        pbar = tqdm(total=no_of_var)
    while cg.max_degree() - 1 > depth:
        depth += 1
        # custom argument 10-11
        if maxFanIn != -1 and depth>maxFanIn:
            break
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
                pbar.set_description(f'Depth={depth}, working on node {x}')

            time_x = int(node_names[x].split('_')[1])
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue

            for y in Neigh_x:
                time_y = int(node_names[y].split('_')[1])
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                if time_x == 0 and time_y == 0:
                    knowledge_ban_edge = True

                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))  # after all conditioning sets at
                        edge_removal.append((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    Stime = [int(node_names[node].split('_')[1]) for node in S]
                    if not (len(Stime) == 0 or sum(Stime) == 0):
                        continue  # only search sepset in subtime=0

                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            break
                        else:
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            for s in S:
                                sepsets.add(s)
                    else:
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress:
        pbar.close()

    return cg

import numpy as np
from typing import Dict, Tuple, List
def skeleton_discovery_selected_pairs_with_pmatrix(
    data: ndarray,
    alpha: float,
    indep_test: CIT,
    stable: bool = True,
    verbose: bool = False,
    node_names: List[str] | None = None,
) -> Tuple[CausalGraph, np.ndarray]:
    """
    Perform skeleton discovery for selected pairs and record p-values.

    Returns
    -------
    cg : CausalGraph
    p_matrix : ndarray, shape (n_features, n_features)
        p_matrix[x, y] = p-value of test x ⫫ y | []
    """
    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)

    # Initialize p-value matrix with NaN
    p_matrix = np.full((no_of_var, no_of_var), np.nan)
    k = infer_k_from_node_names(node_names) - 1
    for x in range(no_of_var):
        time_x = int(node_names[x].split('_')[1])
        Neigh_x = cg.neighbors(x)
        for y in Neigh_x:
            time_y = int(node_names[y].split('_')[1])

            # Select only valid subtime pairs
            if not is_valid_pair(time_x, time_y, k):
                continue

            # CI test with empty conditioning set
            p = cg.ci_test(x, y, [])

            # Record p-value symmetrically
            p_matrix[x, y] = p
            #p_matrix[y, x] = p

            # Edge decision
            if p > alpha:
                if verbose:
                    print(f"{x} ⫫ {y} | []  p={p:.4f} ⇒ remove edge")
                edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                if edge1:
                    cg.G.remove_edge(edge1)
                if edge2:
                    cg.G.remove_edge(edge2)
                append_value(cg.sepset, x, y, ())
                append_value(cg.sepset, y, x, ())
            else:
                if verbose:
                    print(f"{x} ⫫/ {y} | []  p={p:.4f} ⇒ keep edge")

    return cg, p_matrix

def infer_k_from_node_names(node_names: list[str]) -> int: #推断k的值
    times = sorted({int(name.split('_')[1]) for name in node_names})
    if len(times) < 2:
        raise ValueError("Not enough subtime points to infer k")
    return times[1] - times[0]

def is_valid_pair(time_x, time_y, k): #筛选变量对
    return (
        (time_x == 0 and time_y == 0) or
        (time_x == 0 and time_y == k + 1) or
        (time_x == k + 1 and time_y == k + 1)
)


def update_graph_with_cross_time_ci_1(
    cg: CausalGraph,
    data: np.ndarray,
    alpha: float,
    indep_test,
    node_names: List[str],
    p_matrix: np.ndarray,
    verbose: bool = True
) -> Tuple[CausalGraph, np.ndarray]:
    no_of_var = len(node_names)


    times = [int(name.split('_')[1]) for name in node_names]
    unique_times = sorted(list(set(times)))
    if len(unique_times) != 2:
        raise ValueError("Only support two subtime points: t=0 and t=k+1")
    t0, t1 = unique_times
    k = t1 - t0


    var_names = [name.split('_')[0] for name in node_names]
    var_set = sorted(set(var_names))
    var_num = len(var_set)


    t0_indices = [i for i, n in enumerate(node_names) if n.endswith(f"_{t0}")]
    t1_indices = [i for i, n in enumerate(node_names) if n.endswith(f"_{t1}")]


    cross_time_p_matrix = p_matrix.copy()


    cross_time_condition_sets: Dict[Tuple[int, int], List[str]] = {}

    for xi in t0_indices:
        for xj in t1_indices:

            if var_names[xi] == var_names[xj]:
                continue


            if cg.G.get_edge(cg.G.nodes[xi], cg.G.nodes[xj]) is None:
                if verbose:
                    print(f"Skip {node_names[xi]} → {node_names[xj]}: already removed in stage 1")
                continue


            cond_0 = [v for v in t0_indices if v != xi and p_matrix[xi, v] <= alpha]
            cond_1 = [v for v in t1_indices if v != xj and p_matrix[xj, v] <= alpha]
            cond_set = cond_0
            #cond_set = cond_0 + cond_1
            #cond_set = [3,4]


            # cross_time_condition_sets[(xi, xj)] = [node_names[v] for v in cond_set]


            p_val = cg.ci_test(xi, xj, cond_set)
            cross_time_p_matrix[xi, xj] = p_val
            cross_time_p_matrix[xj, xi] = p_val

            if p_val > alpha:


                edge1 = cg.G.get_edge(cg.G.nodes[xi], cg.G.nodes[xj])
                if edge1 is not None:
                    cg.G.remove_edge(edge1)
                edge2 = cg.G.get_edge(cg.G.nodes[xj], cg.G.nodes[xi])
                if edge2 is not None:
                    cg.G.remove_edge(edge2)

                cross_time_condition_sets[(xi, xj)] = [node_names[v] for v in cond_set]
                if verbose:
                    print(f"Remove edge: {node_names[xi]} → {node_names[xj]} | p={p_val:.4f} > α={alpha}")
                    print(f"    Condition set: {cross_time_condition_sets[(xi, xj)]}")
            elif verbose:
                print(f"Keep edge: {node_names[xi]} → {node_names[xj]} | p={p_val:.4f} ≤ α={alpha}")

    return cg, cross_time_p_matrix, cross_time_condition_sets


def update_graph_with_cross_time_ci_2(
    cg,
    data: np.ndarray,
    alpha: float,
    indep_test,
    node_names: List[str],
    p_matrix: np.ndarray,
    verbose: bool = True
) -> Tuple[object, np.ndarray, Dict[Tuple[int, int], List[str]]]:
    no_of_var = len(node_names)

    times = [int(name.split('_')[1]) for name in node_names]
    unique_times = sorted(list(set(times)))
    if len(unique_times) != 2:
        raise ValueError("Only support two subtime points: t=0 and t=k+1")
    t0, t1 = unique_times
    k = t1 - t0

    var_names = [name.split('_')[0] for name in node_names]
    var_set = sorted(set(var_names))
    var_num = len(var_set)

    t0_indices = [i for i, n in enumerate(node_names) if n.endswith(f"_{t0}")]
    t1_indices = [i for i, n in enumerate(node_names) if n.endswith(f"_{t1}")]

    cross_time_p_matrix = p_matrix.copy()
    cross_time_condition_sets: Dict[Tuple[int, int], List[str]] = {}

    for xi in t0_indices:
        for xj in t1_indices:
            if var_names[xi] == var_names[xj]:
                continue

            if cg.G.get_edge(cg.G.nodes[xi], cg.G.nodes[xj]) is None:
                if verbose:
                    print(f"Skip {node_names[xi]} → {node_names[xj]}: already removed in stage 1")
                continue

            cond_0 = [v for v in t0_indices if v != xi and p_matrix[xi, v] <= alpha]

            # Step 1: Test with only t0-related variables
            p_val_0 = cg.ci_test(xi, xj, cond_0)
            if p_val_0 > alpha:
                edge1 = cg.G.get_edge(cg.G.nodes[xi], cg.G.nodes[xj])
                if edge1 is not None:
                    cg.G.remove_edge(edge1)
                edge2 = cg.G.get_edge(cg.G.nodes[xj], cg.G.nodes[xi])
                if edge2 is not None:
                    cg.G.remove_edge(edge2)
                cross_time_condition_sets[(xi, xj)] = [node_names[v] for v in cond_0]
                if verbose:
                    print(f"Remove edge after step 1: {node_names[xi]} → {node_names[xj]} | p={p_val_0:.4f} > α={alpha}")
                    print(f"    Condition set (t0 only): {cross_time_condition_sets[(xi, xj)]}")
                continue

            # Step 2: Test with both t0 and t1-related variables
            cond_1 = [v for v in t1_indices if v != xj and p_matrix[xj, v] <= alpha]
            cond_set = cond_0 + cond_1
            p_val_1 = cg.ci_test(xi, xj, cond_set)
            cross_time_p_matrix[xi, xj] = p_val_1
            cross_time_p_matrix[xj, xi] = p_val_1

            if p_val_1 > alpha*2:
                edge1 = cg.G.get_edge(cg.G.nodes[xi], cg.G.nodes[xj])
                if edge1 is not None:
                    cg.G.remove_edge(edge1)
                edge2 = cg.G.get_edge(cg.G.nodes[xj], cg.G.nodes[xi])
                if edge2 is not None:
                    cg.G.remove_edge(edge2)
                cross_time_condition_sets[(xi, xj)] = [node_names[v] for v in cond_set]
                if verbose:
                    print(f"Remove edge after step 2: {node_names[xi]} → {node_names[xj]} | p={p_val_1:.4f} > α={alpha}*2")
                    print(f"    Condition set (t0 + t1): {cross_time_condition_sets[(xi, xj)]}")
            elif verbose:
                print(f"Keep edge: {node_names[xi]} → {node_names[xj]} | p={p_val_1:.4f} ≤ α={alpha}*2")

    return cg, cross_time_p_matrix, cross_time_condition_sets